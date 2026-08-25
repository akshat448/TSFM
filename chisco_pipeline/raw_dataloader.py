"""Raw (unprocessed) CHISCO .edf loading -- epochs trials directly from the
raw EDF files, skipping the authors' PREP/ICA/autoreject/1Hz-highpass
pipeline entirely, per explicit request. What CANNOT be skipped is event
detection + epoching itself: raw CHISCO .edf files are continuous ~40-90
minute multi-run recordings with no separate BIDS events.tsv, so getting
individual TRIAL arrays out of them at all requires reading the embedded
"Trigger" channel and cutting fixed-length windows around each trigger --
that is data extraction, not artifact-cleaning "preprocessing" in the
neuroscience sense, and mirrors exactly what CHISCO's own preprocessing.py
does for event detection (same trigger code 65380, same tmin/tmax windows),
just without PREP/ICA/autoreject/filtering on top.

GROUND TRUTH (verified against real S3 listing of ds005170, not assumed):
    sub-{SS}/ses-{NN}/eeg/sub-{SS}_ses-{NN}_task-imagine_run-{RR}_eeg.edf
Text labels come from a SEPARATE file, matched by run number:
    textdataset/split_data_{run_number}.xlsx  (first column = sentence per
    trigger event, in order)
No accompanying events.tsv/channels.tsv/json sidecar exists in the raw BIDS
directories on OpenNeuro -- confirmed by listing sub-01/ses-01/eeg/ directly.

Channel handling mirrors dataloader.py's derivative convention EXACTLY for
downstream compatibility (SpatialWhiteningFrontEnd, encoder, etc. don't care
which loader produced their input): 122 cortical + VEO + HEO, Trigger
dropped before anything reaches a model. Unlike dataloader.py, channel roles
here are resolved by NAME (via MNE's ch_names) rather than fixed indices,
since raw EDF channel ordering is not guaranteed identical to the
derivative pkl's array layout.

MEMORY: each .edf is ~400-600MB on disk; MNE expands this to float64 arrays
in memory (several x the file size) when preloaded for epoching. This class
loads/epochs ONE FILE AT A TIME (with a single-file cache for the common
case of sequential access), never all files simultaneously -- but a single
run can still be a few GB in memory while loaded. Do not preload multiple
CHISCORawDataloader instances' full run sets at once.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

from .dataloader import CHISCOSample

TRIGGER_EVENT_CODE = 65380
USELESS_CHANNELS = ["11", "110", "EKG", "EMG", "84", "85", "10", "111"]
EOG_CHANNELS = ["VEO", "HEO"]
TASK_WINDOWS = {"read": (0.0, 5.0), "imagine": (5.0, 8.3)}
DEFAULT_MONTAGE = Path(__file__).parent / "assets" / "montage.csv"
VOLTS_TO_MICROVOLTS = 1_000_000.0


def _parse_run_number(edf_path: Path) -> int:
    match = re.search(r"run-(\d+)", edf_path.name)
    if not match:
        raise ValueError(f"Could not parse run number from {edf_path.name}")
    return int(match.group(1))


class CHISCORawDataloader(Dataset):
    """Lazy, one-file-at-a-time dataset over raw CHISCO .edf runs.

    Parameters
    ----------
    edf_paths : Sequence[str | Path]
        Paths to `sub-*_ses-*_task-imagine_run-*_eeg.edf` files.
    task : "read" | "imagine"
        Which epoch window to extract from each trigger (0-5s post-trigger
        for "read", 5-8.3s for "imagine") -- BOTH come from the SAME raw
        file/events, matching the authors' own preprocessing.py, which
        pulls both epoch types from one set of detected trigger events.
    textdataset_dir : str | Path
        Directory containing `split_data_{run_number}.xlsx` label files.
    highpass_hz : float | None
        Default None: no filtering at all, per explicit "no preprocessing"
        request. Set a value if you later decide you DO want the same
        optional high-pass `CHISCODataloader` supports, for parity.
    """

    def __init__(
        self,
        edf_paths: Sequence[str | Path],
        task: str,
        textdataset_dir: str | Path,
        montage_path: str | Path = DEFAULT_MONTAGE,
        sample_rate_hz: float = 500.0,
        highpass_hz: float | None = None,
        filter_order: int = 4,
    ) -> None:
        if task not in TASK_WINDOWS:
            raise ValueError(f"task must be one of {list(TASK_WINDOWS)}, got {task!r}")
        self.edf_paths = [Path(p) for p in edf_paths]
        self.task = task
        self.textdataset_dir = Path(textdataset_dir)
        self.montage_path = Path(montage_path)
        self.sample_rate_hz = sample_rate_hz

        self._sos = None
        if highpass_hz is not None:
            nyquist = sample_rate_hz / 2.0
            self._sos = butter(filter_order, highpass_hz / nyquist, btype="highpass", output="sos")

        self._trial_counts = [self._count_trials(p) for p in self.edf_paths]
        self._cum_counts = np.cumsum([0] + self._trial_counts)

        self._cache_file_idx: int | None = None
        self._cache_data: np.ndarray | None = None  # (n_trials, C, T)
        self._cache_texts: list[str] | None = None

    def _word_list_for(self, edf_path: Path) -> list[str]:
        run_number = _parse_run_number(edf_path)
        xlsx_path = self.textdataset_dir / f"split_data_{run_number}.xlsx"
        words_df = pd.read_excel(xlsx_path)
        return words_df.iloc[:, 0].astype(str).tolist()

    def _count_trials(self, edf_path: Path) -> int:
        """Cheap pass (preload=False, no full data load) to count real
        trials for __len__.

        Bug this fixes: originally just counted matching trigger events,
        which OVERCOUNTS -- mne.Epochs silently drops any epoch whose
        tmin/tmax window would extend past the start/end of the recording
        (confirmed via a synthetic-EDF test: a trigger near the end of a
        short recording produced a valid "read" epoch, 0-5s post-trigger,
        but no valid "imagine" epoch, 5-8.3s post-trigger, for the exact
        same trigger -- the longer window didn't fit). Real CHISCO runs are
        long (40-90 min) so this would rarely bite beyond the last trial or
        two of a run, but "rarely" still means a real IndexError mid-training.
        Building the actual mne.Epochs object (still preload=False, so no
        full data load) and taking its length is the single source of truth
        for "how many epochs will _load_file actually produce", rather than
        hand-duplicating mne's own boundary logic and risking drift.
        """
        import mne

        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        events = mne.find_events(raw, stim_channel="Trigger", verbose="ERROR")
        events = events[events[:, 2] == TRIGGER_EVENT_CODE]
        n_words = len(self._word_list_for(edf_path))
        n = min(len(events), n_words)
        events = events[:n]

        tmin, tmax = TASK_WINDOWS[self.task]
        epochs = mne.Epochs(
            raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=False, verbose="ERROR"
        )
        epochs.drop_bad()  # required before len() is meaningful -- see MNE's own error otherwise
        return len(epochs)

    def __len__(self) -> int:
        return int(self._cum_counts[-1])

    def _locate(self, idx: int) -> tuple[int, int]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        file_idx = int(np.searchsorted(self._cum_counts, idx, side="right") - 1)
        local_idx = idx - int(self._cum_counts[file_idx])
        return file_idx, local_idx

    def _load_file(self, file_idx: int) -> None:
        if self._cache_file_idx == file_idx:
            return
        import mne

        edf_path = self.edf_paths[file_idx]
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw.resample(self.sample_rate_hz, verbose="ERROR")

        montage = mne.channels.read_custom_montage(self.montage_path)
        raw.set_montage(montage)
        raw.drop_channels(USELESS_CHANNELS)
        raw.set_channel_types({ch: "eog" for ch in EOG_CHANNELS})

        events = mne.find_events(raw, stim_channel="Trigger", verbose="ERROR")
        events = events[events[:, 2] == TRIGGER_EVENT_CODE]

        words_list = self._word_list_for(edf_path)
        n = min(len(events), len(words_list))
        events = events[:n]
        words_list = words_list[:n]
        metadata = pd.DataFrame({"Word": words_list})

        tmin, tmax = TASK_WINDOWS[self.task]
        epochs = mne.Epochs(
            raw, events, tmin=tmin, tmax=tmax, baseline=None, preload=True,
            metadata=metadata, verbose="ERROR",
        )

        ch_names = epochs.ch_names
        cortical_names = [c for c in ch_names if c not in EOG_CHANNELS and c != "Trigger"]
        ordered_names = cortical_names + EOG_CHANNELS
        reindex = [ch_names.index(c) for c in ordered_names]

        data = epochs.get_data(copy=False)[:, reindex, :] * VOLTS_TO_MICROVOLTS  # (n, C, T)
        if self._sos is not None:
            data = sosfiltfilt(self._sos, data, axis=-1)
        data = data.astype(np.float32)

        actual_texts = epochs.metadata["Word"].tolist()

        self._cache_file_idx = file_idx
        self._cache_data = data
        self._cache_texts = actual_texts

    def __getitem__(self, idx: int) -> CHISCOSample:
        file_idx, local_idx = self._locate(idx)
        self._load_file(file_idx)
        assert self._cache_data is not None and self._cache_texts is not None
        eeg = torch.from_numpy(self._cache_data[local_idx])
        text = self._cache_texts[local_idx]
        return CHISCOSample(eeg=eeg, catalog_content=text.strip())

    @staticmethod
    def cortical_indices(n_cortical: int = 122) -> torch.Tensor:
        return torch.arange(n_cortical, dtype=torch.long)

    @staticmethod
    def eog_indices(n_cortical: int = 122) -> torch.Tensor:
        return torch.tensor([n_cortical, n_cortical + 1], dtype=torch.long)
