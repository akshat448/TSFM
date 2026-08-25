"""CHISCO dataset loading and preprocessing.

GROUND TRUTH (verified against a real downloaded chunk — see scripts/download_chisco_chunk.sh
and the inspection notes below — NOT assumed from the original spec, which described this as
an "RSVP reading task"; it is not):

CHISCO (OpenNeuro ds005170, https://openneuro.org/datasets/ds005170) is an IMAGINED-SPEECH
dataset, not RSVP. Each trial has an overt "read" phase (0-5s post-trigger) and an "imagine"
phase (5-8.3s post-trigger); there is no rapid serial visual presentation anywhere in the
paradigm. Source: github.com/zhangzihan-is-good/Chisco (preprocessing.py, data_imagine.py).

Data is distributed as per-run pickles under a BIDS-derivative layout:
    derivatives/preprocessed_pkl/sub-{SS}/eeg/sub-{SS}_task-{imagine,read}_run-{NN}_eeg.pkl
Each pickle is a `list[dict]`, one dict per trial, with EXACTLY two keys:
    "text": str                      -- the imagined/read Chinese sentence (ground truth)
    "input_features": np.ndarray     -- shape (1, 125, T), dtype float64, T=1651 for "imagine"

There is NO "catalog_content" column and NO "unit_values" column in the real data — both were
fabricated in the original architecture spec. "text" is the real ground-truth field (renamed to
`catalog_content` on `CHISCOSample` below purely so the rest of this pipeline, which was written
against that name, doesn't need touching — but the source-of-truth column really is called
"text" on disk); "unit_values" is dropped entirely, there is nothing in real CHISCO it could map to.

Full channel accounting (verified against montage.csv in the CHISCO GitHub repo, 133 rows =
122 cortical + 11 auxiliary): the 11 auxiliary/non-cortical labels are
['11','110','EKG','EMG','84','85','10','111'] (8 unused numeric placeholder positions, all at
coordinate (0,0,0) in the montage -- i.e. not real electrode sites -- plus EKG and EMG) + 'VEO' +
'HEO' + 'Trigger'. Of those 11, only VEO/HEO/Trigger survive into the released derivative: the
other 8 are dropped by `raw.drop_channels(useless_channels)` in the authors' preprocessing.py
BEFORE epoching, which is why `input_features` has 125 rows (122 cortical + VEO + HEO + Trigger),
not 133 or 124.

Channel layout of the 125 rows in `input_features[0]` (verified empirically: cortical rows sit
at ~1e-5 V scale, EOG rows at ~1-20 V scale from blink/saccade artifact, the last row is a
constant digital marker):
    rows   0-121 : 122 cortical EEG channels, volts. Reference repo scales by 1e6 -> microvolts;
                   done here for consistency.
    rows 122-123 : VEO, HEO -- vertical/horizontal EOG. This is the ONLY artifact-reference
                   signal available; there is NO usable EMG channel (it was dropped upstream by
                   the dataset authors' own preprocessing, before this derivative was produced).
                   `SpatialWhiteningFrontEnd` must be constructed with just these 2 noise
                   channels, not "EOG/EMG" as the original spec assumed.
    row      124 : Trigger -- a constant-valued digital event-marker channel, NOT a physiological
                   signal. Always dropped before anything touches the model; exposing it would be
                   a trivial post-hoc leak of trial boundary info, not this dataloader.

Sample rate is 500 Hz (`preprocessing.py`: `raw.resample(sample_rate)`, `sample_rate = 500`),
not 250 Hz as originally assumed.

Double-filtering note: this pkl is already the fully preprocessed DERIVATIVE (the authors' own
1 Hz high-pass + PREP + autoreject has already been applied). Re-applying our own high-pass here
is not wrong, just redundant on top of their 1 Hz cut — kept because the pipeline spec calls for
it explicitly and it's a cheap no-op on already-clean data, but don't expect it to remove
anything real.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset

N_CORTICAL_CHANNELS = 122
CORTICAL_SLICE = slice(0, 122)
EOG_INDICES = (122, 123)  # VEO, HEO
TRIGGER_INDEX = 124
N_TOTAL_CHANNELS_RAW = 125  # before dropping the Trigger row
VOLTS_TO_MICROVOLTS = 1_000_000.0


@dataclasses.dataclass
class CHISCOSample:
    """One imagined/read-speech trial.

    eeg: (C, T) float32 -- cortical + EOG channels only (Trigger row already dropped),
        in microvolts, continuous-valued.
    catalog_content: str -- ground-truth sentence for this trial (source column: "text").
        Threaded through the whole pipeline unmodified; never dropped by any downstream
        transform.
    """

    eeg: torch.Tensor
    catalog_content: str


class CHISCODataloader(Dataset):
    """torch Dataset over CHISCO derivative pickles (see module docstring for the verified
    on-disk schema).

    Parameters
    ----------
    pkl_paths : Sequence[str | Path]
        Paths to one or more `sub-*_task-*_run-*_eeg.pkl` files (as produced by the CHISCO
        authors' `preprocessing.py`). Each is a list of {"text", "input_features"} trial dicts;
        this dataset concatenates all of them.
    sample_rate_hz : float
        Real value is 500.0 Hz for the released derivative; do not change unless pointing this
        at differently-resampled data.
    highpass_hz : float | None
        Optional extra zero-phase high-pass on top of the already-preprocessed derivative
        (see "Double-filtering note" above). Pass None to skip it entirely (the data is already
        filtered). Default 0.5 Hz mirrors the original architecture spec.
    crop_samples : int | None
        If set, truncate every trial's time axis to the first `crop_samples` samples (after
        filtering). Needed to combine trials of different native length into one batch/dataset
        -- e.g. CHISCO's "read" epochs are 2501 samples and "imagine" epochs are 1651 samples;
        without a shared crop length, `collate_chisco`'s `torch.stack` cannot combine them.
    """

    def __init__(
        self,
        pkl_paths: Sequence[str | Path],
        sample_rate_hz: float = 500.0,
        highpass_hz: float | None = 0.5,
        filter_order: int = 4,
        crop_samples: int | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.highpass_hz = highpass_hz
        self.crop_samples = crop_samples

        self._sos = None
        if highpass_hz is not None:
            nyquist = sample_rate_hz / 2.0
            if highpass_hz >= nyquist:
                raise ValueError(f"highpass_hz={highpass_hz} must be < Nyquist={nyquist}")
            self._sos = butter(filter_order, highpass_hz / nyquist, btype="highpass", output="sos")

        self.trials: list[dict] = []
        for p in pkl_paths:
            with open(p, "rb") as f:
                trials = pickle.load(f)
            if not isinstance(trials, list) or (trials and set(trials[0].keys()) != {"text", "input_features"}):
                raise ValueError(
                    f"{p}: expected a list of {{'text','input_features'}} dicts "
                    "(CHISCO derivative pkl schema) -- got something else. If the real file "
                    "differs from this, fix the schema assumptions in this module's docstring."
                )
            self.trials.extend(trials)

    def __len__(self) -> int:
        return len(self.trials)

    def _apply_highpass(self, eeg: np.ndarray) -> np.ndarray:
        """Zero-phase Butterworth high-pass along the time axis. eeg: (C, T) -> (C, T).
        No-op (returns input unchanged) if constructed with highpass_hz=None."""
        if self._sos is None:
            return eeg
        return sosfiltfilt(self._sos, eeg, axis=-1).astype(np.float32)

    def __getitem__(self, idx: int) -> CHISCOSample:
        trial = self.trials[idx]

        raw = np.asarray(trial["input_features"], dtype=np.float64)  # (1, 125, T)
        if raw.ndim == 3:
            raw = raw[0]  # (125, T)
        if raw.shape[0] != N_TOTAL_CHANNELS_RAW:
            raise ValueError(
                f"Row {idx}: input_features has {raw.shape[0]} channels, "
                f"expected {N_TOTAL_CHANNELS_RAW} (122 cortical + VEO + HEO + Trigger)"
            )

        physiological = raw[:TRIGGER_INDEX, :] * VOLTS_TO_MICROVOLTS  # (124, T), drop Trigger row
        eeg_filtered = self._apply_highpass(physiological.astype(np.float32))
        if self.crop_samples is not None:
            if eeg_filtered.shape[-1] < self.crop_samples:
                raise ValueError(
                    f"Row {idx}: trial has {eeg_filtered.shape[-1]} samples, "
                    f"shorter than crop_samples={self.crop_samples}"
                )
            eeg_filtered = eeg_filtered[:, : self.crop_samples]

        catalog_content = trial["text"]
        if not isinstance(catalog_content, str) or len(catalog_content.strip()) == 0:
            raise ValueError(f"Row {idx}: 'text' must be a non-empty string")

        return CHISCOSample(
            eeg=torch.from_numpy(eeg_filtered),
            catalog_content=catalog_content.strip(),
        )

    @staticmethod
    def cortical_indices() -> torch.Tensor:
        """Indices into the (124-channel, Trigger-already-dropped) eeg tensor selecting the
        122 cortical rows."""
        return torch.arange(N_CORTICAL_CHANNELS, dtype=torch.long)

    @staticmethod
    def eog_indices() -> torch.Tensor:
        """Indices selecting the 2 EOG rows (VEO, HEO). No EMG channel exists in real CHISCO --
        `SpatialWhiteningFrontEnd` must be built with just these two."""
        return torch.tensor(EOG_INDICES, dtype=torch.long)


def collate_chisco(batch: Sequence[CHISCOSample]) -> dict:
    """Pad-free collate: assumes all trials in a batch share T (true within one task variant,
    e.g. all "imagine" epochs are 1651 samples). Mixing "imagine" and "read" pkl files in one
    DataLoader will break this -- keep them in separate datasets/batches.

    Returns
    -------
    dict with:
      eeg: (B, 124, T) float32, microvolts, Trigger row already removed.
      catalog_content: list[str] of length B.
    """
    eeg = torch.stack([s.eeg for s in batch], dim=0)
    catalog_content = [s.catalog_content for s in batch]
    return {"eeg": eeg, "catalog_content": catalog_content}
