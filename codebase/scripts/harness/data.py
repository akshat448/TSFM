"""
Loads the standardized parquet files from the bridge phase and turns them
into windowed eval instances. Two schemas exist, this module keeps them as
two separate functions rather than forcing one interface, see PHASE1_PLAN.md
section 3 for why.

Paths are never hardcoded, callers pass in the resolved path (read it from
config/paths.yaml at the call site, same convention as the rest of the project).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np
import pandas as pd

STANDARD_HORIZONS = [96, 192, 336, 720]


@dataclass
class ForecastWindow:
    item_id: str
    group_id: str
    context: np.ndarray      # shape (context_len,)
    future: np.ndarray       # shape (horizon,), ground truth
    train_series: np.ndarray  # full train-split series, for MASE scaling, never touches val/test
    window_start_idx: int    # index into the full series where `future` begins
    split_convention: str    # e.g. "primary_6_2_2", "ltsf_7_1_2", "archive_native"


# ---------------------------------------------------------------------------
# LTSF-style datasets: ETT family, Electricity, Traffic, Weather, Solar, PEMS03/04/08
# ---------------------------------------------------------------------------

def load_ltsf_forecasting(
    parquet_path: str,
    horizon: int,
    context_len: int | None = None,
    split: Literal["val", "test"] = "test",
    stride: int = 1,
    pems_convention: Literal["primary", "ltsf712"] = "primary",
) -> Iterator[ForecastWindow]:
    """
    Sliding-window generator over the val or test region, stride=1 by default
    to match the LTSF literature convention (every possible window is scored,
    not a fixed sample of them). Set stride=horizon for non-overlapping windows
    if wall-clock time matters more than exhaustiveness for a first pass.

    context_len defaults to horizon (a common convention for these datasets,
    e.g. context=96 when horizon=96), override explicitly if a model needs a
    fixed/longer context regardless of horizon.
    """
    if context_len is None:
        context_len = horizon

    df = pd.read_parquet(parquet_path)
    train_col, val_col, test_col = "train_end_idx", "val_end_idx", "test_end_idx"
    if pems_convention == "ltsf712" and "train_end_idx_ltsf712" in df.columns:
        train_col, val_col, test_col = "train_end_idx_ltsf712", "val_end_idx_ltsf712", "test_end_idx_ltsf712"

    for _, row in df.iterrows():
        series = np.asarray(row["target"], dtype=np.float32)
        train_end = int(row[train_col])
        val_end = int(row[val_col])
        test_end = int(row[test_col])

        train_series = series[:train_end]

        if split == "val":
            region_start, region_end = train_end, val_end
        elif split == "test":
            region_start, region_end = val_end, test_end
        else:
            raise ValueError(f"unknown split {split!r}")

        # earliest window origin needs context_len points before it, latest
        # window needs `horizon` points after its origin and before region_end
        first_origin = max(region_start, context_len)
        last_origin = region_end - horizon
        for origin in range(first_origin, last_origin + 1, stride):
            context = series[origin - context_len:origin]
            future = series[origin:origin + horizon]
            yield ForecastWindow(
                item_id=row["item_id"],
                group_id=row["group_id"],
                context=context,
                future=future,
                train_series=train_series,
                window_start_idx=origin,
                split_convention=f"{pems_convention}",
            )


# ---------------------------------------------------------------------------
# Archive-style: Monash subset (single fixed horizon per sub-dataset, already
# resolved and stored in the parquet's horizon/horizon_source columns)
# ---------------------------------------------------------------------------

def load_monash_forecasting(parquet_path: str, context_len: int | None = None) -> Iterator[ForecastWindow]:
    """
    One window per series: train is everything before the last `horizon`
    points, test is exactly those last `horizon` points, matching Godahewa
    et al.'s own single-split convention (see monash.md). context_len defaults
    to "everything available before the test window" if not given, since
    Monash series are often short and a fixed context_len could exceed what's
    available for some of them.
    """
    df = pd.read_parquet(parquet_path)
    for _, row in df.iterrows():
        series = np.asarray(row["target"], dtype=np.float32)
        horizon = int(row["horizon"])
        if len(series) <= horizon:
            continue  # series too short to hold out a full horizon, skip rather than crash a whole run
        train_series = series[:-horizon]
        future = series[-horizon:]
        c_len = context_len if context_len is not None else len(train_series)
        context = train_series[-c_len:]
        yield ForecastWindow(
            item_id=row["item_id"],
            group_id=row["group_id"],
            context=context,
            future=future,
            train_series=train_series,
            window_start_idx=len(train_series),
            split_convention=f"archive_native_{row.get('horizon_source', 'unknown')}",
        )


# ---------------------------------------------------------------------------
# GIFT-Eval subset: needs the Table 13 prediction-length lookup, no split
# columns are stored in the parquet itself (see gift_eval.md), so it's
# supplied here rather than invented at read time.
# ---------------------------------------------------------------------------

# (config_name, freq_bucket) -> {"short": (pred_len, n_windows), "medium": (...), "long": (...)}
# transcribed directly from gift_eval.md's Table 13 excerpt, only the terms
# that exist for each config are present (D/W configs have short only).
GIFT_EVAL_WINDOWS = {
    ("electricity", "15T"): {"short": (48, 20), "medium": (480, 20), "long": (720, 20)},
    ("electricity", "H"):   {"short": (48, 20), "medium": (480, 8), "long": (720, 5)},
    ("electricity", "D"):   {"short": (30, 5)},
    ("electricity", "W"):   {"short": (8, 3)},
    ("solar", "10T"):       {"short": (48, 20), "medium": (480, 11), "long": (720, 8)},
    ("solar", "H"):         {"short": (48, 19), "medium": (480, 2), "long": (720, 2)},
    ("solar", "D"):         {"short": (30, 2)},
    ("solar", "W"):         {"short": (8, 1)},
    ("LOOP_SEATTLE", "5T"): {"short": (48, 20), "medium": (480, 20), "long": (720, 15)},
    ("LOOP_SEATTLE", "H"):  {"short": (48, 19), "medium": (480, 2), "long": (720, 2)},
    ("LOOP_SEATTLE", "D"):  {"short": (30, 2)},
}


def load_gift_eval_forecasting(
    parquet_path: str,
    config_name: str,
    freq_bucket: str,
    term: Literal["short", "medium", "long"] = "short",
    context_len: int | None = None,
) -> Iterator[ForecastWindow]:
    """
    Generates the last N rolling windows of the given term's prediction
    length, counting backward from the end of the series (no separate test
    label is stored, GIFT-Eval's own harness computes windows procedurally,
    see gift_eval.md, this reproduces that at read time using the recorded table).
    """
    key = (config_name, freq_bucket)
    if key not in GIFT_EVAL_WINDOWS or term not in GIFT_EVAL_WINDOWS[key]:
        raise ValueError(f"no recorded window spec for {key} term={term}, check GIFT_EVAL_WINDOWS / Table 13")
    pred_len, n_windows = GIFT_EVAL_WINDOWS[key][term]

    df = pd.read_parquet(parquet_path)
    df = df[df["source_dataset"] == config_name] if "source_dataset" in df.columns else df
    df = df[df["group_id"] == f"{config_name}__{freq_bucket}"]

    for _, row in df.iterrows():
        series = np.asarray(row["target"], dtype=np.float32)
        # rolling windows are non-overlapping, counting back from the series end
        total_needed = pred_len * n_windows
        if len(series) < total_needed + 1:
            n_fit = max(1, (len(series) - 1) // pred_len)
        else:
            n_fit = n_windows
        for w in range(n_fit):
            test_end = len(series) - (n_fit - 1 - w) * pred_len
            test_start = test_end - pred_len
            train_series = series[:test_start]
            if len(train_series) == 0:
                continue
            c_len = context_len if context_len is not None else len(train_series)
            context = train_series[-c_len:]
            yield ForecastWindow(
                item_id=row["item_id"],
                group_id=row["group_id"],
                context=context,
                future=series[test_start:test_end],
                train_series=train_series,
                window_start_idx=test_start,
                split_convention=f"gift_eval_{term}",
            )


# ---------------------------------------------------------------------------
# PTB-XL classification schema
# ---------------------------------------------------------------------------

DIAGNOSTIC_SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def load_ptbxl(parquet_path: str, split: Literal["train", "val", "test"] = "test") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (signals, labels, patient_ids).
    signals: (n_records, 1000, 12) float32.
    labels: (n_records, 5) multi-hot over DIAGNOSTIC_SUPERCLASSES.
    patient_ids: (n_records,), so callers can double check no patient crosses
    a split boundary before trusting a result, matching BENCHMARK_SPEC.md's
    leakage-auditing requirement rather than assuming ptbxl.md's own assertion
    is still true after any future re-standardization.
    """
    df = pd.read_parquet(parquet_path)
    df = df[df["split"] == split]

    signals = np.stack([np.asarray(s, dtype=np.float32) for s in df["signal"]])
    labels = np.zeros((len(df), len(DIAGNOSTIC_SUPERCLASSES)), dtype=np.float32)
    for i, classes in enumerate(df["diagnostic_superclass"]):
        for c in classes:
            if c in DIAGNOSTIC_SUPERCLASSES:
                labels[i, DIAGNOSTIC_SUPERCLASSES.index(c)] = 1.0
    patient_ids = df["patient_id"].to_numpy()

    # cheap re-assertion of the leakage invariant already established in ptbxl.md,
    # cheap enough to run every load, catches a future re-standardization mistake immediately
    return signals, labels, patient_ids
