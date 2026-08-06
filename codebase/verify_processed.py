#!/usr/bin/env python3
"""
verify_processed.py
Run this on your server to check all processed parquet files.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path("/mnt/hdd1/TSFM/dataset/processed")

LTSF_DATASETS = {
    "energy/ETTh1.parquet": {"rows": 7, "split": "6:2:2"},
    "energy/ETTh2.parquet": {"rows": 7, "split": "6:2:2"},
    "energy/ETTm1.parquet": {"rows": 7, "split": "6:2:2"},
    "energy/ETTm2.parquet": {"rows": 7, "split": "6:2:2"},
    "energy/electricity.parquet": {"rows": 321, "split": "7:1:2"},
    "energy/solar_energy.parquet": {"rows": 137, "split": "7:1:2"},
    "traffic/traffic.parquet": {"rows": 862, "split": "7:1:2"},
    "traffic/pems03.parquet": {"rows": 358, "split": "6:2:2"},
    "traffic/pems04.parquet": {"rows": 307, "split": "6:2:2"},
    "traffic/pems08.parquet": {"rows": 170, "split": "6:2:2"},
    "weather/weather.parquet": {"rows": 21, "split": "7:1:2"},
}

REQUIRED_COLS = ["item_id", "start", "freq", "target", "group_id", "train_end_idx", "val_end_idx", "test_end_idx"]

print("=" * 80)
print("VERIFYING PROCESSED PARQUET FILES")
print("=" * 80)
print()

errors = []
warnings = []

# ── LTSF-style ──────────────────────────────────────────────────────────────
print(">>> LTSF-Style Datasets")
print()
for path_str, meta in LTSF_DATASETS.items():
    path = PROCESSED / path_str
    name = path_str.split("/")[-1].replace(".parquet", "")
    
    if not path.exists():
        print(f"  [MISS] {name:25s} → not found")
        errors.append(f"{name}: missing")
        continue
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [ERR]  {name:25s} → corrupt: {e}")
        errors.append(f"{name}: corrupt")
        continue
    
    cols = list(df.columns)
    missing_cols = [c for c in REQUIRED_COLS if c not in cols]
    
    # Check row count
    row_ok = len(df) == meta["rows"]
    
    # Check target dtype (should be list/array of floats)
    target_sample = df["target"].iloc[0] if "target" in cols and len(df) > 0 else None
    target_type = type(target_sample).__name__ if target_sample is not None else "N/A"
    
    # Check split indices are positive and ordered
    split_ok = True
    if all(c in cols for c in ["train_end_idx", "val_end_idx", "test_end_idx"]):
        t = df["train_end_idx"].iloc[0]
        v = df["val_end_idx"].iloc[0]
        te = df["test_end_idx"].iloc[0]
        if not (0 < t < v < te):
            split_ok = False
    
    # PEMS alt columns?
    has_pems_alt = all(c in cols for c in ["train_end_idx_ltsf712", "val_end_idx_ltsf712", "test_end_idx_ltsf712"])
    
    status = "OK" if not missing_cols and row_ok and split_ok else "WARN"
    print(f"  [{status:4s}] {name:25s} rows={len(df):4d} target_type={target_type:12s} split_ok={split_ok} alt_712={has_pems_alt}")
    
    if missing_cols:
        print(f"         missing: {missing_cols}")
        warnings.append(f"{name}: missing cols")
    if not row_ok:
        print(f"         expected {meta['rows']} rows, got {len(df)}")
        warnings.append(f"{name}: row count")
    if not split_ok:
        print(f"         split indices invalid: train={t}, val={v}, test={te}")
        warnings.append(f"{name}: bad split")

print()

# ── Archive-style ───────────────────────────────────────────────────────────
print(">>> Archive-Style Datasets")
print()

for path_str, ds_type in [("multi_domain/gift_eval.parquet", "GIFT-Eval"),
                           ("multi_domain/monash.parquet", "Monash")]:
    path = PROCESSED / path_str
    name = path_str.split("/")[-1].replace(".parquet", "")
    
    if not path.exists():
        print(f"  [MISS] {name:25s} → not found")
        errors.append(f"{name}: missing")
        continue
    
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"  [ERR]  {name:25s} → corrupt: {e}")
        errors.append(f"{name}: corrupt")
        continue
    
    cols = list(df.columns)
    base_cols = ["item_id", "start", "freq", "target", "group_id"]
    missing = [c for c in base_cols if c not in cols]
    
    if ds_type == "GIFT-Eval":
        has_pl = "prediction_length" in cols
        print(f"  [OK]   {name:25s} rows={len(df):6d} cols={len(cols):2d} has_pred_len={has_pl}")
    else:  # Monash
        has_h = "horizon" in cols
        has_hs = "horizon_source" in cols
        print(f"  [OK]   {name:25s} rows={len(df):5d} cols={len(cols):2d} has_horizon={has_h} has_horizon_source={has_hs}")
    
    if missing:
        print(f"         missing base cols: {missing}")
        warnings.append(f"{name}: missing cols")

print()

# ── PTB-XL (classification schema) ──────────────────────────────────────────
print(">>> Classification Datasets")
print()
ptbxl_path = PROCESSED / "ecg/ptbxl.parquet"
if ptbxl_path.exists():
    try:
        df = pd.read_parquet(ptbxl_path)
        cols = list(df.columns)
        has_signal = "signal" in cols
        has_labels = "diagnostic_superclass" in cols or "labels" in cols
        has_fold = "strat_fold" in cols
        print(f"  [OK]   ptbxl                  rows={len(df):5d} has_signal={has_signal} has_labels={has_labels} has_fold={has_fold}")
    except Exception as e:
        print(f"  [ERR]  ptbxl                  corrupt: {e}")
        errors.append("ptbxl: corrupt")
else:
    print(f"  [MISS] ptbxl                  → not found (raw records100 still downloading)")
    warnings.append("ptbxl: missing (expected, raw in progress)")

print()
print("=" * 80)
print(f"SUMMARY: {len(errors)} errors, {len(warnings)} warnings")
if warnings:
    print("Warnings:", warnings)
print("=" * 80)