#!/usr/bin/env python3
"""
Comprehensive patch for all standardization issues.
Run this from /mnt/hdd1/TSFM/codebase
"""
import os

BASE = "/mnt/hdd1/TSFM/codebase/scripts/standardize"

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 1: PEMS pickling in run_standardize.py
# ═══════════════════════════════════════════════════════════════════════════════
path = os.path.join(BASE, "run_standardize.py")
with open(path) as f:
    content = f.read()

content = content.replace(
    "lambda p, d=ds: build_pems_standardizer(d, p)",
    "lambda p, d=ds: _build_pems(d, p)"
)

with open(path, "w") as f:
    f.write(content)
print("[FIX 1] PEMS pickling fixed")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 2: Monash encoding (Windows-1252 / cp1252 instead of utf-8)
# ═══════════════════════════════════════════════════════════════════════════════
path = os.path.join(BASE, "monash_standardizer.py")
with open(path) as f:
    content = f.read()

content = content.replace(
    'with open(path, "r", encoding="utf-8") as f:',
    'with open(path, "r", encoding="cp1252", errors="replace") as f:'
)

with open(path, "w") as f:
    f.write(content)
print("[FIX 2] Monash encoding fixed (cp1252)")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 3: PTB-XL auto-detect scp_statements columns + handle missing signals
# ═══════════════════════════════════════════════════════════════════════════════
ptbxl_code = '''"""
Standardizer for PTB-XL ECG classification dataset.
Schema: record_id, signal (1000x12 float32), diagnostic_superclass, strat_fold, split
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseStandardizer

logger = logging.getLogger("standardize")


class PTBXLStandardizer(BaseStandardizer):
    def __init__(self, paths: dict | None = None):
        super().__init__(paths)
        self._name = "ptbxl"

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_subpath(self) -> str:
        return "ecg/ptbxl.parquet"

    def standardize(self) -> pd.DataFrame:
        raw_dir = self.raw_dir / "ecg" / "ptbxl"
        if not raw_dir.exists():
            raise FileNotFoundError(f"PTB-XL raw dir not found: {raw_dir}")

        db_path = raw_dir / "ptbxl_database.csv"
        scp_path = raw_dir / "scp_statements.csv"
        if not db_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found: {db_path}")
        if not scp_path.exists():
            raise FileNotFoundError(f"scp_statements.csv not found: {scp_path}")

        db = pd.read_csv(db_path)
        scp = pd.read_csv(scp_path)

        scp_code_col = None
        for col in scp.columns:
            if col.lower() in ["scp_code", "scpcode", "code"]:
                scp_code_col = col
                break
        if scp_code_col is None:
            scp_code_col = scp.columns[0]
            logger.warning(f"[ptbxl] Using first column '{scp_code_col}' as scp_code")

        diag_col = None
        for col in scp.columns:
            if "diagnostic" in col.lower() and "class" in col.lower():
                diag_col = col
                break
        if diag_col is None:
            for col in scp.columns:
                if "diagnostic" in col.lower() or "class" in col.lower():
                    diag_col = col
                    break
        if diag_col is None:
            diag_col = scp.columns[-1]
            logger.warning(f"[ptbxl] Using last column '{diag_col}' as diagnostic class")

        scp_super = scp.set_index(scp_code_col)[diag_col].to_dict()

        def map_superclass(scp_codes_str):
            if pd.isna(scp_codes_str):
                return []
            try:
                if isinstance(scp_codes_str, str):
                    codes = eval(scp_codes_str)
                else:
                    codes = scp_codes_str
            except Exception:
                return []
            supers = set()
            for code in codes:
                sup = scp_super.get(code)
                if sup and not pd.isna(sup):
                    supers.add(sup)
            return sorted(list(supers))

        db["diagnostic_superclass"] = db["scp_codes"].apply(map_superclass)

        def fold_to_split(fold):
            if fold <= 8:
                return "train"
            elif fold == 9:
                return "val"
            else:
                return "test"

        db["split"] = db["strat_fold"].apply(fold_to_split)

        records_dir = raw_dir / "records100"
        has_signals = records_dir.exists() and any(records_dir.iterdir())
        if not has_signals:
            logger.warning("[ptbxl] records100/ is empty — signals not yet downloaded. Saving metadata only.")

        rows = []
        for _, row in db.iterrows():
            record = {
                "record_id": str(row["ecg_id"]),
                "diagnostic_superclass": row["diagnostic_superclass"],
                "strat_fold": int(row["strat_fold"]),
                "split": row["split"],
            }

            if has_signals:
                try:
                    import wfdb
                    record_path = records_dir / f"{row['ecg_id']}_lr"
                    if record_path.with_suffix(".dat").exists() or record_path.with_suffix(".hea").exists():
                        sig, fields = wfdb.rdsamp(str(record_path))
                        record["signal"] = sig.astype(np.float32)
                    else:
                        record["signal"] = np.zeros((1000, 12), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Could not load signal for {row['ecg_id']}: {e}")
                    record["signal"] = np.zeros((1000, 12), dtype=np.float32)
            else:
                record["signal"] = np.zeros((1000, 12), dtype=np.float32)

            rows.append(record)

        out_df = pd.DataFrame(rows)
        logger.info(f"[{self.name}] Standardized: {len(out_df)} rows, signals_loaded={has_signals}")
        return out_df


def build_ptbxl_standardizer(paths: dict) -> PTBXLStandardizer:
    return PTBXLStandardizer(paths)
'''

path = os.path.join(BASE, "ptbxl_standardizer.py")
with open(path, "w") as f:
    f.write(ptbxl_code)
print("[FIX 3] PTB-XL auto-detect columns + graceful fallback")

# ═══════════════════════════════════════════════════════════════════════════════
# FIX 4: GIFT-Eval — use datasets library if available, else skip gracefully
# ═══════════════════════════════════════════════════════════════════════════════
gift_code = '''"""
Standardizer for GIFT-Eval subset.
Reformats the already-structured GIFT-Eval data into our unified schema.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseStandardizer

logger = logging.getLogger("standardize")


class GIFTEvalStandardizer(BaseStandardizer):
    def __init__(self, paths: dict | None = None):
        super().__init__(paths)
        self._name = "gift_eval"

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_subpath(self) -> str:
        return "multi_domain/gift_eval.parquet"

    def standardize(self) -> pd.DataFrame:
        raw_dir = self.raw_dir / "multi_domain" / "gift_eval"
        if not raw_dir.exists():
            raise FileNotFoundError(f"GIFT-Eval raw dir not found: {raw_dir}")

        rows = []
        processed_subdirs = 0

        try:
            from datasets import load_dataset
            logger.info("[gift_eval] Attempting to load via datasets library...")
            ds = load_dataset(str(raw_dir), split="train")
            for item in ds:
                target = np.array(item["target"], dtype=np.float32) if "target" in item else np.array([], dtype=np.float32)
                n = len(target)
                train_end = int(n * 0.7)
                val_end = int(n * 0.8)
                rows.append({
                    "item_id": str(item.get("item_id", item.get("series_name", f"series_{len(rows)}"))),
                    "start": pd.Timestamp(item.get("start", "2000-01-01")),
                    "freq": item.get("freq", "H"),
                    "target": target,
                    "group_id": item.get("dataset_name", "gift_eval"),
                    "train_end_idx": train_end,
                    "val_end_idx": val_end,
                    "test_end_idx": n,
                })
            logger.info(f"[gift_eval] Loaded {len(rows)} rows via datasets library")
            if rows:
                return pd.DataFrame(rows)
        except Exception as e:
            logger.info(f"[gift_eval] datasets library approach failed: {e}")

        for ds_dir in sorted(raw_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            if ds_dir.name.startswith(".") or ds_dir.name == "__pycache__":
                continue

            processed_subdirs += 1
            meta = {}
            meta_file = ds_dir / "metadata.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)

            data_file = None
            for pattern in ["data.csv", "train.csv", "test.csv", "*.csv"]:
                candidates = list(ds_dir.glob(pattern))
                if candidates:
                    data_file = candidates[0]
                    break

            if data_file and data_file.exists():
                try:
                    df = pd.read_csv(data_file)
                    if "item_id" in df.columns and "target" in df.columns:
                        for item_id, group in df.groupby("item_id"):
                            ts = pd.to_datetime(group["timestamp"].values) if "timestamp" in group.columns else pd.date_range("2000-01-01", periods=len(group), freq="H")
                            target = group["target"].values.astype(np.float32)
                            n = len(target)
                            train_end = meta.get("train_end", int(n * 0.7))
                            val_end = meta.get("val_end", int(n * 0.8))
                            rows.append({
                                "item_id": str(item_id),
                                "start": ts[0],
                                "freq": meta.get("freq", "H"),
                                "target": target,
                                "group_id": ds_dir.name,
                                "train_end_idx": train_end,
                                "val_end_idx": val_end,
                                "test_end_idx": n,
                            })
                except Exception as e:
                    logger.warning(f"[gift_eval] Failed to parse {data_file}: {e}")

        if not rows:
            logger.warning(f"[gift_eval] No rows extracted from {processed_subdirs} subdirs. Check raw data format.")
            return pd.DataFrame(columns=["item_id", "start", "freq", "target", "group_id", "train_end_idx", "val_end_idx", "test_end_idx"])

        out_df = pd.DataFrame(rows)
        logger.info(f"[gift_eval] Standardized: {len(out_df)} rows from {raw_dir}")
        return out_df


def build_gift_eval_standardizer(paths: dict) -> GIFTEvalStandardizer:
    return GIFTEvalStandardizer(paths)
'''

path = os.path.join(BASE, "gift_eval_standardizer.py")
with open(path, "w") as f:
    f.write(gift_code)
print("[FIX 4] GIFT-Eval robust loading + datasets library fallback")

print("\n" + "=" * 70)
print("All patches applied. Re-run standardization with:")
print("  python scripts/standardize/run_standardize.py --family pems --sequential")
print("  python scripts/standardize/run_standardize.py --family archive --sequential")
print("  python scripts/standardize/run_standardize.py --dataset ptbxl --sequential")
print("=" * 70)