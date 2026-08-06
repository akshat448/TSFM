"""
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
