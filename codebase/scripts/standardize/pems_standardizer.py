"""
Standardizer for PEMS03, PEMS04, PEMS08 .npz files.
Stores both 6:2:2 (primary, literature-correct) and 7:1:2 (alternate) splits.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseStandardizer

logger = logging.getLogger("standardize")


class PEMSStandardizer(BaseStandardizer):
    def __init__(self, dataset_name: str, raw_path: Path, output_subpath: str, paths: dict | None = None):
        super().__init__(paths)
        self._name = dataset_name.lower()
        self.raw_path = raw_path
        self._output_subpath = output_subpath

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_subpath(self) -> str:
        return self._output_subpath

    def standardize(self) -> pd.DataFrame:
        data = np.load(self.raw_path)
        key = list(data.keys())[0]
        arr = data[key]
        logger.info(f"[{self.name}] Raw shape: {arr.shape}, dtype: {arr.dtype}")

        # Handle (T, N, C) vs (T, N)
        if arr.ndim == 3:
            T, N, C = arr.shape
            # Use all channels: flatten to (T, N*C) with item_ids like "sensor_000_ch0"
            logger.info(f"[{self.name}] 3D array detected, flattening {C} channels")
            # Actually for LTSF consistency, let's use channel 0 (flow) as primary
            # and create N rows. This matches existing processed files.
            arr = arr[:, :, 0]  # (T, N)
            T, N = arr.shape
        else:
            T, N = arr.shape

        # Splits
        # Primary: 6:2:2
        total = T
        train_end_622 = int(total * 6 / 10)
        val_end_622 = train_end_622 + int(total * 2 / 10)
        test_end_622 = total

        # Alternate: 7:1:2
        train_end_712 = int(total * 7 / 10)
        val_end_712 = train_end_712 + int(total * 1 / 10)
        test_end_712 = total

        # Frequency: 5-minute intervals
        freq = "5min"
        start_ts = pd.Timestamp("2012-01-01 00:00:00")  # PEMS typical start

        rows = []
        for i in range(N):
            values = arr[:, i].astype(np.float32)
            if np.isnan(values).any():
                values = np.nan_to_num(values, nan=0.0)

            rows.append({
                "item_id": f"sensor_{i:03d}",
                "start": start_ts,
                "freq": freq,
                "target": values,
                "group_id": self._name,
                "train_end_idx": train_end_622,
                "val_end_idx": val_end_622,
                "test_end_idx": test_end_622,
                "train_end_idx_ltsf712": train_end_712,
                "val_end_idx_ltsf712": val_end_712,
                "test_end_idx_ltsf712": test_end_712,
            })

        out_df = pd.DataFrame(rows)
        logger.info(f"[{self.name}] Standardized: {len(out_df)} rows")
        return out_df


def build_pems_standardizer(dataset_name: str, paths: dict) -> PEMSStandardizer:
    raw_dir = Path(paths["raw_dir"])
    name_lower = dataset_name.lower()
    return PEMSStandardizer(
        dataset_name=dataset_name,
        raw_path=raw_dir / "traffic" / "PEMS" / f"{dataset_name}.npz",
        output_subpath=f"traffic/{name_lower}.parquet",
        paths=paths,
    )
