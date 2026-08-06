"""
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
