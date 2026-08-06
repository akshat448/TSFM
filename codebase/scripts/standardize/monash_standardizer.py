"""
Standardizer for Monash Archive .tsf files.
Parses .tsf format and applies frequency-based horizon rules.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseStandardizer

logger = logging.getLogger("standardize")


def parse_tsf(path: Path) -> tuple[pd.DataFrame, str, str, str]:
    """
    Parse Monash .tsf file.
    Returns: (data_df, frequency, forecast_horizon, contain_missing_values)
    """
    with open(path, "r", encoding="cp1252", errors="replace") as f:
        lines = f.readlines()

    # Parse header
    frequency = "H"
    forecast_horizon = None
    contain_missing_values = "false"
    header_end = 0

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("@frequency"):
            frequency = line.split(None, 1)[1].strip()
        elif line.startswith("@horizon"):
            forecast_horizon = int(line.split(None, 1)[1].strip())
        elif line.startswith("@missing"):
            contain_missing_values = line.split(None, 1)[1].strip().lower()
        elif line.startswith("@data"):
            header_end = i + 1
            break

    # Parse data lines
    data_lines = lines[header_end:]
    records = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        # Format: series_name \t start_timestamp \t series_values
        parts = line.split("\t")
        if len(parts) >= 3:
            series_name = parts[0]
            start_timestamp = parts[1]
            values_str = parts[2]
            # Values are comma-separated, possibly with missing as ?
            values = []
            for v in values_str.split(","):
                v = v.strip()
                if v == "?" or v == "":
                    values.append(np.nan)
                else:
                    values.append(float(v))
            records.append({
                "series_name": series_name,
                "start_timestamp": start_timestamp,
                "values": np.array(values, dtype=np.float32),
            })

    df = pd.DataFrame(records)
    return df, frequency, forecast_horizon, contain_missing_values


# Frequency-to-horizon mapping per Godahewa et al. Section 4.1
FREQ_HORIZON = {
    "yearly": 6,
    "quarterly": 8,
    "monthly": 18,
    "weekly": 13,
    "daily": 14,
    "hourly": 48,
    "half_hourly": 48,
    "minutely": 60,
    "10_minutes": 48,
    "seconds": 60,
}


def infer_horizon(freq_str: str, explicit_horizon: int | None) -> tuple[int, str]:
    if explicit_horizon is not None:
        return explicit_horizon, "explicit_file_header"
    # Map frequency string
    freq_lower = freq_str.lower().replace("_", "").replace("-", "").replace(" ", "")
    mapping = {
        "yearly": "yearly", "annual": "yearly",
        "quarterly": "quarterly", "q": "quarterly",
        "monthly": "monthly", "m": "monthly",
        "weekly": "weekly", "w": "weekly",
        "daily": "daily", "d": "daily",
        "hourly": "hourly", "h": "hourly",
        "halfhourly": "half_hourly", "30min": "half_hourly",
        "minutely": "minutely", "minute": "minutely", "min": "minutely",
        "10minutes": "10_minutes", "10min": "10_minutes",
    }
    canonical = mapping.get(freq_lower, "hourly")
    horizon = FREQ_HORIZON.get(canonical, 48)
    return horizon, f"freq_rule_{canonical}"


class MonashStandardizer(BaseStandardizer):
    def __init__(self, paths: dict | None = None):
        super().__init__(paths)
        self._name = "monash"

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_subpath(self) -> str:
        return "multi_domain/monash.parquet"

    def standardize(self) -> pd.DataFrame:
        raw_dir = self.raw_dir / "multi_domain" / "monash"
        if not raw_dir.exists():
            raise FileNotFoundError(f"Monash raw dir not found: {raw_dir}")

        tsf_files = sorted(raw_dir.glob("*.tsf"))
        logger.info(f"[{self.name}] Found {len(tsf_files)} .tsf files")

        rows = []
        for tsf_path in tsf_files:
            ds_name = tsf_path.stem.replace("_dataset", "").replace("_without_missing_values", "")
            try:
                df, freq, explicit_h, _ = parse_tsf(tsf_path)
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to parse {tsf_path.name}: {e}")
                continue

            horizon, horizon_source = infer_horizon(freq, explicit_h)

            for _, row in df.iterrows():
                values = row["values"]
                n = len(values)
                # Default split: use horizon as test length
                test_len = horizon
                train_end = max(1, n - test_len * 2)
                val_end = max(train_end + 1, n - test_len)

                # Parse start timestamp
                start_str = row["start_timestamp"]
                try:
                    start_ts = pd.to_datetime(start_str)
                except Exception:
                    start_ts = pd.Timestamp("2000-01-01")

                rows.append({
                    "item_id": f"{ds_name}_{row['series_name']}",
                    "start": start_ts,
                    "freq": freq,
                    "target": values,
                    "group_id": ds_name,
                    "train_end_idx": train_end,
                    "val_end_idx": val_end,
                    "test_end_idx": n,
                    "horizon": horizon,
                    "horizon_source": horizon_source,
                })

        out_df = pd.DataFrame(rows)
        logger.info(f"[{self.name}] Standardized: {len(out_df)} rows")
        return out_df


def build_monash_standardizer(paths: dict) -> MonashStandardizer:
    return MonashStandardizer(paths)
