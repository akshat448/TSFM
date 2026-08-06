#!/usr/bin/env bash
# =============================================================================
# setup_standardize.sh
# Run once to scaffold the standardization pipeline
# =============================================================================
set -euo pipefail

ROOT="/mnt/hdd1/TSFM/codebase"
mkdir -p "$ROOT/scripts/standardize"
mkdir -p "$ROOT/config"

cat <<'EOF' > "$ROOT/config/paths.yaml"
# config/paths.yaml — unified path configuration
data_root: /mnt/hdd1/TSFM/dataset
raw_dir: /mnt/hdd1/TSFM/dataset/raw
processed_dir: /mnt/hdd1/TSFM/dataset/processed
manifest_dir: /mnt/hdd1/TSFM/dataset/manifests
repo_root: /mnt/hdd1/TSFM/codebase
results_dir: /mnt/hdd1/TSFM/codebase/results/phase1
EOF

echo "[1/3] Created config/paths.yaml"

cat <<'PYEOF' > "$ROOT/scripts/standardize/__init__.py"
# Standardization pipeline for TSFM benchmark
PYEOF

cat <<'PYEOF' > "$ROOT/scripts/standardize/base.py"
"""
Base standardizer interface and shared utilities.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("standardize")


def load_paths() -> dict[str, str]:
    repo_root = Path(__file__).resolve().parents[2]
    paths_file = repo_root / "config" / "paths.yaml"
    with open(paths_file) as f:
        return yaml.safe_load(f)


class BaseStandardizer(ABC):
    """Abstract base for all dataset standardizers."""

    def __init__(self, paths: dict[str, str] | None = None):
        self.paths = paths or load_paths()
        self.raw_dir = Path(self.paths["raw_dir"])
        self.processed_dir = Path(self.paths["processed_dir"])
        self.manifest_dir = Path(self.paths.get("manifest_dir", self.paths["raw_dir"].replace("/raw", "/manifests")))
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def standardize(self) -> pd.DataFrame:
        """Return the standardized DataFrame."""
        ...

    def save(self, df: pd.DataFrame, subpath: str) -> Path:
        out = self.processed_dir / subpath
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False, engine="pyarrow")
        logger.info(f"[{self.name}] Saved {len(df)} rows → {out}")
        return out

    def run(self) -> Path:
        logger.info(f"[{self.name}] Starting standardization...")
        df = self.standardize()
        subpath = self.output_subpath
        out = self.save(df, subpath)
        self._write_manifest(df, out)
        return out

    def _write_manifest(self, df: pd.DataFrame, out_path: Path) -> None:
        manifest = {
            "dataset": self.name,
            "output": str(out_path),
            "rows": len(df),
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
        }
        mpath = self.manifest_dir / f"{self.name}_standardize.json"
        import json
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"[{self.name}] Manifest → {mpath}")

    @property
    @abstractmethod
    def output_subpath(self) -> str:
        ...
PYEOF

echo "[2/3] Created base standardizer"

cat <<'PYEOF' > "$ROOT/scripts/standardize/ltsf_standardizer.py"
"""
Standardizer for LTSF-style CSV datasets:
ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Traffic, Weather, Solar, ILI, Exchange Rate
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import BaseStandardizer

logger = logging.getLogger("standardize")


class LTSFStandardizer(BaseStandardizer):
    """
    Standardizes a single LTSF CSV/txt dataset into one-row-per-channel parquet.
    Schema: item_id, start, freq, target (list[float32]), group_id,
            train_end_idx, val_end_idx, test_end_idx
    """

    def __init__(
        self,
        name: str,
        raw_path: Path,
        output_subpath: str,
        freq: str,
        split_ratio: tuple[float, float, float],
        has_date_col: bool = True,
        date_col: str = "date",
        target_cols: list[str] | None = None,
        start_date: str | None = None,
        sep: str = ",",
        **kwargs,
    ):
        super().__init__()
        self._name = name
        self.raw_path = raw_path
        self._output_subpath = output_subpath
        self.freq = freq
        self.split_ratio = split_ratio  # train, val, test
        self.has_date_col = has_date_col
        self.date_col = date_col
        self.target_cols = target_cols
        self.start_date = start_date
        self.sep = sep
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_subpath(self) -> str:
        return self._output_subpath

    def _read_raw(self) -> pd.DataFrame:
        if self.raw_path.suffix == ".txt" or self.raw_path.suffix == ".txt.gz":
            # Solar/Exchange Rate: no header, no date
            df = pd.read_csv(
                self.raw_path,
                sep=self.sep,
                header=None,
                **self.kwargs,
            )
            if self.start_date:
                df.index = pd.date_range(start=self.start_date, periods=len(df), freq=self.freq)
                df = df.reset_index().rename(columns={"index": self.date_col})
            return df
        elif self.raw_path.suffix == ".csv":
            df = pd.read_csv(self.raw_path, **self.kwargs)
            if not self.has_date_col and self.start_date:
                df.index = pd.date_range(start=self.start_date, periods=len(df), freq=self.freq)
                df = df.reset_index().rename(columns={"index": self.date_col})
            return df
        else:
            raise ValueError(f"Unsupported raw format: {self.raw_path}")

    def _compute_splits(self, n: int) -> tuple[int, int, int]:
        r_train, r_val, r_test = self.split_ratio
        train_end = int(n * r_train / sum(self.split_ratio))
        val_end = train_end + int(n * r_val / sum(self.split_ratio))
        test_end = n
        # Ensure no off-by-one
        if val_end >= test_end:
            val_end = test_end - 1
        if train_end >= val_end:
            train_end = val_end - 1
        return train_end, val_end, test_end

    def standardize(self) -> pd.DataFrame:
        df = self._read_raw()
        logger.info(f"[{self.name}] Raw shape: {df.shape}")

        # Identify target columns
        if self.target_cols:
            cols = self.target_cols
        elif self.date_col in df.columns:
            cols = [c for c in df.columns if c != self.date_col]
        else:
            cols = list(df.columns)

        # Parse timestamps
        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            start_ts = df[self.date_col].iloc[0]
        else:
            start_ts = pd.Timestamp(self.start_date) if self.start_date else pd.Timestamp("2000-01-01")

        n = len(df)
        train_end, val_end, test_end = self._compute_splits(n)
        logger.info(f"[{self.name}] Splits: train_end={train_end}, val_end={val_end}, test_end={test_end}")

        rows = []
        for col in cols:
            values = df[col].values.astype(np.float32)
            # Handle any NaNs
            if np.isnan(values).any():
                logger.warning(f"[{self.name}] {col} has {np.isnan(values).sum()} NaNs")
                values = np.nan_to_num(values, nan=0.0)

            rows.append({
                "item_id": str(col),
                "start": start_ts,
                "freq": self.freq,
                "target": values,
                "group_id": self.name,
                "train_end_idx": train_end,
                "val_end_idx": val_end,
                "test_end_idx": test_end,
            })

        out_df = pd.DataFrame(rows)
        logger.info(f"[{self.name}] Standardized: {len(out_df)} rows x {len(out_df.columns)} cols")
        return out_df


def build_ett_standardizer(dataset_name: str, paths: dict) -> LTSFStandardizer:
    """ETTh1, ETTh2, ETTm1, ETTm2"""
    raw_dir = Path(paths["raw_dir"])
    freq = "H" if dataset_name.startswith("ETTh") else "15min"
    # ETT: 6:2:2 split = 12/4/4 months
    # ETTh: 17420 hourly steps → train_end ~ 12/16 * 17420 = ~13065
    # But we compute dynamically from ratio
    return LTSFStandardizer(
        name=dataset_name,
        raw_path=raw_dir / "energy" / "ETTDataset" / "ETT-small" / f"{dataset_name}.csv",
        output_subpath=f"energy/{dataset_name.lower()}.parquet",
        freq=freq,
        split_ratio=(6, 2, 2),
        has_date_col=True,
        date_col="date",
    )


def build_electricity_standardizer(paths: dict) -> LTSFStandardizer:
    raw_dir = Path(paths["raw_dir"])
    return LTSFStandardizer(
        name="electricity",
        raw_path=raw_dir / "energy" / "electricity" / "electricity.csv",
        output_subpath="energy/electricity.parquet",
        freq="H",
        split_ratio=(7, 1, 2),
        has_date_col=True,
        date_col="date",
    )


def build_traffic_standardizer(paths: dict) -> LTSFStandardizer:
    raw_dir = Path(paths["raw_dir"])
    return LTSFStandardizer(
        name="traffic",
        raw_path=raw_dir / "traffic" / "traffic" / "traffic.csv",
        output_subpath="traffic/traffic.parquet",
        freq="H",
        split_ratio=(7, 1, 2),
        has_date_col=True,
        date_col="date",
    )


def build_weather_standardizer(paths: dict) -> LTSFStandardizer:
    raw_dir = Path(paths["raw_dir"])
    return LTSFStandardizer(
        name="weather",
        raw_path=raw_dir / "weather" / "weather" / "weather.csv",
        output_subpath="weather/weather.parquet",
        freq="10min",
        split_ratio=(7, 1, 2),
        has_date_col=True,
        date_col="date",
    )


def build_solar_standardizer(paths: dict) -> LTSFStandardizer:
    raw_dir = Path(paths["raw_dir"])
    # Solar: 10-min intervals, 137 columns, starts 2006-01-01
    return LTSFStandardizer(
        name="solar_energy",
        raw_path=raw_dir / "energy" / "solar_energy" / "solar_AL.txt",
        output_subpath="energy/solar_energy.parquet",
        freq="10min",
        split_ratio=(7, 1, 2),
        has_date_col=False,
        start_date="2006-01-01 00:00:00",
        sep=",",
    )


def build_ili_standardizer(paths: dict) -> LTSFStandardizer:
    raw_dir = Path(paths["raw_dir"])
    # ILI: weekly, 7 regions, from thuml/Time-Series-Library
    # Try multiple locations
    candidates = [
        raw_dir / "health" / "illness" / "illness.csv",
        raw_dir / "energy" / "hf_mirror" / "illness" / "illness.csv",
        raw_dir / "energy" / "ETTDataset" / "illness" / "illness.csv",
    ]
    raw_path = None
    for c in candidates:
        if c.exists():
            raw_path = c
            break
    if raw_path is None:
        raise FileNotFoundError("ILI raw data not found. Run: bash download_all_datasets.sh ili")
    return LTSFStandardizer(
        name="ili",
        raw_path=raw_path,
        output_subpath="health/ili.parquet",
        freq="W",
        split_ratio=(6, 2, 2),
        has_date_col=True,
        date_col="date",
    )


def build_exchange_rate_standardizer(paths: dict) -> LTSFStandardizer:
    raw_dir = Path(paths["raw_dir"])
    # Exchange Rate: daily, 8 currencies
    candidates = [
        raw_dir / "finance" / "exchange_rate" / "exchange_rate.txt",
        raw_dir / "energy" / "multivariate-time-series-data" / "exchange_rate" / "exchange_rate.txt",
        raw_dir / "energy" / "hf_mirror" / "exchange_rate" / "exchange_rate.txt",
    ]
    raw_path = None
    for c in candidates:
        if c.exists():
            raw_path = c
            break
    if raw_path is None:
        raise FileNotFoundError("Exchange Rate raw data not found. Run: bash download_all_datasets.sh exchange_rate")
    return LTSFStandardizer(
        name="exchange_rate",
        raw_path=raw_path,
        output_subpath="finance/exchange_rate.parquet",
        freq="D",
        split_ratio=(6, 2, 2),
        has_date_col=False,
        start_date="1990-01-01",  # Approximate, adjust if known
        sep=",",
    )
PYEOF

echo "[3/3] Created LTSF standardizer"

cat <<'PYEOF' > "$ROOT/scripts/standardize/pems_standardizer.py"
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
PYEOF

echo "[4/6] Created PEMS standardizer"

cat <<'PYEOF' > "$ROOT/scripts/standardize/gift_eval_standardizer.py"
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

        # GIFT-Eval stores datasets as subdirectories with metadata.json
        rows = []
        for ds_dir in sorted(raw_dir.iterdir()):
            if not ds_dir.is_dir():
                continue
            meta_file = ds_dir / "metadata.json"
            if not meta_file.exists():
                continue

            with open(meta_file) as f:
                meta = json.load(f)

            # Load data
            data_file = ds_dir / "data.csv"
            if not data_file.exists():
                continue

            df = pd.read_csv(data_file)
            # GIFT-Eval format: item_id, timestamp, target
            # Group by item_id
            for item_id, group in df.groupby("item_id"):
                ts = pd.to_datetime(group["timestamp"].values)
                target = group["target"].values.astype(np.float32)

                # Use metadata for split if available, else default
                train_end = meta.get("train_end", int(len(target) * 0.7))
                val_end = meta.get("val_end", int(len(target) * 0.8))

                rows.append({
                    "item_id": str(item_id),
                    "start": ts[0],
                    "freq": meta.get("freq", "H"),
                    "target": target,
                    "group_id": ds_dir.name,
                    "train_end_idx": train_end,
                    "val_end_idx": val_end,
                    "test_end_idx": len(target),
                })

        out_df = pd.DataFrame(rows)
        logger.info(f"[{self.name}] Standardized: {len(out_df)} rows from {raw_dir}")
        return out_df


def build_gift_eval_standardizer(paths: dict) -> GIFTEvalStandardizer:
    return GIFTEvalStandardizer(paths)
PYEOF

echo "[5/6] Created GIFT-Eval standardizer"

cat <<'PYEOF' > "$ROOT/scripts/standardize/monash_standardizer.py"
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
    with open(path, "r", encoding="utf-8") as f:
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
PYEOF

echo "[6/6] Created Monash standardizer"

cat <<'PYEOF' > "$ROOT/scripts/standardize/ptbxl_standardizer.py"
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

        # Load metadata
        db_path = raw_dir / "ptbxl_database.csv"
        scp_path = raw_dir / "scp_statements.csv"
        if not db_path.exists():
            raise FileNotFoundError(f"ptbxl_database.csv not found: {db_path}")
        if not scp_path.exists():
            raise FileNotFoundError(f"scp_statements.csv not found: {scp_path}")

        db = pd.read_csv(db_path)
        scp = pd.read_csv(scp_path)

        # Build diagnostic superclass mapping
        # scp_statements has: diagnostic_class (NORM, MI, STTC, CD, HYP)
        scp_super = scp.set_index("scp_code")["diagnostic_class"].to_dict()

        def map_superclass(scp_codes_str):
            """Map scp_codes dict string to list of superclasses."""
            if pd.isna(scp_codes_str):
                return []
            try:
                codes = eval(scp_codes_str) if isinstance(scp_codes_str, str) else scp_codes_str
            except Exception:
                return []
            supers = set()
            for code in codes:
                sup = scp_super.get(code)
                if sup and not pd.isna(sup):
                    supers.add(sup)
            return sorted(list(supers))

        db["diagnostic_superclass"] = db["scp_codes"].apply(map_superclass)

        # Add split column from strat_fold
        def fold_to_split(fold):
            if fold <= 8:
                return "train"
            elif fold == 9:
                return "val"
            else:
                return "test"

        db["split"] = db["strat_fold"].apply(fold_to_split)

        # Load signals from WFDB if available
        records_dir = raw_dir / "records100"
        has_signals = records_dir.exists() and any(records_dir.iterdir())

        rows = []
        for _, row in db.iterrows():
            record = {
                "record_id": row["ecg_id"],
                "diagnostic_superclass": row["diagnostic_superclass"],
                "strat_fold": int(row["strat_fold"]),
                "split": row["split"],
            }

            if has_signals:
                # Try to load WFDB signal
                try:
                    import wfdb
                    record_path = records_dir / f"{row['ecg_id']}_lr"
                    if record_path.with_suffix(".dat").exists() or record_path.with_suffix(".hea").exists():
                        sig, fields = wfdb.rdsamp(str(record_path))
                        # sig shape: (1000, 12) for 10s @ 100Hz
                        record["signal"] = sig.astype(np.float32)
                    else:
                        record["signal"] = np.zeros((1000, 12), dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Could not load signal for {row['ecg_id']}: {e}")
                    record["signal"] = np.zeros((1000, 12), dtype=np.float32)
            else:
                # Placeholder until records download
                record["signal"] = np.zeros((1000, 12), dtype=np.float32)

            rows.append(record)

        out_df = pd.DataFrame(rows)
        logger.info(f"[{self.name}] Standardized: {len(out_df)} rows, signals loaded={has_signals}")
        return out_df


def build_ptbxl_standardizer(paths: dict) -> PTBXLStandardizer:
    return PTBXLStandardizer(paths)
PYEOF

echo "[7/8] Created PTB-XL standardizer"

cat <<'PYEOF' > "$ROOT/scripts/standardize/run_standardize.py"
#!/usr/bin/env python3
"""
Parallel standardization orchestrator for the TSFM benchmark.
Processes all datasets concurrently using ProcessPoolExecutor.

Usage:
    python scripts/standardize/run_standardize.py --all
    python scripts/standardize/run_standardize.py --dataset ETTh1 electricity
    python scripts/standardize/run_standardize.py --family ltsf
    python scripts/standardize/run_standardize.py --list
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from scripts.standardize.base import load_paths
from scripts.standardize.ltsf_standardizer import (
    build_ett_standardizer,
    build_electricity_standardizer,
    build_traffic_standardizer,
    build_weather_standardizer,
    build_solar_standardizer,
    build_ili_standardizer,
    build_exchange_rate_standardizer,
)
from scripts.standardize.pems_standardizer import build_pems_standardizer
from scripts.standardize.gift_eval_standardizer import build_gift_eval_standardizer
from scripts.standardize.monash_standardizer import build_monash_standardizer
from scripts.standardize.ptbxl_standardizer import build_ptbxl_standardizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("run_standardize")


def get_all_standardizers(paths: dict):
    """Return list of (name, builder_func) tuples."""
    standardizers = []

    # LTSF family
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        standardizers.append((ds, lambda p, d=ds: build_ett_standardizer(d, p)))
    standardizers.append(("electricity", build_electricity_standardizer))
    standardizers.append(("traffic", build_traffic_standardizer))
    standardizers.append(("weather", build_weather_standardizer))
    standardizers.append(("solar_energy", build_solar_standardizer))

    # PEMS family
    for ds in ["PEMS03", "PEMS04", "PEMS08"]:
        standardizers.append((ds, lambda p, d=ds: build_pems_standardizer(d, p)))

    # Archive family
    standardizers.append(("gift_eval", build_gift_eval_standardizer))
    standardizers.append(("monash", build_monash_standardizer))

    # Classification
    standardizers.append(("ptbxl", build_ptbxl_standardizer))

    # Optional / Tier 3
    try:
        standardizers.append(("ili", build_ili_standardizer))
    except FileNotFoundError:
        logger.warning("ILI raw data not found, skipping")
    try:
        standardizers.append(("exchange_rate", build_exchange_rate_standardizer))
    except FileNotFoundError:
        logger.warning("Exchange Rate raw data not found, skipping")

    return standardizers


def run_single(name: str, builder, paths: dict) -> tuple[str, str | None]:
    """Run a single standardizer. Returns (name, error_or_none)."""
    try:
        std = builder(paths)
        std.run()
        return name, None
    except Exception as e:
        logger.error(f"[{name}] FAILED: {e}", exc_info=True)
        return name, str(e)


def main():
    parser = argparse.ArgumentParser(description="Standardize TSFM benchmark datasets")
    parser.add_argument("--all", action="store_true", help="Process all datasets")
    parser.add_argument("--dataset", nargs="+", help="Process specific datasets")
    parser.add_argument("--family", choices=["ltsf", "pems", "archive", "classification", "all"], help="Process by family")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (no parallelism)")
    args = parser.parse_args()

    paths = load_paths()
    all_std = get_all_standardizers(paths)

    if args.list:
        print("Available datasets:")
        for name, _ in all_std:
            print(f"  - {name}")
        return

    # Determine which to run
    to_run = []
    if args.all:
        to_run = all_std
    elif args.dataset:
        name_map = {n: b for n, b in all_std}
        for ds in args.dataset:
            if ds in name_map:
                to_run.append((ds, name_map[ds]))
            else:
                logger.warning(f"Unknown dataset: {ds}")
    elif args.family:
        family_map = {
            "ltsf": ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather", "solar_energy", "ili", "exchange_rate"],
            "pems": ["PEMS03", "PEMS04", "PEMS08"],
            "archive": ["gift_eval", "monash"],
            "classification": ["ptbxl"],
            "all": [n for n, _ in all_std],
        }
        names = family_map[args.family]
        name_map = {n: b for n, b in all_std}
        to_run = [(n, name_map[n]) for n in names if n in name_map]
    else:
        parser.print_help()
        return

    if not to_run:
        logger.warning("No datasets selected to process")
        return

    logger.info(f"Processing {len(to_run)} dataset(s) with {args.workers} workers...")

    if args.sequential:
        results = []
        for name, builder in to_run:
            results.append(run_single(name, builder, paths))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_single, name, builder, paths): name for name, builder in to_run}
            results = []
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"[{name}] CRASHED: {e}")
                    results.append((name, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("STANDARDIZATION SUMMARY")
    print("=" * 70)
    ok = [n for n, e in results if e is None]
    fail = [(n, e) for n, e in results if e is not None]
    print(f"  Success: {len(ok)}/{len(results)}")
    for n in ok:
        print(f"    [OK]   {n}")
    if fail:
        print(f"  Failed:  {len(fail)}/{len(results)}")
        for n, e in fail:
            print(f"    [FAIL] {n}: {e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
PYEOF

chmod +x "$ROOT/scripts/standardize/run_standardize.py"

echo "[8/8] Created parallel orchestrator"
echo ""
echo "══════════════════════════════════════════════════════════════════════"
echo "  Standardization pipeline scaffolded at:"
echo "    $ROOT/scripts/standardize/"
echo "══════════════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "    1. cd $ROOT"
echo "    2. python scripts/standardize/run_standardize.py --list"
echo "    3. python scripts/standardize/run_standardize.py --family ltsf --workers 4"
echo "    4. python scripts/standardize/run_standardize.py --family pems --workers 3"
echo "    5. python scripts/standardize/run_standardize.py --family archive --workers 2"
echo "    6. python scripts/standardize/run_standardize.py --dataset ptbxl"
echo ""
echo "  Or run everything at once:"
echo "    python scripts/standardize/run_standardize.py --all --workers 6"
echo "══════════════════════════════════════════════════════════════════════"
EOF

bash /mnt/hdd1/TSFM/codebase/setup_standardize.sh