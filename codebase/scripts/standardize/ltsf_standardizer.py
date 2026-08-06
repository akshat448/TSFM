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
