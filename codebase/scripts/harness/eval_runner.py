"""
Orchestrates dataset x model x horizon and writes raw, unaggregated results.
Never imports a specific model library directly, only talks to the
ForecastModel interface via registry.py. Aggregation, ranking, and
significance testing are a separate downstream step (Phase 2's job per
BENCHMARK_SPEC.md), this file's only job is to produce a trustworthy raw table.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import pandas as pd

from .contamination import ContaminationMatrix
from .data import ForecastWindow, load_ltsf_forecasting, load_monash_forecasting, STANDARD_HORIZONS
from .metrics import mae, mase, mse, weighted_quantile_loss, crps_from_quantiles
from .models.base import ForecastModel
from .registry import build_forecast_model


@dataclass
class ResultRow:
    dataset: str
    model: str
    item_id: str
    group_id: str
    horizon: int
    window_start_idx: int
    split_convention: str
    contamination_label: str
    mse: float
    mae: float
    mase: float
    wql: float | None
    crps: float | None
    predict_seconds: float


def evaluate_window(model: ForecastModel, window: ForecastWindow, horizon: int,
                     dataset_name: str, contamination: ContaminationMatrix,
                     season_length_for_mase: int = 1) -> ResultRow:
    t0 = time.perf_counter()
    result = model.predict(window.context, horizon)
    predict_seconds = time.perf_counter() - t0

    wql_val = crps_val = None
    if result.quantiles is not None:
        wql_val = weighted_quantile_loss(window.future, result.quantiles)
        crps_val = crps_from_quantiles(window.future, result.quantiles)

    return ResultRow(
        dataset=dataset_name,
        model=model.name,
        item_id=window.item_id,
        group_id=window.group_id,
        horizon=horizon,
        window_start_idx=window.window_start_idx,
        split_convention=window.split_convention,
        contamination_label=contamination.label(dataset_name, model.name),
        mse=mse(window.future, result.point),
        mae=mae(window.future, result.point),
        mase=mase(window.future, result.point, window.train_series, season_length=season_length_for_mase),
        wql=wql_val,
        crps=crps_val,
        predict_seconds=predict_seconds,
    )


def run_zero_shot_ltsf(
    dataset_name: str,
    parquet_path: str,
    model_name: str,
    model_kwargs: dict,
    contamination: ContaminationMatrix,
    horizons: list[int] = STANDARD_HORIZONS,
    stride: int = 1,
    season_length_for_mase: int = 1,
) -> pd.DataFrame:
    """
    Zero-shot eval loop for an LTSF-style dataset. Builds one model instance
    per horizon (context_len typically equals horizon for these models, so a
    fresh instance per horizon keeps that assumption explicit rather than hidden).
    """
    rows: list[ResultRow] = []
    for horizon in horizons:
        # model_kwargs is the caller's complete constructor kwargs (e.g.
        # season_length for seasonal_naive, checkpoint/context_len for a
        # TSFM adapter). Forecast horizon is a predict()-time argument, not
        # injected into the constructor here, since not every model needs
        # it fixed at construction (Chronos-Bolt/TimesFM don't, TTM does via
        # its checkpoint revision, which the caller already encodes in model_kwargs).
        model = build_forecast_model(model_name, **model_kwargs)
        model.fit(None)  # no-op for zero-shot, kept for interface consistency
        for window in load_ltsf_forecasting(parquet_path, horizon=horizon, stride=stride):
            rows.append(evaluate_window(model, window, horizon, dataset_name, contamination, season_length_for_mase))
    return pd.DataFrame([asdict(r) for r in rows])


def run_trained_ltsf(
    dataset_name: str,
    parquet_path: str,
    model_name: str,
    model_kwargs: dict,
    contamination: ContaminationMatrix,
    horizons: list[int] = STANDARD_HORIZONS,
    context_len: int | None = None,
    stride: int = 1,
    season_length_for_mase: int = 1,
) -> pd.DataFrame:
    """
    Same idea but for models that need real per-series training (DLinear,
    PatchTST, iTransformer, S-Mamba). Trains once per (series, horizon) on
    that series' own train split, then evaluates every val/test window for
    that series. This is the "train per series" convention the LTSF
    literature itself uses for these datasets, not a project invention.
    """
    import pandas as pd_local

    rows: list[ResultRow] = []
    df = pd_local.read_parquet(parquet_path)
    for horizon in horizons:
        c_len = context_len or horizon
        for _, series_row in df.iterrows():
            model = build_forecast_model(
                model_name, context_len=c_len, horizon=horizon,
                **{k: v for k, v in model_kwargs.items() if k not in ("context_len", "horizon")},
            )
            train_end = int(series_row["train_end_idx"])
            model.fit(series_row["target"][:train_end])

            for window in load_ltsf_forecasting(parquet_path, horizon=horizon, context_len=c_len, stride=stride):
                if window.item_id != series_row["item_id"]:
                    continue
                rows.append(evaluate_window(model, window, horizon, dataset_name, contamination, season_length_for_mase))
    return pd.DataFrame([asdict(r) for r in rows])


def run_monash(
    dataset_name: str,
    parquet_path: str,
    model_name: str,
    model_kwargs: dict,
    contamination: ContaminationMatrix,
) -> pd.DataFrame:
    """Monash-style: one fixed horizon per series, already resolved in the parquet."""
    rows: list[ResultRow] = []
    for window in load_monash_forecasting(parquet_path):
        horizon = len(window.future)
        # same reasoning as run_zero_shot_ltsf: model_kwargs is the caller's
        # complete constructor kwargs already, nothing gets injected here.
        # For a trained model that needs context_len/horizon fixed at
        # construction (DLinear-style), the caller should build model_kwargs
        # per-series outside this helper instead, Monash series lengths vary
        # too much to assume one context_len fits all of them anyway.
        model = build_forecast_model(model_name, **model_kwargs)
        model.fit(window.train_series)
        rows.append(evaluate_window(model, window, horizon, dataset_name, contamination))
    return pd.DataFrame([asdict(r) for r in rows])
