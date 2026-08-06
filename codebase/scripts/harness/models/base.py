"""
Abstract interfaces every model adapter implements.

The eval runner only ever talks to these two interfaces, it never imports a
specific model library directly. Adding a new model means writing one adapter
file here, not touching eval_runner.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ForecastResult:
    """One prediction for one (series, window, horizon)."""
    point: np.ndarray                      # shape (horizon,), point forecast
    quantiles: dict[float, np.ndarray] | None = None  # {quantile_level: array of shape (horizon,)}, None if model is point-only


class ForecastModel(ABC):
    """
    Interface for point/probabilistic forecasting models (general TSFMs,
    transformer baselines, statistical models).

    Zero-shot models (Chronos-Bolt, TimesFM, TTM) typically implement fit()
    as a no-op and do everything in predict(). Trained baselines (DLinear,
    PatchTST, iTransformer, S-Mamba) use fit() for real, called once per
    (dataset, series or dataset-wide depending on the model).
    """

    name: str = "unnamed_model"
    supports_probabilistic: bool = False

    @abstractmethod
    def fit(self, context: np.ndarray, **kwargs) -> None:
        """context: shape (n_obs,) or (n_series, n_obs) for multivariate-joint models.
        No-op for zero-shot models."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        """context: the lookback window immediately preceding the forecast origin.
        Returns a ForecastResult with a point forecast and, if the model supports it,
        quantile forecasts at standard levels (see metrics.py QUANTILE_LEVELS)."""
        raise NotImplementedError


@dataclass
class ClassificationResult:
    """One prediction for one record (PTB-XL style multi-label classification)."""
    probs: np.ndarray   # shape (n_classes,), predicted probability per diagnostic superclass


class ClassifierModel(ABC):
    """Interface for classification models (ECG-FM, HuBERT-ECG, or a simple baseline)."""

    name: str = "unnamed_classifier"
    class_names: list[str] = field(default_factory=list)

    @abstractmethod
    def fit(self, signals: np.ndarray, labels: np.ndarray, **kwargs) -> None:
        """signals: shape (n_records, n_samples, n_leads). labels: shape (n_records, n_classes), multi-hot."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, signal: np.ndarray, **kwargs) -> ClassificationResult:
        """signal: shape (n_samples, n_leads)."""
        raise NotImplementedError
