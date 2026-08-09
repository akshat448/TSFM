"""Moirai 1.1-R and Moirai-MoE adapters via uni2ts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastModel, ForecastResult


class _MoiraiBase(ForecastModel):
    forecast_class_name = "MoiraiForecast"
    model_name = "moirai_1_1_r"
    supports_probabilistic = True

    def __init__(self, checkpoint: str, device: str | None = None, num_samples: int = 100):
        from uni2ts.model.moirai import MoiraiForecast
        from uni2ts.model.moirai_moe import MoiraiMoEForecast

        klass = {"MoiraiForecast": MoiraiForecast, "MoiraiMoEForecast": MoiraiMoEForecast}[self.forecast_class_name]
        self.name = self.model_name
        self._predictor = klass.load_from_checkpoint(checkpoint).create_predictor(batch_size=1, num_samples=num_samples)
        self._num_samples = num_samples

    def fit(self, context: np.ndarray, **kwargs) -> None:
        return None

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        values = np.asarray(context, dtype=np.float32)
        item = {"start": pd.Period("2000-01-01", freq="h"), "target": values}
        forecast = next(self._predictor.predict([item], prediction_length=horizon))
        samples = np.asarray(forecast.samples, dtype=np.float32)
        levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        return ForecastResult(
            point=np.median(samples, axis=0),
            quantiles={q: np.quantile(samples, q, axis=0) for q in levels},
        )


class MoiraiAdapter(_MoiraiBase):
    name = "moirai_1_1_r"
    model_name = "moirai_1_1_r"
    forecast_class_name = "MoiraiForecast"

    def __init__(self, checkpoint: str = "Salesforce/moirai-1.1-R-base", **kwargs):
        super().__init__(checkpoint, **kwargs)


class MoiraiMoEAdapter(_MoiraiBase):
    name = "moirai_moe"
    model_name = "moirai_moe"
    forecast_class_name = "MoiraiMoEForecast"

    def __init__(self, checkpoint: str = "Salesforce/moirai-moe-1.0-R-base", **kwargs):
        super().__init__(checkpoint, **kwargs)
