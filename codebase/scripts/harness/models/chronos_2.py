"""Chronos-2 adapter using chronos-forecasting >=2.0."""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ForecastResult


class Chronos2Adapter(ForecastModel):
    name = "chronos_2"
    supports_probabilistic = True

    def __init__(self, checkpoint: str = "amazon/chronos-2", device: str | None = None):
        import torch
        from chronos import Chronos2Pipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = Chronos2Pipeline.from_pretrained(checkpoint, device_map=self.device)

    def fit(self, context: np.ndarray, **kwargs) -> None:
        return None

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        import pandas as pd

        values = np.asarray(context, dtype=np.float32)
        frame = pd.DataFrame({"item_id": "series", "timestamp": np.arange(len(values)), "target": values})
        out = self._pipeline.predict_df(
            frame,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )
        point = np.asarray(out["0.5"], dtype=np.float32)
        quantiles = {q: np.asarray(out[str(q)], dtype=np.float32) for q in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
        return ForecastResult(point=point, quantiles=quantiles)
