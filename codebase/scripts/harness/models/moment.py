"""MOMENT forecasting adapter."""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ForecastResult


class MomentAdapter(ForecastModel):
    name = "moment_1_large"
    supports_probabilistic = False

    def __init__(self, checkpoint: str = "AutonLab/MOMENT-1-large", device: str | None = None):
        import torch
        from momentfm import MOMENTPipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._checkpoint = checkpoint
        self._model_class = MOMENTPipeline
        self._torch = torch

    def fit(self, context: np.ndarray, **kwargs) -> None:
        return None

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        model = self._model_class.from_pretrained(
            self._checkpoint,
            model_kwargs={"task_name": "forecasting", "forecast_horizon": horizon},
        )
        model.init()
        model.to(self.device)
        x = self._torch.tensor(np.asarray(context, dtype=np.float32), device=self.device).view(1, 1, -1)
        with self._torch.no_grad():
            output = model(x_enc=x)
        return ForecastResult(point=output.forecast.squeeze().detach().cpu().numpy()[:horizon])
