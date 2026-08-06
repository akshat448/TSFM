"""
TimesFM-2.5 adapter. Zero-shot only, fit() is a no-op.

Same caveat as chronos_bolt.py: not tested in this session, no GPU/internet
here. TimesFM's Python API has changed shape across major versions (1.0 vs
2.x checkpoint loading in particular), verify the checkpoint loading call
below against whatever `timesfm` version ends up installed before trusting
real numbers out of this.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ForecastResult
from ..metrics import QUANTILE_LEVELS


class TimesFMAdapter(ForecastModel):
    name = "timesfm"
    supports_probabilistic = True

    def __init__(self, checkpoint: str = "google/timesfm-2.5-200m-pytorch",
                 context_len: int = 512, horizon_len: int = 128, device: str | None = None):
        import timesfm  # pip install timesfm

        self.context_len = context_len
        self.horizon_len = horizon_len
        # VERIFY: constructor args and checkpoint loading method against the
        # installed timesfm version, this call shape is the most likely thing
        # to have drifted since this adapter was written
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                context_len=context_len,
                horizon_len=horizon_len,
                backend="gpu" if device != "cpu" else "cpu",
            ),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
        )

    def fit(self, context: np.ndarray, **kwargs) -> None:
        pass  # zero-shot

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        freq = kwargs.get("freq_code", 0)  # 0=high freq, 1=medium, 2=low, per TimesFM's own convention, set per-dataset at call time
        point_forecast, quantile_forecast = self._model.forecast(
            [np.asarray(context, dtype=np.float32)],
            freq=[freq],
        )
        point = np.asarray(point_forecast[0])[:horizon]
        quantile_levels = kwargs.get("quantile_levels", QUANTILE_LEVELS)
        q_array = np.asarray(quantile_forecast[0])  # shape (horizon, n_quantiles), VERIFY column order matches quantile_levels for your version
        q_dict = None
        if q_array.ndim == 2 and q_array.shape[1] >= len(quantile_levels):
            q_dict = {q: q_array[:horizon, i] for i, q in enumerate(quantile_levels)}
        return ForecastResult(point=point, quantiles=q_dict)


if __name__ == "__main__":
    synthetic = np.sin(np.linspace(0, 20, 512)).astype(np.float32) + 10
    model = TimesFMAdapter(context_len=512, horizon_len=128)
    result = model.predict(synthetic, horizon=96)
    print("point shape:", result.point.shape)
    assert result.point.shape == (96,)
    print("smoke test passed, shapes look right, eyeball the values for sanity")
