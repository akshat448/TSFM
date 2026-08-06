"""
TTM-r2 (IBM Tiny Time Mixers) adapter. Zero-shot only, fit() is a no-op.

Same caveat as the other Wave 2 adapters: not tested in this session.
TTM ships fixed context/horizon combinations per checkpoint variant (it is
not a fully flexible any-length model like Chronos/TimesFM), verify the
checkpoint name matches a context/horizon pair that's actually available
before trusting a run, check the model card on the HF repo listed below.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ForecastResult


class TTMAdapter(ForecastModel):
    name = "ttm"
    supports_probabilistic = False  # TTM is point-forecast only as of the r2 release, no quantile head

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-ttm-r2",
                 context_len: int = 512, horizon_len: int = 96, device: str | None = None):
        import torch
        from tsfm_public import TinyTimeMixerForPrediction  # pip install granite-tsfm

        self.context_len = context_len
        self.horizon_len = horizon_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # VERIFY: revision/branch argument, TTM checkpoints are organized by
        # context/horizon variant on the HF repo (not a single default), pick
        # the branch matching context_len/horizon_len explicitly
        self._model = TinyTimeMixerForPrediction.from_pretrained(
            checkpoint, revision=f"{context_len}-{horizon_len}-r2"
        ).to(self.device)
        self._model.eval()

    def fit(self, context: np.ndarray, **kwargs) -> None:
        pass  # zero-shot

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        import torch
        context = np.asarray(context, dtype=np.float32)
        if len(context) != self.context_len:
            context = context[-self.context_len:]
        x = torch.tensor(context, device=self.device).view(1, self.context_len, 1)
        with torch.no_grad():
            out = self._model(past_values=x)
        point = out.prediction_outputs.squeeze().cpu().numpy()[:horizon]
        return ForecastResult(point=point, quantiles=None)


if __name__ == "__main__":
    synthetic = np.sin(np.linspace(0, 20, 512)).astype(np.float32) + 10
    model = TTMAdapter(context_len=512, horizon_len=96)
    result = model.predict(synthetic, horizon=96)
    print("point shape:", result.point.shape)
    assert result.point.shape == (96,)
    print("smoke test passed, shapes look right, eyeball the values for sanity")
