"""
Chronos-Bolt adapter. Zero-shot only, fit() is a no-op.

NOT independently tested in this session, no GPU or internet access here to
pull real weights. The call shapes below match chronos-forecasting's
documented API as of this project's last training data, but that package's
interface has moved before (Chronos vs Chronos-Bolt already needed different
pipeline classes at one point). Before trusting this for a real run:
  1. `pip show chronos-forecasting` and check the version against what's documented
     for the `amazon/chronos-bolt-base` checkpoint at the time you're reading this.
  2. Run the __main__ smoke test at the bottom against a tiny synthetic series
     and eyeball that predict() returns sane numbers before pointing it at real data.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ForecastResult
from ..metrics import QUANTILE_LEVELS


class ChronosBoltAdapter(ForecastModel):
    name = "chronos_bolt"
    supports_probabilistic = True

    def __init__(self, checkpoint: str = "amazon/chronos-bolt-base", device: str | None = None):
        import torch
        from chronos import BaseChronosPipeline  # pip install chronos-forecasting

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline = BaseChronosPipeline.from_pretrained(
            checkpoint,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

    def fit(self, context: np.ndarray, **kwargs) -> None:
        pass  # zero-shot

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        import torch
        context_t = torch.tensor(np.asarray(context, dtype=np.float32))
        quantile_levels = kwargs.get("quantile_levels", QUANTILE_LEVELS)
        quantiles, mean = self._pipeline.predict_quantiles(
            context=context_t,
            prediction_length=horizon,
            quantile_levels=quantile_levels,
        )
        # quantiles: (1, horizon, n_quantiles), mean: (1, horizon) per the documented shape,
        # VERIFY this against your installed version, this is the most likely thing to have drifted
        quantiles_np = quantiles.squeeze(0).cpu().numpy()  # (horizon, n_quantiles)
        point = mean.squeeze(0).cpu().numpy()
        q_dict = {q: quantiles_np[:, i] for i, q in enumerate(quantile_levels)}
        return ForecastResult(point=point, quantiles=q_dict)


if __name__ == "__main__":
    # smoke test, run this directly once the package is installed, before wiring into a real eval
    synthetic = np.sin(np.linspace(0, 20, 200)).astype(np.float32) + 10
    model = ChronosBoltAdapter()
    result = model.predict(synthetic, horizon=24)
    print("point shape:", result.point.shape)
    print("point[:5]:", result.point[:5])
    assert result.point.shape == (24,)
    print("smoke test passed, shapes look right, eyeball the values for sanity")
