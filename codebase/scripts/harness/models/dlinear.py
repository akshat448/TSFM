"""
DLinear (Zeng et al., AAAI 2023, "Are Transformers Effective for Time Series
Forecasting?"). Non-negotiable baseline alongside Seasonal Naive per
BENCHMARK_SPEC.md. Decomposes the series into trend + seasonal via a moving
average, fits one linear layer per component, sums the two forecasts.

Trained per-channel on the dataset's own train split. Cheap enough to run on
CPU, but uses torch (already in the project's venv) so it runs on the A100
without extra work if one's free.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .base import ForecastModel, ForecastResult


class _MovingAvg(nn.Module):
    """Moving average for series decomposition, padded at both ends so the
    output length matches the input (replicate-pad, matches the reference
    implementation's own choice, not zero-pad, to avoid an edge bias)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, 1)
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class _DLinearNet(nn.Module):
    def __init__(self, context_len: int, horizon: int, moving_avg_kernel: int = 25):
        super().__init__()
        if moving_avg_kernel % 2 == 0:
            moving_avg_kernel += 1  # keep it odd so the pad is symmetric
        self.decomp = _MovingAvg(moving_avg_kernel)
        self.linear_seasonal = nn.Linear(context_len, horizon)
        self.linear_trend = nn.Linear(context_len, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, context_len, 1)
        trend = self.decomp(x)
        seasonal = x - trend
        out = self.linear_seasonal(seasonal.squeeze(-1)) + self.linear_trend(trend.squeeze(-1))
        return out  # (batch, horizon)


class DLinear(ForecastModel):
    name = "dlinear"
    supports_probabilistic = False

    def __init__(self, context_len: int, horizon: int, epochs: int = 20, lr: float = 5e-3,
                 device: str | None = None, seed: int = 0):
        self.context_len = context_len
        self.horizon = horizon
        self.epochs = epochs
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self._net: _DLinearNet | None = None

    def fit(self, context: np.ndarray, **kwargs) -> None:
        """context here is the FULL training series (not just one lookback window).
        Builds (context_len -> horizon) training pairs by sliding over it."""
        torch.manual_seed(self.seed)
        series = np.asarray(context, dtype=np.float32)
        n = len(series)
        needed = self.context_len + self.horizon
        if n < needed:
            raise ValueError(f"series length {n} shorter than context_len+horizon={needed}")

        xs, ys = [], []
        for start in range(0, n - needed + 1):
            xs.append(series[start:start + self.context_len])
            ys.append(series[start + self.context_len:start + needed])
        X = torch.tensor(np.stack(xs), dtype=torch.float32, device=self.device).unsqueeze(-1)
        Y = torch.tensor(np.stack(ys), dtype=torch.float32, device=self.device)

        self._net = _DLinearNet(self.context_len, self.horizon).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self._net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            pred = self._net(X)
            loss = loss_fn(pred, Y)
            loss.backward()
            opt.step()

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        if self._net is None:
            raise RuntimeError("call fit() before predict()")
        if horizon != self.horizon:
            raise ValueError(f"this instance was fit for horizon={self.horizon}, got {horizon}")
        context = np.asarray(context, dtype=np.float32)
        if len(context) != self.context_len:
            context = context[-self.context_len:]  # use the most recent context_len points
        x = torch.tensor(context, dtype=torch.float32, device=self.device).view(1, -1, 1)
        self._net.eval()
        with torch.no_grad():
            point = self._net(x).squeeze(0).cpu().numpy()
        return ForecastResult(point=point, quantiles=None)
