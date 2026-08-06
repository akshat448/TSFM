"""
Seasonal Naive. Non-negotiable baseline per BENCHMARK_SPEC.md, every TSFM
claim in this project has to beat it. No training, pure lookup.
"""
from __future__ import annotations

import numpy as np

from .base import ForecastModel, ForecastResult


class SeasonalNaive(ForecastModel):
    """
    Forecast at step h is the observed value exactly one season back from
    step h. If the context is shorter than the season length, falls back to
    plain last-value naive (season = 1) rather than raising, since some
    Monash sub-series are short.
    """

    name = "seasonal_naive"
    supports_probabilistic = False

    def __init__(self, season_length: int):
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length

    def fit(self, context: np.ndarray, **kwargs) -> None:
        pass  # nothing to fit

    def predict(self, context: np.ndarray, horizon: int, **kwargs) -> ForecastResult:
        context = np.asarray(context, dtype=np.float32)
        season = self.season_length if len(context) >= self.season_length else 1
        # tile the last `season` observed values forward to cover horizon
        last_season = context[-season:]
        reps = int(np.ceil(horizon / season))
        point = np.tile(last_season, reps)[:horizon]
        return ForecastResult(point=point, quantiles=None)


# season length lookup by dataset frequency, used by eval_runner to construct
# a SeasonalNaive instance per dataset without hardcoding it dataset by dataset
SEASON_LENGTH_BY_FREQ = {
    "h": 24,        # hourly -> daily season
    "15min": 96,    # 15-min -> daily season (24*4)
    "10min": 144,   # 10-min -> daily season (24*6)
    "d": 7,         # daily -> weekly season
    "w": 1,         # weekly -> no strong sub-season, falls back to last-value
    "m": 12,        # monthly -> yearly season
    "q": 4,         # quarterly -> yearly season
    "y": 1,
    "0.5h": 48,     # half-hourly -> daily season
}
