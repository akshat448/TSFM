"""
Model name -> adapter class lookup. eval_runner.py only ever goes through
this registry, it never imports a model library directly, so adding model 16
means adding one line here plus one adapter file, nothing else changes.

Wave 2+ adapters (chronos_bolt, timesfm, ttm) import their model libraries
lazily inside the adapter's own __init__, not at module load time, so this
file can be imported (and Wave 1 models run) even before those pip packages
are installed. See PHASE1_PLAN.md section 4 for the wave ordering.
"""
from __future__ import annotations

from .models.base import ForecastModel
from .models.seasonal_naive import SeasonalNaive
from .models.dlinear import DLinear

_FORECAST_MODEL_BUILDERS = {
    "seasonal_naive": lambda **kw: SeasonalNaive(**kw),
    "dlinear": lambda **kw: DLinear(**kw),
}


def register_forecast_model(name: str, builder) -> None:
    """Call this to add a model without editing this file, e.g. from a
    notebook or a project-specific extension script."""
    _FORECAST_MODEL_BUILDERS[name] = builder


def build_forecast_model(name: str, **kwargs) -> ForecastModel:
    if name not in _FORECAST_MODEL_BUILDERS:
        raise KeyError(
            f"unknown model '{name}', registered models: {sorted(_FORECAST_MODEL_BUILDERS)}. "
            f"Wave 2/3 models (chronos_bolt, timesfm, ttm, ...) need their adapter registered "
            f"explicitly once the pip package is installed, see models/chronos_bolt.py etc."
        )
    return _FORECAST_MODEL_BUILDERS[name](**kwargs)


def try_register_wave2() -> list[str]:
    """
    Attempts to register the Wave 2 light-TSFM adapters, skipping any whose
    pip package isn't installed yet rather than crashing the whole registry.
    Returns the list of adapter names that failed to register, so the caller
    can print a clear "install X to use Y" message instead of a silent gap.
    """
    failed = []
    try:
        from .models.chronos_bolt import ChronosBoltAdapter
        register_forecast_model("chronos_bolt", lambda **kw: ChronosBoltAdapter(**kw))
    except ImportError:
        failed.append("chronos_bolt (pip install chronos-forecasting)")
    try:
        from .models.timesfm import TimesFMAdapter
        register_forecast_model("timesfm", lambda **kw: TimesFMAdapter(**kw))
    except ImportError:
        failed.append("timesfm (pip install timesfm)")
    try:
        from .models.ttm import TTMAdapter
        register_forecast_model("ttm", lambda **kw: TTMAdapter(**kw))
    except ImportError:
        failed.append("ttm (pip install granite-tsfm / tsfm_public)")
    return failed


def register_pretrained_models() -> None:
    """Register all approved zero-shot TSFM families without importing extras eagerly."""
    from .models.chronos_2 import Chronos2Adapter
    from .models.moirai import MoiraiAdapter, MoiraiMoEAdapter
    from .models.moment import MomentAdapter
    from .models.chronos_bolt import ChronosBoltAdapter
    from .models.timesfm import TimesFMAdapter
    from .models.ttm import TTMAdapter

    register_forecast_model("chronos_bolt", lambda **kw: ChronosBoltAdapter(**kw))
    register_forecast_model("timesfm", lambda **kw: TimesFMAdapter(**kw))
    register_forecast_model("moirai_1_1_r", lambda **kw: MoiraiAdapter(**kw))
    register_forecast_model("moirai_moe", lambda **kw: MoiraiMoEAdapter(**kw))
    register_forecast_model("moment_1_large", lambda **kw: MomentAdapter(**kw))
    register_forecast_model("ttm_r2", lambda **kw: TTMAdapter(**kw))
    register_forecast_model("chronos_2", lambda **kw: Chronos2Adapter(**kw))
