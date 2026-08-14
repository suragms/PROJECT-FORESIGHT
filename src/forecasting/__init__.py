"""Phase 12 production forecasting package.

`baselines` is the unchanged Phase 7/8 dashboard engine (moved from
src/forecasting.py so this directory can be a package). Production
inference uses Phase 11 `FinalForecaster` via `inference.ForecastEngine`.
"""

from src.forecasting.baselines import (  # noqa: F401
    BaselineForecaster,
    MLDemandForecaster,
    generate_multi_step_forecast,
    train_and_benchmark_models,
)
from src.forecasting.inference import ForecastEngine  # noqa: F401

__all__ = [
    "BaselineForecaster",
    "MLDemandForecaster",
    "generate_multi_step_forecast",
    "train_and_benchmark_models",
    "ForecastEngine",
]
