"""src/eduforecast/forecasting/__init__.py"""

from .intervals import IntervalResult, empirical_pi, normal_pi
from .predict_births import predict_births_all_regions, predict_births_for_region
from .predict_mortality import predict_mortality_stub
from .predict_population import predict_population_0_19

__all__ = [
    "IntervalResult",
    "normal_pi",
    "empirical_pi",
    "predict_births_for_region",
    "predict_births_all_regions",
    "predict_population_0_19",
    "predict_mortality_stub",
]