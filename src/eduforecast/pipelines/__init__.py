"""src/eduforecast/pipelines/__init__.py"""

from .run_eda_births import run_eda_births
from .run_etl import run_etl
from .run_forecast import run_forecast
from .run_train import run_train

__all__ = [
    "run_eda_births",
    "run_etl",
    "run_forecast",
    "run_train",
]
