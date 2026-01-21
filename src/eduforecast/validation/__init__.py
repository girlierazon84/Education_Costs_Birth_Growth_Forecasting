"""src/eduforecast/validation/__init__.py"""

from __future__ import annotations

from .checks import (
    CheckResult,
    validate_df,
    validate_births_canonical,
    validate_mortality_canonical,
    validate_population_canonical,
)
from .schemas import (
    SchemaSpec,
    assert_schema,
    BIRTHS_CANONICAL,
    MORTALITY_CANONICAL,
    POPULATION_CANONICAL,
    MIGRATION_CANONICAL,
    COSTS_PER_CHILD_CANONICAL,
    BIRTHS_FORECAST,
    POP_0_19_FORECAST,
    EDU_COSTS_FORECAST,
)

__all__ = [
    # checks
    "CheckResult",
    "validate_df",
    "validate_births_canonical",
    "validate_mortality_canonical",
    "validate_population_canonical",
    # schemas
    "SchemaSpec",
    "assert_schema",
    "BIRTHS_CANONICAL",
    "MORTALITY_CANONICAL",
    "POPULATION_CANONICAL",
    "MIGRATION_CANONICAL",
    "COSTS_PER_CHILD_CANONICAL",
    "BIRTHS_FORECAST",
    "POP_0_19_FORECAST",
    "EDU_COSTS_FORECAST",
]
