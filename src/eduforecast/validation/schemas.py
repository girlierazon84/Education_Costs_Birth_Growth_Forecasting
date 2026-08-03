"""src/eduforecast/validation/schemas.py"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SchemaSpec:
    """Minimal schema specification for a DataFrame."""
    name: str
    required_cols: tuple[str, ...]
    dtype_hints: dict[str, str] | None = None  # e.g. {"Year": "int", "Region_Code": "string"}


def _missing_cols(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    req = list(required)
    return [c for c in req if c not in df.columns]


# ---- Canonical schema specs (aligned with your project) ----

BIRTHS_CANONICAL = SchemaSpec(
    name="births_canonical",
    required_cols=("Region_Code", "Region_Name", "Year", "Number"),
    dtype_hints={"Region_Code": "string", "Region_Name": "string", "Year": "int", "Number": "float"},
)

MORTALITY_CANONICAL = SchemaSpec(
    name="mortality_canonical",
    required_cols=("Region_Code", "Region_Name", "Age", "Year", "Number"),
    dtype_hints={"Region_Code": "string", "Region_Name": "string", "Age": "int", "Year": "int", "Number": "float"},
)

POPULATION_CANONICAL = SchemaSpec(
    name="population_canonical",
    required_cols=("Region_Code", "Region_Name", "Age", "Year", "Number"),
    dtype_hints={"Region_Code": "string", "Region_Name": "string", "Age": "int", "Year": "int", "Number": "float"},
)

MIGRATION_CANONICAL = SchemaSpec(
    name="migration_canonical",
    required_cols=("Region_Code", "Region_Name", "Age", "Year", "Number"),
    dtype_hints={"Region_Code": "string", "Region_Name": "string", "Age": "int", "Year": "int", "Number": "float"},
)

COSTS_PER_CHILD_CANONICAL = SchemaSpec(
    name="costs_per_child_canonical",
    required_cols=("Year", "Fixed_cost_per_child_kr", "Current_cost_per_child_kr"),
    dtype_hints={"Year": "int", "Fixed_cost_per_child_kr": "float", "Current_cost_per_child_kr": "float"},
)

BIRTHS_FORECAST = SchemaSpec(
    name="births_forecast",
    required_cols=("Region_Code", "Year", "Model", "Forecast_Births"),
    dtype_hints={"Region_Code": "string", "Year": "int", "Model": "string", "Forecast_Births": "float"},
)

POP_0_19_FORECAST = SchemaSpec(
    name="population_0_19_forecast",
    required_cols=("Region_Code", "Year", "Age", "Forecast_Population"),
    dtype_hints={"Region_Code": "string", "Year": "int", "Age": "int", "Forecast_Population": "float"},
)

EDU_COSTS_FORECAST = SchemaSpec(
    name="education_costs_forecast",
    required_cols=("Region_Code", "Year", "School_Type", "Forecast_Students", "Fixed_Total_Cost_kr", "Current_Total_Cost_kr"),
    dtype_hints={
        "Region_Code": "string",
        "Year": "int",
        "School_Type": "string",
        "Forecast_Students": "float",
        "Fixed_Total_Cost_kr": "float",
        "Current_Total_Cost_kr": "float",
    },
)


def assert_schema(df: pd.DataFrame, spec: SchemaSpec) -> None:
    """
    Raise a KeyError if required columns are missing,
    and raise a TypeError if types diverge from SchemaSpec definitions.
    """
    # 1. Structural column check
    missing = _missing_cols(df, spec.required_cols)
    if missing:
        raise KeyError(
            f"{spec.name}: missing columns {missing}. Found: {list(df.columns)}"
        )

    # 2. ✅ TYPE VALIDATION LAYER: Verify data types against structural rules
    if spec.dtype_hints:
        for col, hint in spec.dtype_hints.items():
            if col not in df.columns:
                continue

            actual_type = str(df[col].dtype)

            if hint == "int":
                # Accept both standard int64 and pandas Nullable Int64
                if not ("int" in actual_type.lower() or "integer" in actual_type.lower()):
                    raise TypeError(f"{spec.name}: Column '{col}' expected integer type, got {actual_type}")
            elif hint == "float":
                if not ("float" in actual_type.lower() or "double" in actual_type.lower()):
                    raise TypeError(f"{spec.name}: Column '{col}' expected float type, got {actual_type}")
            # ✅ TYPSÄKRING: Inkludera explicit stöd för råa 'str'-klasser bredvid object/string
            elif hint == "string":
                # Accepterar standard-pandas 'object', den dedikerade typen 'string' samt råa 'str'-representationer
                actual_lower = actual_type.lower()
                if not ("string" in actual_lower or "object" in actual_lower or "str" in actual_lower):
                    raise TypeError(f"{spec.name}: Column '{col}' expected string/object type, got {actual_type}")
