"""
src/eduforecast/common/utils.py

eduforecast.common.utils

Small utility helpers used across the codebase.
(Placeholder module — extend as needed.)
"""

from __future__ import annotations

from typing import Any


def safe_int(value: Any, default: int) -> int:
    """Best-effort int conversion with fallback."""
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float) -> float:
    """Best-effort float conversion with fallback."""
    try:
        return float(value)
    except Exception:
        return default
