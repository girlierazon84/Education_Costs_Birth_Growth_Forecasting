"""src/eduforecast/common/config.py"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


@dataclass(frozen=True)
class AppConfig:
    """Config wrapper with project-root relative paths."""

    raw: Dict[str, Any]
    config_path: Path

    @property
    def project_root(self) -> Path:
        # configs/config.yaml -> project root is parent of "configs"
        return self.config_path.parent.parent.resolve()

    @property
    def paths(self) -> Dict[str, Path]:
        p = self.raw.get("paths", {})
        return {k: (self.project_root / Path(v)).resolve() for k, v in p.items()}

    @property
    def logging(self) -> Dict[str, Any]:
        return self.raw.get("logging", {})

    @property
    def database(self) -> Dict[str, Any]:
        return self.raw.get("database", {})

    @property
    def forecast(self) -> Dict[str, Any]:
        return self.raw.get("forecast", {})

    @property
    def modeling(self) -> Dict[str, Any]:
        return self.raw.get("modeling", {})

    @property
    def regions(self) -> Dict[str, Any]:
        return self.raw.get("regions", {})

    # ✅ NEW PROPERTY: Expose the feature parameters directly to your cohort loops
    @property
    def features(self) -> Dict[str, Any]:
        return self.raw.get("features", {})

    # ✅ NEW PROPERTY: Expose the unified cost parameter block
    @property
    def costs(self) -> Dict[str, Any]:
        return self.raw.get("costs", {})

    def ensure_directories(self) -> List[str]:
        created: List[str] = []
        for _, path in self.paths.items():
            if path.suffix:  # treat as file path
                path.parent.mkdir(parents=True, exist_ok=True)
                continue
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(str(path))
        return created


def load_config(config_path: str | Path) -> AppConfig:
    """
    Loads the central config layout and recursively merges parameters files
    housed inside a sibling 'params/' subdirectory if present.
    """
    config_path = _as_path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    # ✅ COMPILER ADDITION: Search for nested parameters layers
    params_dir = config_path.parent / "params"
    if params_dir.exists() and params_dir.is_dir():
        for param_file in params_dir.glob("*.yaml"):
            try:
                with param_file.open("r", encoding="utf-8") as pf:
                    sub_raw = yaml.safe_load(pf) or {}

                # Safe marge layout: merge sub-dictionaries into the core config raw root
                # e.g., merging {'costs': {...}} directly over root fields
                for k, v in sub_raw.items():
                    if isinstance(v, dict) and k in raw and isinstance(raw[k], dict):
                        # Deep merge dictionary layers to preserve manual core file overrides
                        raw[k] = {**v, **raw[k]}
                    else:
                        # Append new root sections directly (like 'features')
                        raw[k] = v
            except Exception:
                # Fallback to keep loading the primary configuration file if a sub-file is broken
                pass

    return AppConfig(raw=raw, config_path=config_path)
