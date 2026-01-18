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
    config_path = _as_path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(raw=raw, config_path=config_path)
