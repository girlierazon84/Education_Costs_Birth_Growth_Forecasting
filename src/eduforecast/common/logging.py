"""src/eduforecast/common/logging.py"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from eduforecast.common.config import AppConfig


def setup_logging(cfg: AppConfig) -> None:
    level_str = str(cfg.logging.get("level", "INFO")).upper()
    level = getattr(logging, level_str, logging.INFO)

    handlers: list[logging.Handler] = []

    console = logging.StreamHandler()
    console.setLevel(level)
    handlers.append(console)

    log_file = cfg.logging.get("file")
    if log_file:
        lf = (cfg.project_root / Path(log_file)).resolve()
        lf.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(lf, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
