"""
src/eduforecast/common/paths.py

eduforecast.common.paths

Central place for project path helpers.
(Placeholder module — extend as the project grows.)
"""

from __future__ import annotations

from pathlib import Path


def project_root_from_file(file_path: str | Path, *, parents: int = 1) -> Path:
    """
    Resolve a project root relative to a file path.

    Example:
        project_root_from_file(__file__, parents=2)

    Args:
        file_path: Typically __file__.
        parents: How many parent levels to go up.

    Returns:
        Path to the inferred project root.
    """
    p = Path(file_path).resolve()
    return p.parents[parents]
