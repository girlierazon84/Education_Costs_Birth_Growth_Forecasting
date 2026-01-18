"""
scripts/download_data.py

Lightweight "data bootstrap" script:
- Validates that required input files exist (or downloads them if you later add URLs)
- Creates expected folder structure
- Prints a clear status report

This repo is intentionally conservative about auto-downloading because SCB data often
comes via APIs, saved queries, or manual exports. Use this as a reproducibility hub.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RequiredFile:
    rel_path: str
    description: str


REQUIRED_FILES: list[RequiredFile] = [
    RequiredFile(
        "data/raw/birth_data_per_region.csv",
        "Births by region and year (raw). Expected cols e.g. Region_Code, Year, Total_Births",
    ),
    RequiredFile(
        "data/external/grundskola_costs_per_child.csv",
        "Grundskola cost per child table (external).",
    ),
    RequiredFile(
        "data/external/gymnasieskola_costs_per_child.csv",
        "Gymnasieskola cost per child table (external).",
    ),
]


def ensure_dirs() -> None:
    """Create expected directories if missing."""
    for rel in [
        "data/raw",
        "data/external",
        "data/processed",
        "artifacts/forecasts",
        "artifacts/metrics",
        "artifacts/figures",
        "artifacts/models",
    ]:
        (PROJECT_ROOT / rel).mkdir(parents=True, exist_ok=True)


def check_required_files() -> tuple[list[RequiredFile], list[RequiredFile]]:
    """Return (present, missing)."""
    present, missing = [], []
    for rf in REQUIRED_FILES:
        if (PROJECT_ROOT / rf.rel_path).exists():
            present.append(rf)
        else:
            missing.append(rf)
    return present, missing


def print_report(present: list[RequiredFile], missing: list[RequiredFile]) -> None:
    print("\nEduForecast • Data bootstrap report\n" + "-" * 40)
    print(f"Project root: {PROJECT_ROOT}")

    if present:
        print("\n✅ Present files:")
        for rf in present:
            print(f"  - {rf.rel_path} :: {rf.description}")

    if missing:
        print("\n❌ Missing files:")
        for rf in missing:
            print(f"  - {rf.rel_path} :: {rf.description}")

        print(
            "\nHow to fix:\n"
            "1) Add the missing CSV(s) in the specified path.\n"
            "2) Re-run this script.\n\n"
            "Tip: If you later decide to automate downloads (SCB API / URLs),\n"
            "you can implement that logic here in a controlled way."
        )


def main() -> int:
    ensure_dirs()
    present, missing = check_required_files()
    print_report(present, missing)

    # Non-zero exit if missing, so CI/Makefile can detect it
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
