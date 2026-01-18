#!/usr/bin/env bash
set -euo pipefail

# scripts/run_all.sh
# One-command project runner:
# 1) Validate inputs
# 2) Backfill DB
# 3) Run forecasting pipeline
#
# Usage:
#   bash scripts/run_all.sh
#
# Note: assumes you are in project root and venv is active.

echo "=== EduForecast: run_all ==="

echo "[1/3] Validate inputs"
python scripts/download_data.py

echo "[2/3] Backfill SQLite database"
python scripts/backfill_db.py

echo "[3/3] Run forecasting pipeline"
eduforecast forecast

echo "✅ Done. Artifacts are in: artifacts/"
echo "Tip: run dashboard with: streamlit run dashboards/Home.py"
