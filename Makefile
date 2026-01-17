# EduForecast Makefile
# Usage examples:
#   make help
#   make install
#   make forecast
#   make dashboard
#   make clean

SHELL := /bin/bash

PY ?= python
PIP ?= pip

# If you want Streamlit to run the multi-page app from Home.py
STREAMLIT_APP ?= dashboards/Home.py

# Forecast CLI
FORECAST_CMD ?= eduforecast forecast

# Artifact paths
ARTIFACTS_DIR ?= artifacts
FORECASTS_DIR ?= artifacts/forecasts
METRICS_DIR ?= artifacts/metrics
FIGURES_DIR ?= artifacts/figures

.PHONY: help install install-dev forecast dashboard dashboard-home lint format test clean clean-artifacts clean-cache

help:
	@echo ""
	@echo "EduForecast targets:"
	@echo "  make install         Install project dependencies (pip install -r requirements.txt if present)"
	@echo "  make install-dev     Install dev deps (requirements-dev.txt if present)"
	@echo "  make forecast        Run pipeline: $(FORECAST_CMD)"
	@echo "  make dashboard       Run Streamlit: streamlit run $(STREAMLIT_APP)"
	@echo "  make dashboard-home  Alias for dashboard"
	@echo "  make test            Run tests (pytest)"
	@echo "  make lint            Lint (ruff) if installed"
	@echo "  make format          Format (ruff format) if installed"
	@echo "  make clean           Remove caches + artifacts"
	@echo "  make clean-artifacts Remove only artifacts/"
	@echo "  make clean-cache     Remove python caches"
	@echo ""

install:
	@if [ -f requirements.txt ]; then \
		$(PIP) install -r requirements.txt; \
	else \
		echo "requirements.txt not found. Skipping."; \
	fi

install-dev:
	@if [ -f requirements-dev.txt ]; then \
		$(PIP) install -r requirements-dev.txt; \
	else \
		echo "requirements-dev.txt not found. Skipping."; \
	fi

forecast:
	@echo "Running: $(FORECAST_CMD)"
	@$(FORECAST_CMD)

dashboard:
	@echo "Starting Streamlit: $(STREAMLIT_APP)"
	@streamlit run $(STREAMLIT_APP)

dashboard-home: dashboard

test:
	@pytest -q

lint:
	@ruff check .

format:
	@ruff format .

clean: clean-cache clean-artifacts
	@echo "Clean complete."

clean-artifacts:
	@echo "Removing $(ARTIFACTS_DIR)/"
	@rm -rf "$(ARTIFACTS_DIR)" || true
	@mkdir -p "$(FORECASTS_DIR)" "$(METRICS_DIR)" "$(FIGURES_DIR)"
	@touch "$(ARTIFACTS_DIR)/.gitkeep" 2>/dev/null || true

clean-cache:
	@echo "Removing Python cache files..."
	@find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
