"""src/eduforecast/cli.py"""

from __future__ import annotations

import typer
from rich import print

from eduforecast.common.config import load_config
from eduforecast.common.logging import setup_logging
from eduforecast.pipelines.run_eda_births import run_eda_births
from eduforecast.pipelines.run_etl import run_etl
from eduforecast.pipelines.run_forecast import run_forecast
from eduforecast.pipelines.run_train import run_train

app = typer.Typer(help="Education costs & birth growth forecasting CLI")

DEFAULT_CONFIG = "configs/config.yaml"


@app.command()
def init(config_path: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config")) -> None:
    """Create expected directories from config (data/, artifacts/, etc.)."""
    cfg = load_config(config_path)
    setup_logging(cfg)

    created = cfg.ensure_directories()
    print("[bold green]Init complete.[/bold green]")
    if created:
        print("Created directories:")
        for p in created:
            print(f"  - {p}")
    else:
        print("No directories needed (already exist).")


@app.command()
def etl(config_path: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config")) -> None:
    """Run ETL: raw → processed + SQLite."""
    cfg = load_config(config_path)
    setup_logging(cfg)
    cfg.ensure_directories()
    run_etl(cfg)
    print("[bold green]ETL complete.[/bold green]")


@app.command()
def train(config_path: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config")) -> None:
    """Train and evaluate models; persist artifacts."""
    cfg = load_config(config_path)
    setup_logging(cfg)
    cfg.ensure_directories()
    run_train(cfg)
    print("[bold green]Training complete.[/bold green]")


@app.command()
def forecast(config_path: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config")) -> None:
    """Generate forecasts and compute education costs."""
    cfg = load_config(config_path)
    setup_logging(cfg)
    cfg.ensure_directories()
    run_forecast(cfg)
    print("[bold green]Forecasting complete.[/bold green]")


@app.command()
def run_all(config_path: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config")) -> None:
    """Convenience command: init → etl → train → forecast"""
    cfg = load_config(config_path)
    setup_logging(cfg)
    cfg.ensure_directories()
    run_etl(cfg)
    run_train(cfg)
    run_forecast(cfg)
    print("[bold green]All steps complete.[/bold green]")


@app.command("eda-births")
def eda_births(config_path: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config")) -> None:
    """Run births EDA checks and save CSV outputs."""
    cfg = load_config(config_path)
    setup_logging(cfg)
    cfg.ensure_directories()
    run_eda_births(cfg)
    print("[bold green]Births EDA checks complete.[/bold green]")


if __name__ == "__main__":
    app()
