"""
Commands Initialization

This module initializes and registers all Typer subcommands for the CLI application.
"""

import typer

# Import individual command groups (Typer apps)
from .indicator import indicator_app
from .price import price_app
# from .analyzer import analyzer_app  # Removed to avoid import loop
from .config import config_app
from .status import status_app
from .clean import clean_app
from .multi_timeframe import mtf_app # mtf for multi_timeframe
from .adaptive import adaptive_app
from .risk import risk_app # Added risk_app

# Create the main Typer app for all commands
main_app = typer.Typer(
    name="analysis_cli",
    help="A CLI tool for advanced market analysis, providing insights and educational content."
)

# Register subcommands
main_app.add_typer(indicator_app, name="indicator", help="Access technical indicators.")
main_app.add_typer(price_app, name="price", help="Fetch and display price data.")
# main_app.add_typer(analyzer_app, name="analyze", help="Run market analysis.")  # Removed
main_app.add_typer(config_app, name="config", help="Manage application configuration.")
main_app.add_typer(status_app, name="status", help="Check application and data status.")
main_app.add_typer(clean_app, name="clean", help="Manage cache and temporary files.")
main_app.add_typer(mtf_app, name="mtf", help="Multi-timeframe analysis tools.")
main_app.add_typer(adaptive_app, name="adaptive", help="Adaptive and ML-enhanced indicators.")
main_app.add_typer(risk_app, name="risk", help="Risk management tools.") # Added risk_app registration

__all__ = ["main_app"]