#!/usr/bin/env python3
"""
BTC-USDT Market Analysis CLI

Main entry point for the CLI application.
Handles command-line parsing and delegates to appropriate modules.
"""

import os
import sys
import typer

# Add the src directory to the Python path if running from the project root
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import command groups
from src.cli.commands import (
    analyzer_app,
    clean_app,
    config_app,
    indicator_app,
    price_app,
    status_app,
    mta_app,
    adaptive_app
)

# Create the main app instance
app = typer.Typer(help="BTC-USDT Market Analysis CLI")

# Register command groups
app.add_typer(indicator_app, name="indicator")
app.add_typer(price_app, name="price")
app.add_typer(status_app, name="status")
app.add_typer(clean_app, name="clean")
app.add_typer(analyzer_app, name="analyzer")
app.add_typer(config_app, name="config")
app.add_typer(mta_app, name="mta")
app.add_typer(adaptive_app, name="adaptive")

# Display information about the local calculation feature
@app.callback()
def callback():
    """BTC-USDT Market Analysis CLI"""
    print("ℹ Using local calculations for technical indicators.")
    print("ℹ No API rate limits apply to indicator calculations.")

if __name__ == "__main__":
    app() 