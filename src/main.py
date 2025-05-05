#!/usr/bin/env python3
"""
BTC-USDT Market Analysis CLI

Main entry point for the CLI application.
Handles command-line parsing and delegates to appropriate modules.
"""

import os
import sys
import typer
from typing import Optional
from colorama import Fore, Style

# Add the src directory to the Python path if running from the project root
src_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import command groups
from src.cli.commands import (
    indicator_app,
    price_app,
    analysis_app,
    status_app,
    clean_app
)

# Create the main app instance
app = typer.Typer(help="BTC-USDT Market Analysis CLI")

# Register command groups
app.add_typer(indicator_app, name="indicator")
app.add_typer(price_app, name="price")
app.add_typer(analysis_app, name="analysis")
app.add_typer(status_app, name="status")
app.add_typer(clean_app, name="clean")

# Display information about the local calculation feature
@app.callback()
def callback():
    """BTC-USDT Market Analysis CLI"""
    print(f"{Fore.BLUE}ℹ Using local calculations for technical indicators.")
    print(f"ℹ No API rate limits apply to indicator calculations.{Style.RESET_ALL}")

if __name__ == "__main__":
    app() 