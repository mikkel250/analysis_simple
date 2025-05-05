"""
Clean Command Handler

Handles the 'clean' command for cleaning the cache.
"""

import os
import logging
from typing import Dict, Any, List, Optional

import typer
from tabulate import tabulate

from src.cli.display import (
    display_info, 
    display_error, 
    display_success, 
    display_warning,
    display_spinner
)
from src.services.cache_service import (
    clear_cache_by_age,
    clear_cache_by_type,
    clear_cache_by_symbol,
    clear_all_cache
)

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
clean_app = typer.Typer()


@clean_app.callback()
def callback():
    """Handle clean commands."""
    pass


@clean_app.command()
def all(
    force: bool = typer.Option(False, "--force", "-f", help="Force clean without confirmation"),
):
    """
    Clean all cached data.
    """
    try:
        if not force:
            # Ask for confirmation
            confirmed = typer.confirm("Are you sure you want to clear all cached data?")
            if not confirmed:
                display_info("Operation canceled.")
                return
        
        # Clean all cache
        with display_spinner("Clearing all cached data..."):
            result = clear_all_cache()
        
        if result:
            display_success(f"Successfully cleared all cached data. Removed {result} file(s).")
        else:
            display_warning("No cached data found to clear.")
        
    except Exception as e:
        display_error(f"Error clearing cache: {str(e)}")
        logger.exception("Error in clean command")


@clean_app.command()
def by_age(
    days: int = typer.Option(30, "--days", "-d", help="Clear cache older than specified days"),
    force: bool = typer.Option(False, "--force", "-f", help="Force clean without confirmation"),
):
    """
    Clean cached data older than specified days.
    """
    try:
        if not force:
            # Ask for confirmation
            confirmed = typer.confirm(f"Are you sure you want to clear all cached data older than {days} days?")
            if not confirmed:
                display_info("Operation canceled.")
                return
        
        # Clean cache by age
        with display_spinner(f"Clearing cached data older than {days} days..."):
            result = clear_cache_by_age(days)
        
        if result:
            display_success(f"Successfully cleared cached data older than {days} days. Removed {result} file(s).")
        else:
            display_warning(f"No cached data older than {days} days found.")
        
    except Exception as e:
        display_error(f"Error clearing cache: {str(e)}")
        logger.exception("Error in clean by age command")


@clean_app.command()
def by_type(
    type_name: str = typer.Argument(..., help="Cache type to clear (e.g., 'price', 'indicator')"),
    force: bool = typer.Option(False, "--force", "-f", help="Force clean without confirmation"),
):
    """
    Clean cached data of a specific type.
    """
    try:
        if not force:
            # Ask for confirmation
            confirmed = typer.confirm(f"Are you sure you want to clear all cached {type_name} data?")
            if not confirmed:
                display_info("Operation canceled.")
                return
        
        # Clean cache by type
        with display_spinner(f"Clearing cached {type_name} data..."):
            result = clear_cache_by_type(type_name)
        
        if result:
            display_success(f"Successfully cleared cached {type_name} data. Removed {result} file(s).")
        else:
            display_warning(f"No cached {type_name} data found.")
        
    except Exception as e:
        display_error(f"Error clearing cache: {str(e)}")
        logger.exception("Error in clean by type command")


@clean_app.command()
def by_symbol(
    symbol: str = typer.Argument(..., help="Symbol to clear cache for (e.g., 'BTC')"),
    force: bool = typer.Option(False, "--force", "-f", help="Force clean without confirmation"),
):
    """
    Clean cached data for a specific symbol.
    """
    try:
        if not force:
            # Ask for confirmation
            confirmed = typer.confirm(f"Are you sure you want to clear all cached data for {symbol.upper()}?")
            if not confirmed:
                display_info("Operation canceled.")
                return
        
        # Clean cache by symbol
        with display_spinner(f"Clearing cached data for {symbol.upper()}..."):
            result = clear_cache_by_symbol(symbol)
        
        if result:
            display_success(f"Successfully cleared cached data for {symbol.upper()}. Removed {result} file(s).")
        else:
            display_warning(f"No cached data found for {symbol.upper()}.")
        
    except Exception as e:
        display_error(f"Error clearing cache: {str(e)}")
        logger.exception("Error in clean by symbol command") 