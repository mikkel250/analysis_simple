"""
Status Command Handler

Handles the 'status' command for displaying API call status and cache information.
"""

import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, List, Tuple

import typer
from tabulate import tabulate
import humanize

from src.cli.display import (
    display_info, 
    display_error, 
    display_success, 
    display_warning
)
from src.services.cache_service import get_cache_status

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
status_app = typer.Typer()


def format_cache_status(status: Dict[str, Any]) -> str:
    """
    Format cache status for display.
    
    Args:
        status: Cache status data
        
    Returns:
        str: Formatted table for display
    """
    formatted_data = []
    
    # General cache info
    formatted_data.append(["Cache Directory", status.get("cache_dir", "Not specified")])
    formatted_data.append(["Total Cache Files", str(status.get("total_files", 0))])
    formatted_data.append(["Total Cache Size", humanize.naturalsize(status.get("total_size", 0))])
    
    # Format the cache files by type
    cache_by_type = status.get("files_by_type", {})
    if cache_by_type:
        formatted_data.append(["", ""])
        formatted_data.append(["Cache by Type", "Count"])
        
        for cache_type, count in cache_by_type.items():
            formatted_data.append([cache_type.capitalize(), str(count)])
    
    # Format the cache files by symbol
    cache_by_symbol = status.get("files_by_symbol", {})
    if cache_by_symbol:
        formatted_data.append(["", ""])
        formatted_data.append(["Cache by Symbol", "Count"])
        
        for symbol, count in sorted(cache_by_symbol.items(), key=lambda x: x[1], reverse=True)[:10]:
            formatted_data.append([symbol.upper(), str(count)])
    
    # Format recent cache files
    recent_files = status.get("recent_files", [])
    if recent_files:
        formatted_data.append(["", ""])
        formatted_data.append(["Recent Cache Files", "Last Modified"])
        
        for file_info in recent_files[:10]:
            file_name = os.path.basename(file_info.get("path", ""))
            modified_time = datetime.fromtimestamp(file_info.get("mtime", 0)).strftime("%Y-%m-%d %H:%M:%S")
            formatted_data.append([file_name, modified_time])
    
    # Format oldest cache files
    oldest_files = status.get("oldest_files", [])
    if oldest_files:
        formatted_data.append(["", ""])
        formatted_data.append(["Oldest Cache Files", "Last Modified"])
        
        for file_info in oldest_files[:10]:
            file_name = os.path.basename(file_info.get("path", ""))
            modified_time = datetime.fromtimestamp(file_info.get("mtime", 0)).strftime("%Y-%m-%d %H:%M:%S")
            formatted_data.append([file_name, modified_time])
    
    # Format cache statistics
    cache_stats = status.get("statistics", {})
    if cache_stats:
        formatted_data.append(["", ""])
        formatted_data.append(["Cache Statistics", "Value"])
        
        for stat_name, stat_value in cache_stats.items():
            if isinstance(stat_value, (int, float)):
                formatted_data.append([stat_name.replace("_", " ").capitalize(), str(stat_value)])
    
    return tabulate(formatted_data, tablefmt="fancy_grid")


@status_app.callback()
def callback():
    """Handle status commands."""
    pass


@status_app.command()
def cache():
    """Display information about the cache."""
    try:
        # Get cache status
        cache_status = get_cache_status()
        
        if not cache_status:
            display_warning("No cache information available.")
            return
        
        # Format and display the cache status
        formatted_output = format_cache_status(cache_status)
        print(formatted_output)
        
        display_success("Cache status retrieved successfully")
        
    except Exception as e:
        display_error(f"Error getting cache status: {str(e)}")
        logger.exception("Error in status command")


@status_app.command()
def api():
    """Display information about API usage."""
    try:
        # This would ideally check API rate limits, usage statistics, etc.
        # For now, we'll just display a placeholder message
        formatted_data = []
        
        formatted_data.append(["API", "CoinGecko"])
        formatted_data.append(["Plan", "Free Tier"])
        formatted_data.append(["Rate Limit", "10-50 calls per minute"])
        formatted_data.append(["Remaining Calls", "Unknown (not tracked in free tier)"])
        
        print(tabulate(formatted_data, tablefmt="fancy_grid"))
        
        display_info("Using local calculations for technical indicators.")
        display_info("No API rate limits apply to indicator calculations.")
        display_info("CoinGecko API is only used for fetching price data.")
        
    except Exception as e:
        display_error(f"Error checking API status: {str(e)}")
        logger.exception("Error in API status command") 