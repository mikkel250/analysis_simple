"""
Terminal Display Utilities

Functions for formatting and displaying output in the terminal.
"""

from typing import List, Dict, Any, Union, Optional
from tabulate import tabulate
import datetime
from tqdm import tqdm
import contextlib

def supports_color() -> bool:
    """Check if the current terminal supports color."""
    # Always return False since we're not using colors anymore
    return False

def display_info(message: str) -> None:
    """
    Display informational message
    
    Args:
        message: Message to display
    """
    print(f"ℹ {message}")

def display_success(message: str) -> None:
    """
    Display success message
    
    Args:
        message: Message to display
    """
    print(f"✓ {message}")

def display_warning(message: str) -> None:
    """
    Display warning message
    
    Args:
        message: Message to display
    """
    print(f"⚠ {message}")

def display_error(message: str) -> None:
    """
    Display error message
    
    Args:
        message: Message to display
    """
    print(f"✗ Error: {message}")

def display_table(data: List[List[Any]]) -> None:
    """
    Display data as a table
    
    Args:
        data: Table data (including headers)
    """
    print(tabulate(data, headers="firstrow", tablefmt="grid"))

def format_price(price: Union[float, str], trend: str = 'neutral', use_color: bool = True) -> str:
    """
    Format price data with trend indicators
    
    Args:
        price: The price value
        trend: Trend direction ('up', 'down', or 'neutral')
        use_color: Parameter kept for compatibility, has no effect
        
    Returns:
        Formatted price string with optional trend indicator
    """
    formatted_price = f"{float(price):.2f}" if isinstance(price, (int, float)) else price
    
    # Add trend indicators instead of colors
    if trend == 'up':
        return f"{formatted_price} ↑"
    elif trend == 'down':
        return f"{formatted_price} ↓"
    else:
        return formatted_price

def display_data_age(timestamp: datetime.datetime) -> None:
    """
    Display data age notice
    
    Args:
        timestamp: When the data was last updated
    """
    now = datetime.datetime.now()
    if isinstance(timestamp, str):
        # Try to parse string timestamp if provided
        try:
            timestamp = datetime.datetime.fromisoformat(timestamp)
        except ValueError:
            timestamp = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    
    age_delta = now - timestamp
    age_in_minutes = age_delta.total_seconds() // 60
    
    if age_in_minutes < 5:
        display_info(f"Data is current ({int(age_in_minutes)} minutes old)")
    elif age_in_minutes < 60:
        display_warning(f"Data is {int(age_in_minutes)} minutes old")
    else:
        hours = int(age_in_minutes // 60)
        mins = int(age_in_minutes % 60)
        hour_text = "hour" if hours == 1 else "hours"
        min_text = "minute" if mins == 1 else "minutes"
        display_warning(f"Data is {hours} {hour_text} and {mins} {min_text} old")

@contextlib.contextmanager
def display_spinner(text: str):
    """
    Display a loading spinner with message
    
    Args:
        text: Message to display with spinner
        
    Returns:
        Context manager for spinner
    """
    spinner = tqdm(total=0, desc=text, bar_format="{desc}", dynamic_ncols=True)
    try:
        yield spinner
    finally:
        spinner.close() 