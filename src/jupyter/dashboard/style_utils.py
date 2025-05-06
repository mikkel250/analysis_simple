"""
Style Utilities for Cryptocurrency Dashboard

This module provides utility functions for determining colors and styling
elements based on value trends and market conditions.
"""

from typing import Dict, Any, Union, List, Tuple, Optional


def get_color_for_value(value: float) -> str:
    """
    Get the appropriate color for a value based on whether it's positive or negative.
    
    Args:
        value: The numeric value to determine color for
        
    Returns:
        CSS color string (green for positive, red for negative, orange for zero)
    """
    if value > 0:
        return "chartreuse"  # Green for positive values
    elif value < 0:
        return "red"  # Red for negative values
    else:
        return "orange"  # Orange for zero/neutral values


def get_trend_color(trend: str) -> str:
    """
    Get the appropriate color for a market trend.
    
    Args:
        trend: Trend string ("bullish", "bearish", or "neutral")
        
    Returns:
        CSS color string
    """
    trend = trend.lower() if trend else "neutral"
    if trend == "bullish":
        return "chartreuse"
    elif trend == "bearish":
        return "red"
    else:
        return "orange"  # Default/neutral


def get_trend_symbol(trend: str) -> str:
    """
    Get the appropriate symbol for a market trend.
    
    Args:
        trend: Trend string ("bullish", "bearish", or "neutral")
        
    Returns:
        Unicode symbol representing the trend
    """
    trend = trend.lower() if trend else "neutral"
    if trend == "bullish":
        return "▲"  # Up triangle for bullish
    elif trend == "bearish":
        return "▼"  # Down triangle for bearish
    else:
        return "◆"  # Diamond for neutral


def format_value_with_color(value: float, prefix: str = '', suffix: str = '') -> str:
    """
    Format a numeric value with color coding and optional prefix/suffix.
    
    Args:
        value: The numeric value to format
        prefix: Optional prefix string (e.g., "$" or "+")
        suffix: Optional suffix string (e.g., "%" or " USD")
        
    Returns:
        HTML string with the formatted value
    """
    color = get_color_for_value(value)
    formatted_value = f"{prefix}{value:.2f}{suffix}"
    return f'<span style="color: {color}">{formatted_value}</span>' 