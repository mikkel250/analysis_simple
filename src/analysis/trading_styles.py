"""
Trading style modifiers for market analysis.

This module provides configuration for different trading timeframes:
- short - For 5m, 15m, 30m timeframes (scalping, HFT, short-term swing trading)
- medium - For intra-day swing trading timeframes
- long - For spot/buy-and-hold timeframes

Each style configures the appropriate time windows and parameters for analysis.
"""

import datetime as dt
from functools import wraps
from typing import Optional, Dict, Any, List, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from src.plotting.charts import plot_technical_analysis
# from src.analysis.market_analyzer import MarketAnalyzer # Removed to break circular import
import logging

logger = logging.getLogger(__name__)

# Global state to store current trading style settings
# This will be accessed by other functions in the kernel
TRADING_STYLE = {
    'name': 'medium',  # Default to medium
    'intervals': ['1h', '4h', '1d'],
    'periods': ['1mo', '3mo', '6mo'],
    'window_sizes': [20, 50, 100],
    'description': 'Intra-day swing trading'
}

SUPPORTED_STYLES = ['short', 'medium', 'long']

# Short timeframe settings
SHORT_SETTINGS = {
    'name': 'short',
    'intervals': ['5m', '15m', '30m'],
    'periods': ['1d', '5d', '1mo'],
    'window_sizes': [10, 20, 50],
    'description': 'Scalping, HFT, and short-term swing trading'
}

# Medium timeframe settings
MEDIUM_SETTINGS = {
    'name': 'medium',
    'intervals': ['1h', '4h', '1d'],
    'periods': ['1mo', '3mo', '6mo'],
    'window_sizes': [20, 50, 100],
    'description': 'Intra-day swing trading'
}

# Long timeframe settings
LONG_SETTINGS = {
    'name': 'long',
    'intervals': ['1d', '1wk', '1mo'],
    'periods': ['6mo', '1y', 'max'],
    'window_sizes': [50, 100, 200],
    'description': 'Spot, buy-and-hold, intra-week/month trading'
}

def _create_magic_function(settings: Dict[str, Any]) -> Callable:
    """
    Create a function for a specific trading style.
    
    Args:
        settings: Dictionary containing the trading style settings
        
    Returns:
        Function that updates the trading style
    """
    @wraps(_create_magic_function)
    def style_function(line: str = "") -> None:
        """
        Function to set trading style.
        
        Args:
            line: Command line passed to the function
        """
        global TRADING_STYLE
        TRADING_STYLE.update(settings)
        
        # Process any additional arguments
        args = line.strip().split()
        if args and args[0] == '--verbose':
            _print_trading_style_info(TRADING_STYLE)
        
        # Return a confirmation message
        print(f"Trading style set to '{settings['name']}': {settings['description']}")
        
        # Return examples of configured timeframes
        print(f"Intervals: {', '.join(settings['intervals'])}")
        print(f"Analysis periods: {', '.join(str(p) for p in settings['periods'])}")
        print(f"Window sizes for indicators: {', '.join(str(w) for w in settings['window_sizes'])}")
    
    return style_function

def _print_trading_style_info(style: Dict[str, Any]) -> None:
    """
    Print detailed information about the current trading style.
    
    Args:
        style: Dictionary containing the trading style settings
    """
    print(f"\n{'=' * 40}")
    print(f"Trading Style: {style['name'].upper()}")
    print(f"{'=' * 40}")
    print(f"Description: {style['description']}")
    print(f"\nTimeframes:")
    for interval in style['intervals']:
        print(f"  - {interval}")
    
    print(f"\nAnalysis Periods:")
    for period in style['periods']:
        print(f"  - {period}")
    
    print(f"\nMoving Average Windows:")
    for window in style['window_sizes']:
        print(f"  - {window} bars")
    
    print(f"\nRecommended Use Cases:")
    if style['name'] == 'short':
        print("  - Day trading and scalping")
        print("  - High-frequency trading strategies")
        print("  - Quick momentum plays")
    elif style['name'] == 'medium':
        print("  - Swing trading over several days")
        print("  - Trend following on daily charts")
        print("  - Pattern recognition strategies")
    else:  # long
        print("  - Position trading")
        print("  - Long-term investing")
        print("  - Macro trend analysis")
    
    print(f"{'=' * 40}\n")

# Create functions for each trading style
short_style = _create_magic_function(SHORT_SETTINGS)
medium_style = _create_magic_function(MEDIUM_SETTINGS)
long_style = _create_magic_function(LONG_SETTINGS)

def get_current_trading_style() -> Dict[str, Any]:
    """Return the currently active trading style settings."""
    return TRADING_STYLE

def apply_trading_style(style_name: str, line: str = "") -> None:
    """
    Set the trading style by name.
    
    Args:
        style_name: Name of the style ('short', 'medium', 'long').
        line: Optional command line arguments for the style function.
    """
    if style_name == 'short':
        short_style(line)
    elif style_name == 'medium':
        medium_style(line)
    elif style_name == 'long':
        long_style(line)
    else:
        logger.error(f"Unsupported trading style: {style_name}. Supported: {SUPPORTED_STYLES}")
        # Optionally raise an error or print a message
        print(f"Error: Unsupported trading style '{style_name}'. Choose from {SUPPORTED_STYLES}.")

# Utility functions that use the current trading style

def get_default_interval() -> str:
    """
    Get the default interval based on the current trading style.
    
    Returns:
        Default interval string
    """
    return TRADING_STYLE['intervals'][1]  # Middle option as default

def get_default_period() -> str:
    """
    Get the default period based on the current trading style.
    
    Returns:
        Default period string
    """
    return TRADING_STYLE['periods'][1]  # Middle option as default

def get_default_window_size() -> int:
    """
    Get the default window size for technical indicators.
    
    Returns:
        Default window size
    """
    return TRADING_STYLE['window_sizes'][1]  # Middle option as default

def fetch_data_for_current_style(
    symbol: str, 
    interval: Optional[str] = None, 
    period: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch data for a symbol using the current trading style settings.
    
    Args:
        symbol: Stock ticker symbol
        interval: Optional interval override
        period: Optional period override
        
    Returns:
        DataFrame containing the stock data
    """
    # Use provided values or defaults from current trading style
    interval = interval or get_default_interval()
    period = period or get_default_period()
    
    # Fetch the data
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    
    if data.empty:
        raise ValueError(f"No data found for {symbol} with interval={interval}, period={period}")
    
    # Add symbol column
    data['symbol'] = symbol
    
    return data

def apply_analysis_for_current_style(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply technical analysis based on the current trading style.
    
    Args:
        data: DataFrame containing price data
        
    Returns:
        DataFrame with additional technical analysis columns
    """
    from src.analysis.market_data import add_technical_indicators
    
    # Get default window size for this trading style
    window_size = get_default_window_size()
    
    # Apply technical indicators
    data = add_technical_indicators(data)
    
    # Calculate additional moving averages specific to this trading style
    for window in TRADING_STYLE['window_sizes']:
        data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
    
    return data

def plot_for_current_style(
    data: pd.DataFrame, 
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a plot suitable for the current trading style.
    
    Args:
        data: DataFrame containing price data
        title: Optional title for the plot
        
    Returns:
        Plotly figure
    """
    if data is None or data.empty:
        logger.warning(f"No data available for {data['symbol'].iloc[0]} to plot analysis.")
        return None
    fig = plot_technical_analysis(data, title=title)
    return fig

def analyze_pair_trading_strategy(symbol1_data: pd.DataFrame, 
    symbol2_data: pd.DataFrame, 
    spread: float, 
    threshold: float, 
    timeframe: str = '1d'
) -> Optional[str]:
    # Implementation of analyze_pair_trading_strategy function
    # This function should return a string describing the result of the analysis
    # or None if no conclusion can be drawn
    pass
