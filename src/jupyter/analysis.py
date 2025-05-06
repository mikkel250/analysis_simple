"""
Notebook-friendly Wrappers for Analysis Functions

This module provides wrapper functions that adapt the core analysis logic 
for use in Jupyter notebooks, with appropriate caching and interactive parameter handling.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from IPython.display import display, HTML

# Import the core analysis functions
from src.services.data_fetcher import get_historical_data, get_current_price
from src.cli.commands.analysis import (
    generate_analysis, 
    prepare_analysis_data
)

# Cache for data and analysis results to avoid redundant API calls
_CACHE = {
    "data": {},
    "analysis": {}
}


def get_price_data(
    symbol: str, 
    days: int = 30, 
    timeframe: str = "1d", 
    force_refresh: bool = False,
    vs_currency: str = "usd"
) -> pd.DataFrame:
    """
    Get historical price data as a DataFrame for visualization.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., "bitcoin", "ethereum")
        days: Number of days of historical data to fetch
        timeframe: Timeframe for data (e.g., "1d", "4h", "1h")
        force_refresh: Force refresh data from API
        vs_currency: Currency to calculate prices against (default: "usd")
        
    Returns:
        pandas.DataFrame with OHLCV price data
    """
    # Generate cache key
    cache_key = f"price_data_{symbol}_{timeframe}_{days}_{vs_currency}"
    
    # Check cache first if not forcing refresh
    if not force_refresh and cache_key in _CACHE["data"]:
        cached_item = _CACHE["data"][cache_key]
        # Check if cache is still fresh (less than 1 hour old for most timeframes, 15 min for shorter ones)
        max_age = timedelta(minutes=15) if timeframe in ["5m", "15m", "30m"] else timedelta(hours=1)
        if datetime.now() - cached_item["timestamp"] < max_age:
            return cached_item["df"]
    
    try:
        # Get historical data
        df = get_historical_data(
            symbol=symbol, 
            timeframe=timeframe, 
            limit=max(days * 24 // _timeframe_to_hours(timeframe), 100),  # Convert days to candles
            use_cache=not force_refresh,
            vs_currency=vs_currency
        )
        
        # Store in cache
        _CACHE["data"][cache_key] = {
            "df": df,
            "timestamp": datetime.now()
        }
        
        return df
    
    except Exception as e:
        # Provide user-friendly error message
        error_html = f"""
        <div style="background-color:#ffebee;padding:10px;border-radius:5px;margin:10px 0;">
            <h3 style="color:#c62828;margin:0;">Error Fetching Price Data</h3>
            <p>{str(e)}</p>
            <p>Try refreshing the data or selecting a different symbol/timeframe.</p>
        </div>
        """
        display(HTML(error_html))
        # Return empty data frame
        return pd.DataFrame()


def get_data(
    symbol: str, 
    timeframe: str, 
    days: int, 
    force_refresh: bool = False,
    vs_currency: str = "usd"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Get historical and current price data for analysis, with caching for interactive use.
    
    Args:
        symbol: Symbol to analyze (e.g., "BTC")
        timeframe: Timeframe for data (e.g., "1d", "4h", "1h")
        days: Number of days of historical data to fetch
        force_refresh: Force refresh data from API
        vs_currency: Currency to calculate prices against (default: "usd")
        
    Returns:
        Tuple containing historical data DataFrame and current price data dictionary
    """
    # Generate cache key
    cache_key = f"{symbol}_{timeframe}_{days}_{vs_currency}"
    
    # Check cache first if not forcing refresh
    if not force_refresh and cache_key in _CACHE["data"]:
        cached_item = _CACHE["data"][cache_key]
        # Check if cache is still fresh (less than 1 hour old for most timeframes, 15 min for shorter ones)
        max_age = timedelta(minutes=15) if timeframe in ["5m", "15m", "30m"] else timedelta(hours=1)
        if datetime.now() - cached_item["timestamp"] < max_age:
            return cached_item["df"], cached_item["current_price_data"]
    
    # Fetch new data
    try:
        # Get historical data
        df = get_historical_data(
            symbol=symbol, 
            timeframe=timeframe, 
            limit=max(days * 24 // _timeframe_to_hours(timeframe), 100),  # Convert days to candles
            use_cache=not force_refresh,
            vs_currency=vs_currency
        )
        
        # Get current price data
        current_price_data = get_current_price(
            symbol=symbol, 
            force_refresh=force_refresh,
            vs_currency=vs_currency
        )
        
        # Store in cache
        _CACHE["data"][cache_key] = {
            "df": df,
            "current_price_data": current_price_data,
            "timestamp": datetime.now()
        }
        
        return df, current_price_data
    
    except Exception as e:
        # Provide user-friendly error message
        error_html = f"""
        <div style="background-color:#ffebee;padding:10px;border-radius:5px;margin:10px 0;">
            <h3 style="color:#c62828;margin:0;">Error Fetching Data</h3>
            <p>{str(e)}</p>
            <p>Try refreshing the data or selecting a different symbol/timeframe.</p>
        </div>
        """
        display(HTML(error_html))
        # Return empty data frames
        return pd.DataFrame(), {}


def _timeframe_to_hours(timeframe: str) -> int:
    """
    Convert timeframe string to hours for calculations.
    
    Args:
        timeframe: Timeframe string (e.g., "1d", "4h", "1h", "15m")
        
    Returns:
        Number of hours in the timeframe
    """
    if timeframe.endswith('m'):
        # Convert minutes to hours (as fraction)
        return int(timeframe[:-1]) / 60
    elif timeframe.endswith('h'):
        return int(timeframe[:-1])
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 24
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 24 * 7
    else:
        # Default to 1 day if unrecognized
        return 24


def run_analysis(
    symbol: str, 
    indicator: str = "rsi",
    days: int = 30,
    timeframe: str = "1d", 
    force_refresh: bool = False, 
    forecast: bool = False,
    vs_currency: str = "usd"
) -> Dict[str, Any]:
    """
    Run technical indicator analysis and return indicator data, with caching for interactive use.
    
    Args:
        symbol: Symbol to analyze (e.g., "bitcoin", "ethereum")
        indicator: Indicator to calculate (e.g., "rsi", "macd", "bbands")
        days: Number of days of historical data to fetch
        timeframe: Timeframe for data (e.g., "1d", "4h", "1h")
        force_refresh: Force refresh data from API
        forecast: Include forecasting based on historical data
        vs_currency: Currency to calculate prices against (default: "usd")
        
    Returns:
        Dictionary containing analysis results for the specified indicator
    """
    # Generate cache key
    cache_key = f"indicator_{symbol}_{indicator}_{timeframe}_{days}_{vs_currency}"
    
    # Check cache first if not forcing refresh
    if not force_refresh and cache_key in _CACHE["analysis"]:
        cached_item = _CACHE["analysis"][cache_key]
        # Check if cache is still fresh (less than 15 minutes old)
        if datetime.now() - cached_item["timestamp"] < timedelta(minutes=15):
            return cached_item["results"]
    
    # Get fresh data
    df = get_price_data(symbol, days, timeframe, force_refresh, vs_currency)
    
    # Skip analysis if data fetch failed
    if df.empty:
        return {}
    
    try:
        # Prepare data for analysis
        analysis_data = prepare_analysis_data({
            "price_data": df.iloc[-1].to_dict() if not df.empty else {},
            "metadata": {
                "symbol": symbol,
                "timeframe": timeframe,
                "vs_currency": vs_currency
            }
        })
        
        # Run the analysis based on indicator type
        from src.services.indicators import calculate_indicator
        
        # Calculate the indicator
        indicator_data = calculate_indicator(df, indicator)
        
        # Format the result with signal interpretation
        if indicator.lower() == "rsi":
            signal = "bullish" if indicator_data.get("value", 0) < 30 else "bearish" if indicator_data.get("value", 0) > 70 else "neutral"
            indicator_data["signal"] = signal
            indicator_data["type"] = "rsi"
        elif indicator.lower() == "macd":
            histogram = indicator_data.get("histogram", 0)
            signal = "bullish" if histogram > 0 else "bearish" if histogram < 0 else "neutral"
            indicator_data["signal"] = signal
            indicator_data["type"] = "macd"
        else:
            # Default signal determination
            indicator_data["signal"] = "neutral"
            indicator_data["type"] = indicator.lower()
        
        # Store in cache
        _CACHE["analysis"][cache_key] = {
            "results": indicator_data,
            "timestamp": datetime.now()
        }
        
        return indicator_data
    
    except Exception as e:
        # Provide user-friendly error message
        error_html = f"""
        <div style="background-color:#ffebee;padding:10px;border-radius:5px;margin:10px 0;">
            <h3 style="color:#c62828;margin:0;">Error Calculating {indicator.upper()}</h3>
            <p>{str(e)}</p>
            <p>Try refreshing the data or selecting a different symbol/timeframe.</p>
        </div>
        """
        display(HTML(error_html))
        return {}


def get_analysis_data(
    analysis_results: Dict[str, Any], 
    explain: bool = False
) -> List[List[Any]]:
    """
    Get formatted analysis data ready for display in a notebook.
    
    Args:
        analysis_results: Analysis results dictionary
        explain: Whether to include educational explanations
        
    Returns:
        List of formatted data rows
    """
    if not analysis_results:
        return []
    
    return prepare_analysis_data(analysis_results, explain)


def clear_cache() -> None:
    """
    Clear the analysis and data cache.
    
    This is useful when you want to force fresh data to be fetched.
    """
    _CACHE["data"].clear()
    _CACHE["analysis"].clear()
    print("Cache cleared. Next analysis will fetch fresh data.")


def batch_analyze(
    symbols: List[str], 
    timeframe: str = "1d", 
    days: int = 100,
    vs_currency: str = "usd"
) -> Dict[str, Dict[str, Any]]:
    """
    Run analysis on multiple symbols and return results.
    
    Args:
        symbols: List of symbols to analyze
        timeframe: Timeframe for data
        days: Number of days of historical data
        vs_currency: Currency to calculate prices against
        
    Returns:
        Dictionary mapping symbols to their analysis results
    """
    results = {}
    
    for symbol in symbols:
        try:
            results[symbol] = run_analysis(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                vs_currency=vs_currency
            )
        except Exception as e:
            print(f"Error analyzing {symbol}: {str(e)}")
            results[symbol] = {"error": str(e)}
    
    return results


def get_comparison_data(
    analysis_results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a comparison DataFrame from multiple analysis results.
    
    Args:
        analysis_results: Dictionary mapping symbols to their analysis results
        
    Returns:
        DataFrame with comparative analysis data
    """
    # Extract key metrics for comparison
    data = []
    
    for symbol, results in analysis_results.items():
        if "error" in results:
            continue
            
        # Get price data
        price_data = results.get("price_data", {})
        current_price = price_data.get("current_price", 0)
        price_change_24h = price_data.get("price_change_percentage_24h", 0)
        
        # Get summary
        summary = results.get("summary", {})
        trend = summary.get("trend", {})
        signals = summary.get("signals", {})
        
        data.append({
            "Symbol": symbol,
            "Price": current_price,
            "24h Change": price_change_24h,
            "Trend": trend.get("direction", "neutral"),
            "Strength": trend.get("strength", "neutral"),
            "Short Term": signals.get("short_term", "neutral"),
            "Medium Term": signals.get("medium_term", "neutral"),
            "Long Term": signals.get("long_term", "neutral"),
            "Action": signals.get("action", "hold")
        })
    
    # Convert to DataFrame
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    return df 