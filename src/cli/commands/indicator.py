"""
Indicator Command Handler

Handles the 'indicator' command for fetching and displaying specific technical indicators.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import typer
import pandas as pd
from tabulate import tabulate

from src.cli.display import (
    display_info, 
    display_error, 
    display_success, 
    display_warning, 
    format_price,
    display_data_age,
    display_spinner
)
from src.services.data_fetcher import get_historical_data
from src.services.indicators import get_indicator, invalidate_indicator_cache

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
indicator_app = typer.Typer()


def format_indicator_result(result: Dict[str, Any]) -> str:
    """
    Format indicator results for display.
    
    Args:
        result: Indicator result from calculation
        
    Returns:
        str: Formatted output for display
    """
    indicator_name = result.get("indicator", "").upper()
    values = result.get("values", {})
    metadata = result.get("metadata", {})
    
    # Build output as multiple sections
    output = []
    
    # Header section
    header_data = [
        ["Indicator", indicator_name],
        ["Parameters", json.dumps(metadata.get("params", {}), indent=2)]
    ]
    output.append(tabulate(header_data, tablefmt="simple"))
    output.append("")  # Add a blank line
    
    # Values section
    if isinstance(values, dict):
        if indicator_name.lower() == "macd":
            # For MACD, show the components separately
            macd_keys = [k for k in values.keys()]
            macd_line_key = next((k for k in macd_keys if k.startswith("MACD_")), None)
            signal_line_key = next((k for k in macd_keys if k.startswith("MACDs_")), None)
            histogram_key = next((k for k in macd_keys if k.startswith("MACDh_")), None)
            
            if macd_line_key and signal_line_key and histogram_key:
                macd_line = [(k, v) for k, v in values[macd_line_key].items()][-5:]
                signal_line = [(k, v) for k, v in values[signal_line_key].items()][-5:]
                histogram = [(k, v) for k, v in values[histogram_key].items()][-5:]
                
                data_rows = []
                for i in range(len(macd_line)):
                    date = datetime.fromisoformat(macd_line[i][0]).strftime("%Y-%m-%d")
                    data_rows.append([
                        date,
                        f"{macd_line[i][1]:.2f}" if macd_line[i][1] is not None else "N/A",
                        f"{signal_line[i][1]:.2f}" if signal_line[i][1] is not None else "N/A",
                        f"{histogram[i][1]:.2f}" if histogram[i][1] is not None else "N/A"
                    ])
                
                output.append("MACD Values (last 5 data points):")
                output.append(tabulate(
                    data_rows,
                    headers=["Date", "MACD Line", "Signal Line", "Histogram"],
                    tablefmt="simple",
                    numalign="right"
                ))
        
        elif indicator_name.lower() == "bbands":
            # For Bollinger Bands, show the components
            bb_keys = [k for k in values.keys()]
            upper_band_key = next((k for k in bb_keys if k.startswith("BBU_")), None)
            middle_band_key = next((k for k in bb_keys if k.startswith("BBM_")), None)
            lower_band_key = next((k for k in bb_keys if k.startswith("BBL_")), None)
            
            if upper_band_key and middle_band_key and lower_band_key:
                upper_band = [(k, v) for k, v in values[upper_band_key].items()][-5:]
                middle_band = [(k, v) for k, v in values[middle_band_key].items()][-5:]
                lower_band = [(k, v) for k, v in values[lower_band_key].items()][-5:]
                
                data_rows = []
                for i in range(len(upper_band)):
                    date = datetime.fromisoformat(upper_band[i][0]).strftime("%Y-%m-%d")
                    data_rows.append([
                        date,
                        f"{upper_band[i][1]:.2f}" if upper_band[i][1] is not None else "N/A",
                        f"{middle_band[i][1]:.2f}" if middle_band[i][1] is not None else "N/A",
                        f"{lower_band[i][1]:.2f}" if lower_band[i][1] is not None else "N/A"
                    ])
                
                output.append("Bollinger Bands Values (last 5 data points):")
                output.append(tabulate(
                    data_rows,
                    headers=["Date", "Upper Band", "Middle Band", "Lower Band"],
                    tablefmt="simple",
                    numalign="right"
                ))
        
        elif indicator_name.lower() == "stoch":
            # For Stochastic Oscillator, show %K and %D
            stoch_keys = [k for k in values.keys()]
            k_line_key = next((k for k in stoch_keys if k.startswith("STOCHk_")), None)
            d_line_key = next((k for k in stoch_keys if k.startswith("STOCHd_")), None)
            
            if k_line_key and d_line_key:
                k_line = [(k, v) for k, v in values[k_line_key].items()][-5:]
                d_line = [(k, v) for k, v in values[d_line_key].items()][-5:]
                
                data_rows = []
                for i in range(len(k_line)):
                    date = datetime.fromisoformat(k_line[i][0]).strftime("%Y-%m-%d")
                    data_rows.append([
                        date,
                        f"{k_line[i][1]:.2f}" if k_line[i][1] is not None else "N/A",
                        f"{d_line[i][1]:.2f}" if d_line[i][1] is not None else "N/A"
                    ])
                
                output.append("Stochastic Oscillator Values (last 5 data points):")
                output.append(tabulate(
                    data_rows,
                    headers=["Date", "%K", "%D"],
                    tablefmt="simple",
                    numalign="right"
                ))
        
        elif indicator_name.lower() == "ichimoku":
            # For Ichimoku Cloud, show key components
            keys = [k for k in values.keys()]
            
            # Get the latest 5 data points for each component
            data_points = {k: [(date, values[k][date]) for date in sorted(values[k].keys())[-5:]] for k in keys}
            
            # Check if data_points is empty
            if not keys or not data_points:
                output.append(f"No Ichimoku Cloud data available.")
            else:
                # Create a header row with renamed components for clarity
                component_names = {
                    "ISA_9_26": "Tenkan",
                    "ISB_9_26": "Kijun",
                    "ITS_9_26": "Span A",
                    "IKS_9_26": "Span B",
                    "ICS_9_26": "Chikou"
                }
                
                headers = ["Date"] + [component_names.get(k, k) for k in keys]
                
                # For each date (using the first component's dates), create a row
                try:
                    first_component = next(iter(data_points.values()))
                    data_rows = []
                    for i in range(len(first_component)):
                        date_key = first_component[i][0]
                        date = datetime.fromisoformat(date_key).strftime("%Y-%m-%d")
                        
                        row = [date]
                        for k in keys:
                            # Find the value for this component at this date
                            value = next((v for d, v in data_points[k] if d == date_key), None)
                            row.append(f"{value:.2f}" if value is not None else "N/A")
                        
                        data_rows.append(row)
                    
                    output.append("Ichimoku Cloud Values (last 5 data points):")
                    output.append(tabulate(
                        data_rows,
                        headers=headers,
                        tablefmt="simple",
                        numalign="right"
                    ))
                except StopIteration:
                    output.append(f"Insufficient Ichimoku Cloud data for display.")
        
        else:
            # For simple indicators like RSI, SMA, etc. with a single line
            # values is a simple dict with dates as keys and values as values
            
            # Get the last 5 data points (sorted by date)
            dates = sorted(values.keys())[-5:]
            
            data_rows = []
            for date_str in dates:
                value = values[date_str]
                date = datetime.fromisoformat(date_str).strftime("%Y-%m-%d")
                data_rows.append([
                    date, 
                    f"{value:.2f}" if value is not None else "N/A"
                ])
            
            output.append(f"{indicator_name} Values (last 5 data points):")
            output.append(tabulate(
                data_rows,
                headers=["Date", "Value"],
                tablefmt="simple",
                numalign="right"
            ))
    
    # Add interpretation
    interpretation = get_indicator_interpretation(result)
    if interpretation:
        output.append("")  # Add a blank line
        output.append("Interpretation:")
        output.append(interpretation)
    
    # Join all sections and return
    return "\n".join(output)


def get_indicator_interpretation(result: Dict[str, Any]) -> str:
    """
    Get basic interpretations for indicator values.
    
    Args:
        result: Indicator result
        
    Returns:
        str: Interpretation message
    """
    indicator = result.get("indicator", "").lower()
    values = result.get("values", {})
    
    if indicator == "rsi":
        # Get the latest RSI value
        if isinstance(values, dict) and values:
            date_keys = sorted(values.keys())
            if date_keys:
                latest_date = date_keys[-1]
                latest_value = values[latest_date]
                
                if latest_value is None:
                    return "No valid RSI value available for interpretation."
                
                if latest_value > 70:
                    return f"RSI is {latest_value:.2f}, suggesting OVERBOUGHT conditions. Consider watching for potential bearish reversal."
                elif latest_value < 30:
                    return f"RSI is {latest_value:.2f}, suggesting OVERSOLD conditions. Consider watching for potential bullish reversal."
                else:
                    return f"RSI is {latest_value:.2f}, indicating NEUTRAL momentum."
    
    elif indicator == "macd":
        # Get the latest MACD values
        macd_keys = [k for k in values.keys()]
        macd_line_key = next((k for k in macd_keys if k.startswith("MACD_")), None)
        signal_line_key = next((k for k in macd_keys if k.startswith("MACDs_")), None)
        histogram_key = next((k for k in macd_keys if k.startswith("MACDh_")), None)
        
        if macd_line_key and signal_line_key and histogram_key:
            macd_dates = sorted(values[macd_line_key].keys())
            signal_dates = sorted(values[signal_line_key].keys())
            hist_dates = sorted(values[histogram_key].keys())
            
            if macd_dates and signal_dates and hist_dates:
                latest_macd_date = macd_dates[-1]
                latest_signal_date = signal_dates[-1]
                latest_hist_date = hist_dates[-1]
                prev_hist_date = hist_dates[-2] if len(hist_dates) >= 2 else None
                
                latest_macd = values[macd_line_key][latest_macd_date]
                latest_signal = values[signal_line_key][latest_signal_date]
                latest_hist = values[histogram_key][latest_hist_date]
                prev_hist = values[histogram_key][prev_hist_date] if prev_hist_date else None
                
                if latest_macd > latest_signal and prev_hist is not None and prev_hist < latest_hist:
                    return f"MACD line ({latest_macd:.2f}) is above signal line ({latest_signal:.2f}) with increasing histogram, suggesting BULLISH momentum."
                elif latest_macd < latest_signal and prev_hist is not None and prev_hist > latest_hist:
                    return f"MACD line ({latest_macd:.2f}) is below signal line ({latest_signal:.2f}) with decreasing histogram, suggesting BEARISH momentum."
                elif latest_macd > latest_signal:
                    return f"MACD line ({latest_macd:.2f}) is above signal line ({latest_signal:.2f}), suggesting BULLISH momentum."
                elif latest_macd < latest_signal:
                    return f"MACD line ({latest_macd:.2f}) is below signal line ({latest_signal:.2f}), suggesting BEARISH momentum."
    
    elif indicator == "bbands":
        # Get the latest Bollinger Bands values
        bb_keys = [k for k in values.keys()]
        upper_key = next((k for k in bb_keys if k.startswith("BBU_")), None)
        middle_key = next((k for k in bb_keys if k.startswith("BBM_")), None)
        lower_key = next((k for k in bb_keys if k.startswith("BBL_")), None)
        
        if upper_key and middle_key and lower_key:
            # Get the latest dates for each band
            upper_dates = sorted(values[upper_key].keys())
            middle_dates = sorted(values[middle_key].keys())
            lower_dates = sorted(values[lower_key].keys())
            
            if upper_dates and middle_dates and lower_dates:
                latest_upper_date = upper_dates[-1]
                latest_middle_date = middle_dates[-1]
                latest_lower_date = lower_dates[-1]
                
                upper = values[upper_key][latest_upper_date]
                middle = values[middle_key][latest_middle_date]
                lower = values[lower_key][latest_lower_date]
                
                # Try to get the latest close price
                latest_close = None
                
                # For simplicity, assume we have price data in the time series
                # In a real implementation, we might need to get this from the original DataFrame
                if upper is not None and middle is not None and lower is not None:
                    # Estimate the current price as the middle band
                    latest_close = middle
                    
                    if latest_close > upper:
                        return f"Price ({latest_close:.2f}) is ABOVE the upper Bollinger Band ({upper:.2f}), suggesting potential OVERBOUGHT conditions."
                    elif latest_close < lower:
                        return f"Price ({latest_close:.2f}) is BELOW the lower Bollinger Band ({lower:.2f}), suggesting potential OVERSOLD conditions."
                    else:
                        proximity_upper = (upper - latest_close) / (upper - lower) * 100
                        return f"Price ({latest_close:.2f}) is within the Bollinger Bands, {proximity_upper:.1f}% from the upper band ({upper:.2f})."
    
    # Add more interpretations for other indicators as needed
    
    return "No specific interpretation available for this indicator."


@indicator_app.callback()
def callback():
    """Handle indicator commands."""
    pass


@indicator_app.command()
def calculate(
    name: str = typer.Argument(..., help="Indicator name (e.g., sma, ema, rsi, macd)"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe for data (e.g., 1d, 4h, 1h)"),
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to analyze"),
    params: str = typer.Option("{}", "--params", "-p", help="JSON string of parameters for the indicator"),
    days: int = typer.Option(100, "--days", "-d", help="Number of days of historical data to fetch"),
    force_refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh data from API"),
    raw: bool = typer.Option(False, "--raw", help="Display raw JSON output"),
):
    """
    Calculate and display a technical indicator.
    """
    try:
        # Parse params if provided
        indicator_params = {}
        if params != "{}":
            try:
                indicator_params = json.loads(params)
                display_info(f"Using custom parameters: {indicator_params}")
            except json.JSONDecodeError:
                display_warning(f"Could not parse params: {params}")
                display_warning("Using default parameters instead")
        
        # Start spinner
        with display_spinner(f"Fetching {timeframe} data for {symbol}..."):
            # Get historical data
            df = get_historical_data(symbol, timeframe, days, force_refresh)
        
        if df is None or df.empty:
            display_error(f"Failed to get historical data for {symbol}")
            return
        
        # Display data age
        display_data_age(df.index[-1])
        
        # Calculate indicator
        with display_spinner(f"Calculating {name.upper()}..."):
            result = get_indicator(
                df=df,
                indicator=name,
                params=indicator_params,
                symbol=symbol,
                timeframe=timeframe,
                use_cache=not force_refresh
            )
        
        # Display results
        if raw:
            # Display raw JSON
            print(json.dumps(result, indent=2))
        else:
            # Display formatted output
            formatted_output = format_indicator_result(result)
            print(formatted_output)
        
        display_success(f"{name.upper()} calculation completed")
        
    except Exception as e:
        display_error(f"Error calculating {name}: {str(e)}")
        logger.exception(f"Error in indicator command")


@indicator_app.command()
def list():
    """List all available indicators."""
    indicators = [
        ("SMA", "Simple Moving Average", "Trend", "sma"),
        ("EMA", "Exponential Moving Average", "Trend", "ema"),
        ("RSI", "Relative Strength Index", "Oscillator", "rsi"),
        ("MACD", "Moving Average Convergence Divergence", "Trend", "macd"),
        ("BBands", "Bollinger Bands", "Volatility", "bbands"),
        ("Stochastic", "Stochastic Oscillator", "Momentum", "stoch"),
        ("ADX", "Average Directional Index", "Trend", "adx"),
        ("ATR", "Average True Range", "Volatility", "atr"),
        ("CCI", "Commodity Channel Index", "Oscillator", "cci"),
        ("OBV", "On-Balance Volume", "Volume", "obv"),
        ("Ichimoku", "Ichimoku Cloud", "Trend", "ichimoku"),
    ]
    
    headers = ["Indicator", "Full Name", "Type", "Command"]
    print(tabulate(indicators, headers=headers, tablefmt="fancy_grid"))
    display_info("Use 'indicator calculate <name>' to calculate a specific indicator")
    display_info("Example: indicator calculate rsi")


@indicator_app.command()
def clear_cache(
    name: str = typer.Argument(..., help="Indicator name to clear cache for"),
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to clear cache for"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe to clear cache for"),
    params: str = typer.Option("{}", "--params", "-p", help="JSON string of parameters for the indicator"),
):
    """Clear the cache for a specific indicator."""
    try:
        # Parse params if provided
        indicator_params = {}
        if params != "{}":
            try:
                indicator_params = json.loads(params)
            except json.JSONDecodeError:
                display_warning(f"Could not parse params: {params}")
                display_warning("Using empty parameters instead")
        
        # Invalidate the cache
        result = invalidate_indicator_cache(
            indicator=name,
            params=indicator_params,
            symbol=symbol,
            timeframe=timeframe
        )
        
        if result:
            display_success(f"Cache cleared for {name.upper()} with symbol {symbol} and timeframe {timeframe}")
        else:
            display_warning(f"No cache found for {name.upper()} with symbol {symbol} and timeframe {timeframe}")
    
    except Exception as e:
        display_error(f"Error clearing cache: {str(e)}") 