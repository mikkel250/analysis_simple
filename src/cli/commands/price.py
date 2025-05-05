"""
Price Command Handler

Handles the 'price' command for fetching and displaying BTC-USDT price information.
"""

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
from src.services.data_fetcher import get_current_price, get_historical_data

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
price_app = typer.Typer()


def format_price_result(
    price_data: Dict[str, Any],
    df: Optional[pd.DataFrame] = None
) -> str:
    """
    Format price data for display.
    
    Args:
        price_data: Current price data
        df: Optional historical data DataFrame
        
    Returns:
        str: Formatted table for display
    """
    formatted_data = []
    
    # Current price data
    current_price = price_data.get("current_price", {}).get("usd", 0)
    market_cap = price_data.get("market_cap", {}).get("usd", 0)
    total_volume = price_data.get("total_volume", {}).get("usd", 0)
    high_24h = price_data.get("high_24h", {}).get("usd", 0)
    low_24h = price_data.get("low_24h", {}).get("usd", 0)
    price_change_24h = price_data.get("price_change_24h", 0)
    price_change_percentage_24h = price_data.get("price_change_percentage_24h", 0)
    
    # Basic price info
    formatted_data.append(["BTC/USDT Price", format_price(current_price)])
    formatted_data.append(["24h Change", format_price(price_change_24h) + f" ({price_change_percentage_24h:.2f}%)"])
    formatted_data.append(["24h High", format_price(high_24h)])
    formatted_data.append(["24h Low", format_price(low_24h)])
    formatted_data.append(["24h Volume", f"${total_volume:,.0f}"])
    formatted_data.append(["Market Cap", f"${market_cap:,.0f}"])
    
    # Calculate additional metrics if historical data is available
    if df is not None and not df.empty:
        # Add a separator
        formatted_data.append(["", ""])
        formatted_data.append(["Historical Analysis", ""])
        
        # Calculate metrics
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else None
        
        # 24h price change calculated from DataFrame
        close_price = last_row['close']
        prev_close = prev_row['close'] if prev_row is not None else None
        
        if prev_close is not None:
            change = close_price - prev_close
            change_pct = (change / prev_close) * 100
            formatted_data.append(["Last Close", format_price(close_price)])
            formatted_data.append(["Previous Close", format_price(prev_close)])
            formatted_data.append(["Change", format_price(change) + f" ({change_pct:.2f}%)"])
        
        # Volatility (based on high/low range)
        volatility = (last_row['high'] - last_row['low']) / last_row['low'] * 100
        formatted_data.append(["Daily Volatility", f"{volatility:.2f}%"])
        
        # Daily trading volume
        formatted_data.append(["Daily Volume", f"{last_row['volume']:,.0f}"])
        
        # Simple moving averages
        sma_20 = df['close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else None
        sma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
        sma_200 = df['close'].rolling(window=200).mean().iloc[-1] if len(df) >= 200 else None
        
        if sma_20 is not None:
            formatted_data.append(["SMA (20)", format_price(sma_20)])
        if sma_50 is not None:
            formatted_data.append(["SMA (50)", format_price(sma_50)])
        if sma_200 is not None:
            formatted_data.append(["SMA (200)", format_price(sma_200)])
        
        # Add price analysis
        formatted_data.append(["", ""])
        formatted_data.append(["Price Analysis", get_price_analysis(df, sma_20, sma_50, sma_200)])
    
    return tabulate(formatted_data, tablefmt="fancy_grid")


def get_price_analysis(
    df: pd.DataFrame,
    sma_20: Optional[float] = None, 
    sma_50: Optional[float] = None, 
    sma_200: Optional[float] = None
) -> str:
    """
    Get basic price analysis based on the historical data.
    
    Args:
        df: Historical price data
        sma_20: 20-day simple moving average
        sma_50: 50-day simple moving average
        sma_200: 200-day simple moving average
        
    Returns:
        str: Price analysis text
    """
    analysis = []
    
    if df.empty or len(df) < 2:
        return "Insufficient data for analysis"
    
    # Get the latest prices
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    
    # Calculate price change
    price_change = (current_price - prev_price) / prev_price * 100
    
    # Analyze price change
    if price_change > 5:
        analysis.append("Strong bullish price action (>5% gain).")
    elif price_change > 2:
        analysis.append("Moderate bullish price action (2-5% gain).")
    elif price_change > 0:
        analysis.append("Slight bullish price action (<2% gain).")
    elif price_change > -2:
        analysis.append("Slight bearish price action (<2% loss).")
    elif price_change > -5:
        analysis.append("Moderate bearish price action (2-5% loss).")
    else:
        analysis.append("Strong bearish price action (>5% loss).")
    
    # Analyze moving averages if available
    if sma_20 and sma_50 and sma_200:
        # Check for golden cross / death cross
        if sma_20 > sma_50 > sma_200:
            analysis.append("Bullish trend: SMA20 > SMA50 > SMA200 (strong uptrend).")
        elif sma_20 < sma_50 < sma_200:
            analysis.append("Bearish trend: SMA20 < SMA50 < SMA200 (strong downtrend).")
        elif sma_20 > sma_50 and sma_50 < sma_200:
            analysis.append("Potential trend reversal: SMA20 above SMA50 but below SMA200.")
        elif sma_20 < sma_50 and sma_50 > sma_200:
            analysis.append("Potential trend reversal: SMA20 below SMA50 but above SMA200.")
        
        # Price position relative to moving averages
        if current_price > sma_20:
            analysis.append("Price is above SMA20, suggesting short-term bullish momentum.")
        else:
            analysis.append("Price is below SMA20, suggesting short-term bearish pressure.")
            
        if current_price > sma_200:
            analysis.append("Price is above SMA200, indicating a long-term bullish market.")
        else:
            analysis.append("Price is below SMA200, indicating a long-term bearish market.")
    
    # Analyze volatility
    volatility = (df['high'].iloc[-1] - df['low'].iloc[-1]) / df['low'].iloc[-1] * 100
    if volatility > 8:
        analysis.append("Extremely high daily volatility (>8%), indicating potential market uncertainty.")
    elif volatility > 5:
        analysis.append("High daily volatility (5-8%), suggesting active market conditions.")
    elif volatility > 2:
        analysis.append("Moderate daily volatility (2-5%), typical of normal market conditions.")
    else:
        analysis.append("Low daily volatility (<2%), suggesting a consolidation phase.")
    
    return " ".join(analysis)


@price_app.callback()
def callback():
    """Handle price commands."""
    pass


@price_app.command()
def get(
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to check price for"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe for historical data (e.g., 1d, 4h, 1h)"),
    days: int = typer.Option(30, "--days", "-d", help="Number of days of historical data to fetch"),
    force_refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh data from API"),
):
    """
    Get current price and basic analysis for BTC-USDT.
    """
    try:
        # Get current price
        with display_spinner("Fetching current price..."):
            price_data = get_current_price(symbol, force_refresh)
        
        if not price_data:
            display_error(f"Failed to get current price for {symbol}")
            return
        
        # Get historical data for additional analysis
        with display_spinner(f"Fetching {timeframe} historical data..."):
            df = get_historical_data(symbol, timeframe, days, force_refresh)
        
        if df is None or df.empty:
            display_warning(f"Could not fetch historical data. Showing basic price information only.")
            df = None
        else:
            display_data_age(df.index[-1])
        
        # Format and display the price data
        formatted_output = format_price_result(price_data, df)
        print(formatted_output)
        
        display_success("Price information retrieved successfully")
        
    except Exception as e:
        display_error(f"Error fetching price information: {str(e)}")
        logger.exception("Error in price command") 