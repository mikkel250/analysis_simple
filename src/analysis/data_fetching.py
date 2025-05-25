"""
Data fetching module for market analysis.

This module provides functions to fetch various types of market data including
historical price data, open interest, and funding rates.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Default number of candles to fetch for historical data
DEFAULT_CANDLE_LIMIT = 200


def fetch_historical_data(symbol: str, timeframe: str = "1d", use_test_data: bool = False) -> pd.DataFrame:
    """
    Fetch historical price data for a given symbol and timeframe.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC-USDT')
        timeframe: Timeframe for data (e.g., '1h', '1d')
        use_test_data: Whether to use generated test data instead of real data
        
    Returns:
        DataFrame with OHLCV data
    """
    if use_test_data:
        return _generate_test_data(symbol)
    
    logger.info(
        f"Fetching data for {symbol} with timeframe={timeframe}, "
        f"limit={DEFAULT_CANDLE_LIMIT}"
    )
    
    try:
        from src.services.data_fetcher import get_historical_data
        data = get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            limit=DEFAULT_CANDLE_LIMIT,
            use_cache=True,
        )
        
        if data is not None and not data.empty and data.index.tz is not None:
            data.index = data.index.tz_convert(None)
            
        if data is None or data.empty:
            logger.warning(
                f"No data returned for {symbol} with timeframe {timeframe}"
            )
            return pd.DataFrame()
        else:
            logger.debug(
                f"Successfully fetched data for {symbol}, shape: "
                f"{data.shape}. Columns: {data.columns.tolist()}"
            )
            return data
            
    except Exception as e:
        logger.error(
            f"Error fetching data for {symbol}, timeframe {timeframe}: {e}",
            exc_info=True,
        )
        return pd.DataFrame()


def _generate_test_data(symbol: str) -> pd.DataFrame:
    """
    Generate test data for development and testing purposes.
    
    Args:
        symbol: Trading symbol (used for logging)
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="15min")
    base_price = 50000 + np.random.randint(-5000, 5000)
    trend_val = np.random.choice([-0.05, 0, 0.05])
    trend = np.linspace(0, trend_val, 100)
    noise = np.random.normal(0, 0.015, 100)
    
    close_prices: np.ndarray = base_price * (1 + trend + noise)
    open_prices: np.ndarray = close_prices * (1 + np.random.normal(0, 0.002, 100))
    high_prices: np.ndarray = np.maximum(close_prices, open_prices) * (
        1 + abs(np.random.normal(0, 0.005, 100))
    )
    low_prices: np.ndarray = np.minimum(close_prices, open_prices) * (
        1 - abs(np.random.normal(0, 0.005, 100))
    )
    volumes = np.random.randint(500, 2000, 100) * (1 + abs(noise) * 10)
    
    data = pd.DataFrame(
        {
            "date": dates,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volumes.astype(float),
        }
    )
    data.set_index("date", inplace=True)
    
    if data.index.tz is not None:
        data.index = data.index.tz_convert(None)
        
    logger.info(f"Using test data for {symbol}, shape: {data.shape}")
    return data


def fetch_open_interest_data(symbol: str, exchange: str = "okx") -> Dict[str, Any]:
    """
    Fetch open interest data for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC-USDT')
        exchange: Exchange to fetch data from
        
    Returns:
        Dictionary containing open interest data or error information
    """
    try:
        from src.services.open_interest import fetch_open_interest
        oi_data = fetch_open_interest(symbol, exchange=exchange)
        
        value = oi_data.get("open_interest_value") or oi_data.get("value")
        prev_value = None
        
        if value is not None and "open_interest_change_24h" in oi_data:
            change_pct = oi_data["open_interest_change_24h"]
            prev_value = value / (1 + change_pct / 100) if change_pct is not None else None
            
        return {
            "value": value,
            "prev_value": prev_value,
            "raw": oi_data,
        }
        
    except Exception as e:
        logger.error(f"Error fetching open interest for {symbol}: {e}", exc_info=True)
        return {"error": str(e)}


def fetch_funding_rate_data(symbol: str) -> Dict[str, Any]:
    """
    Fetch funding rate data for a given symbol.
    
    Args:
        symbol: Trading symbol (e.g., 'BTC-USDT')
        
    Returns:
        Dictionary containing funding rate data or error information
    """
    try:
        from src.services.funding_rates import fetch_okx_funding_rate
        fr_data = fetch_okx_funding_rate(symbol)
        return fr_data
        
    except Exception as e:
        logger.error(f"Error fetching funding rate for {symbol}: {e}", exc_info=True)
        return {"error": str(e)} 