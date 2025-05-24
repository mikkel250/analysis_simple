"""
Ultimate Oscillator Indicator Module

This module provides functions for calculating the Ultimate Oscillator
technical indicator using the pandas-ta library.
"""

import logging
from typing import Dict, Any

import pandas as pd
import pandas_ta as ta

from .utils import validate_dataframe, format_indicator_response
from ..cache_service import (
    store_indicator,
    get_cached_indicator,
    generate_indicator_cache_key
)

# Configure logging
logger = logging.getLogger(__name__)


def calculate_ultimate_oscillator(
    df: pd.DataFrame,
    fast: int = 7,
    medium: int = 14,
    slow: int = 28,
    ws: float = 4.0,
    wm: float = 2.0,
    wl: float = 1.0,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Ultimate Oscillator for the given DataFrame.
    
    The Ultimate Oscillator is a momentum oscillator designed to capture 
    momentum across three different timeframes. It uses weighted sums of 
    three oscillators, each of which is the average of true range over 
    different periods. Values range from 0 to 100, with readings above 70 
    considered overbought and below 30 oversold.
    
    Args:
        df: DataFrame with OHLC price data
        fast: Fast period (default: 7)
        medium: Medium period (default: 14)
        slow: Slow period (default: 28)
        ws: Weight for short period (default: 4.0)
        wm: Weight for medium period (default: 2.0)
        wl: Weight for long period (default: 1.0)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Ultimate Oscillator values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {
            "fast": fast, 
            "medium": medium, 
            "slow": slow,
            "ws": ws,
            "wm": wm,
            "wl": wl
        }
        cache_key = generate_indicator_cache_key("ultimate_oscillator", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Ultimate Oscillator result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if fast <= 0 or medium <= 0 or slow <= 0:
            raise ValueError(f"All periods must be positive, got fast={fast}, medium={medium}, slow={slow}")
        
        if fast >= medium or medium >= slow:
            raise ValueError(f"Periods must be in ascending order: fast < medium < slow")
        
        if ws <= 0 or wm <= 0 or wl <= 0:
            raise ValueError(f"All weights must be positive, got ws={ws}, wm={wm}, wl={wl}")
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate Ultimate Oscillator using pandas-ta
        uo_values = ta.uo(
            df['high'], 
            df['low'], 
            df['close'], 
            fast=fast, 
            medium=medium, 
            slow=slow,
            ws=ws,
            wm=wm,
            wl=wl
        )
        
        # Format response
        response = format_indicator_response("ultimate_oscillator", uo_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored Ultimate Oscillator result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Ultimate Oscillator: {str(e)}")
        raise 