"""
KAMA (Kaufman's Adaptive Moving Average) Indicator Module

This module provides functions for calculating KAMA
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


def calculate_kama(
    df: pd.DataFrame,
    length: int = 10,
    fast: int = 2,
    slow: int = 30,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate KAMA (Kaufman's Adaptive Moving Average) for the given DataFrame.
    
    KAMA is a moving average designed to account for market noise or volatility. 
    KAMA will closely follow prices when the price swings are relatively small 
    and the noise is low. KAMA will adjust when the price swings widen and 
    follow prices from a greater distance.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for efficiency ratio (default: 10)
        fast: Number of periods for the fastest EMA constant (default: 2)
        slow: Number of periods for the slowest EMA constant (default: 30)
        column: Column name to use for calculation (default: 'close')
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with KAMA values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "fast": fast, "slow": slow, "column": column}
        cache_key = generate_indicator_cache_key("kama", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached KAMA result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if fast <= 0:
            raise ValueError(f"Fast period must be positive, got {fast}")
        
        if slow <= 0:
            raise ValueError(f"Slow period must be positive, got {slow}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate KAMA using pandas-ta
        kama_values = ta.kama(df[column], window=length, pow1=fast, pow2=slow)
        
        # Format response
        response = format_indicator_response("kama", kama_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="overlap"
            )
            logger.debug(f"Stored KAMA result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating KAMA: {str(e)}")
        raise 