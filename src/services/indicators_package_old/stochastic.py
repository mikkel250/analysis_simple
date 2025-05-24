"""
Stochastic Oscillator Indicator Module

This module provides functions for calculating the Stochastic Oscillator
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


def calculate_stochastic(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Stochastic Oscillator for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        k: %K period
        d: %D period
        smooth_k: %K smoothing period
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Stochastic Oscillator values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"k": k, "d": d, "smooth_k": smooth_k}
        cache_key = generate_indicator_cache_key("stoch", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Stochastic result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if k <= 0 or d <= 0 or smooth_k <= 0:
            raise ValueError(f"Periods must be positive, got k={k}, d={d}, smooth_k={smooth_k}")
        
        # Calculate Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d, smooth_k=smooth_k)
        
        # Format response
        response = format_indicator_response("stoch", stoch, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored Stochastic result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
        raise 