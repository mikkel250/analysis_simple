"""
TRIX Indicator Module

This module provides functions for calculating TRIX
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


def calculate_trix(
    df: pd.DataFrame,
    length: int = 15,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate TRIX for the given DataFrame.
    
    TRIX shows the percent rate of change of a triple exponentially 
    smoothed moving average. It is primarily used to identify oversold 
    and overbought markets and can also be used as a momentum indicator.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for TRIX calculation (default: 15)
        column: Column name to use for calculation (default: 'close')
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with TRIX values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "column": column}
        cache_key = generate_indicator_cache_key("trix", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached TRIX result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate TRIX using pandas-ta
        trix_values = ta.trix(df[column], window=length)
        
        # Format response
        response = format_indicator_response("trix", trix_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored TRIX result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating TRIX: {str(e)}")
        raise 