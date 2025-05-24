"""
On-Balance Volume (OBV) Indicator Module

This module provides functions for calculating the On-Balance Volume (OBV)
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


def calculate_obv(
    df: pd.DataFrame,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate On-Balance Volume (OBV) for the given DataFrame.
    
    Args:
        df: DataFrame with price data and volume data
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with OBV values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {}
        cache_key = generate_indicator_cache_key("obv", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached OBV result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if 'volume' not in df.columns:
            raise ValueError("DataFrame missing 'volume' column required for OBV calculation")
        
        # Calculate OBV
        obv_values = ta.obv(df['close'], df['volume'])
        
        # Format response
        response = format_indicator_response("obv", obv_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="volume"
            )
            logger.debug(f"Stored OBV result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating OBV: {str(e)}")
        raise 