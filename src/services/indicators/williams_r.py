"""
Williams %R Indicator Module

This module provides functions for calculating Williams %R
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


def calculate_williams_r(
    df: pd.DataFrame,
    length: int = 14,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Williams %R for the given DataFrame.
    
    Williams %R is a momentum indicator that shows the level of the close 
    relative to the highest high for the look-back period. It oscillates 
    from 0 to -100. Readings from 0 to -20 are considered overbought. 
    Readings from -80 to -100 are considered oversold.
    
    Args:
        df: DataFrame with OHLC price data
        length: Number of periods for Williams %R calculation (default: 14)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Williams %R values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("williams_r", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Williams %R result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate Williams %R using pandas-ta
        williams_r_values = ta.willr(df['high'], df['low'], df['close'], length=length)
        
        # Format response
        response = format_indicator_response("williams_r", williams_r_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored Williams %R result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Williams %R: {str(e)}")
        raise 