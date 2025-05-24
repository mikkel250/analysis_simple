"""
Awesome Oscillator Indicator Module

This module provides functions for calculating the Awesome Oscillator
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


def calculate_awesome_oscillator(
    df: pd.DataFrame,
    fast: int = 5,
    slow: int = 34,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Awesome Oscillator for the given DataFrame.
    
    The Awesome Oscillator (AO) is a momentum indicator that measures the 
    difference between a 5-period and 34-period simple moving average of 
    the midpoint (high + low)/2. It helps identify momentum changes and 
    potential trend reversals.
    
    Args:
        df: DataFrame with OHLC price data
        fast: Fast period for SMA calculation (default: 5)
        slow: Slow period for SMA calculation (default: 34)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Awesome Oscillator values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"fast": fast, "slow": slow}
        cache_key = generate_indicator_cache_key("awesome_oscillator", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Awesome Oscillator result for {symbol} {timeframe} with fast {fast}, slow {slow}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if fast <= 0 or slow <= 0:
            raise ValueError(f"Fast and slow periods must be positive, got fast={fast}, slow={slow}")
        
        if fast >= slow:
            raise ValueError(f"Fast period must be less than slow period, got fast={fast}, slow={slow}")
        
        required_columns = ['high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate Awesome Oscillator using pandas-ta
        ao_values = ta.ao(df['high'], df['low'], fast=fast, slow=slow)
        
        # Format response
        response = format_indicator_response("awesome_oscillator", ao_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored Awesome Oscillator result in cache for {symbol} {timeframe} with fast {fast}, slow {slow}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Awesome Oscillator: {str(e)}")
        raise 