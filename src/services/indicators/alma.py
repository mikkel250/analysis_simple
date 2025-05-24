"""
ALMA (Arnaud Legoux Moving Average) Indicator Module

This module provides functions for calculating the ALMA
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


def calculate_alma(
    df: pd.DataFrame,
    length: int = 14,
    offset: float = 0.85,
    sigma: float = 6.0,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate ALMA (Arnaud Legoux Moving Average) for the given DataFrame.
    
    ALMA uses the curve of the Normal (Gauss) distribution to allow regulating 
    the smoothness and high sensitivity of the indicator. It provides a smooth 
    moving average with less lag than traditional moving averages.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for ALMA calculation (default: 14)
        offset: Phase parameter (0-1, default: 0.85)
        sigma: Smoothness parameter (default: 6.0)
        column: Column name to use for calculation (default: 'close')
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with ALMA values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "offset": offset, "sigma": sigma, "column": column}
        cache_key = generate_indicator_cache_key("alma", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached ALMA result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if not 0 <= offset <= 1:
            raise ValueError(f"Offset must be between 0 and 1, got {offset}")
        
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate ALMA using pandas-ta
        alma_values = ta.alma(df[column], length=length, offset=offset, sigma=sigma)
        
        # Format response
        response = format_indicator_response("alma", alma_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="overlap"
            )
            logger.debug(f"Stored ALMA result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating ALMA: {str(e)}")
        raise 