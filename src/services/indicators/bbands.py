"""
Bollinger Bands (BBands) Indicator Module

This module provides functions for calculating the Bollinger Bands
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


def calculate_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for SMA calculation within BBands
        std: Standard deviation multiplier
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Bollinger Bands values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "std": std, "column": column}
        cache_key = generate_indicator_cache_key("bbands", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached BBands result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if std <= 0:
            raise ValueError(f"Standard deviation multiplier must be positive, got {std}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate Bollinger Bands
        bbands = ta.bbands(df[column], length=length, std=std)
        
        # Format response
        response = format_indicator_response("bbands", bbands, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="volatility"
            )
            logger.debug(f"Stored BBands result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        raise 