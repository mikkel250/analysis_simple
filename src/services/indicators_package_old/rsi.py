"""
Relative Strength Index (RSI) Indicator Module

This module provides functions for calculating the Relative Strength Index (RSI)
technical indicator using the pandas-ta library.
"""

from typing import Dict, Any

import pandas as pd
import pandas_ta as ta

from .utils import validate_dataframe, format_indicator_response
from ..cache_service import (
    store_indicator,
    get_cached_indicator,
    generate_indicator_cache_key
)

# Import the centralized logging configuration
from src.config.logging_config import get_logger

# Configure logger for this module
logger = get_logger(__name__)


def calculate_rsi(
    df: pd.DataFrame,
    length: int = 14,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI) for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for RSI calculation
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with RSI values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    logger.info(f"Calculating RSI for {symbol}/{timeframe} with length={length}")
    try:
        # Create cache key
        params = {"length": length, "column": column}
        cache_key = generate_indicator_cache_key("rsi", params, symbol, timeframe)
        logger.debug(f"Generated cache key: {cache_key}")
        
        # Try to get from cache if use_cache is True
        if use_cache:
            logger.debug(f"Attempting to retrieve from cache with key: {cache_key}")
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit! Using cached RSI for {symbol}/{timeframe}")
                return cached_result
            logger.debug("Cache miss. Calculating RSI from scratch.")
        else:
            logger.debug("Cache usage disabled. Calculating RSI from scratch.")
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            logger.error(f"Invalid length parameter: {length} (must be positive)")
            raise ValueError(f"Length must be positive, got {length}")
        
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate RSI
        logger.debug(f"Calculating RSI with length={length} on column '{column}'")
        rsi_values = ta.rsi(df[column], length=length)
        
        # Format response
        logger.debug(f"Formatting response for {len(rsi_values)} RSI values")
        response = format_indicator_response("rsi", rsi_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            logger.debug(f"Storing RSI result in cache with key: {cache_key}")
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="oscillator"
            )
            logger.info(f"Successfully cached RSI result for {symbol}/{timeframe}")
        
        logger.info(f"Successfully calculated RSI for {symbol}/{timeframe}")
        return response
        
    except Exception as e:
        logger.error(f"Error calculating RSI for {symbol}/{timeframe}: {str(e)}", exc_info=True)
        raise 