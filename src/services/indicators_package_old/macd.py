"""
Moving Average Convergence Divergence (MACD) Indicator Module

This module provides functions for calculating the MACD technical indicator
using the pandas-ta library.
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


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Moving Average Convergence Divergence (MACD) for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        fast: Number of periods for fast EMA
        slow: Number of periods for slow EMA
        signal: Number of periods for signal line
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with MACD values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    logger.info(f"Calculating MACD for {symbol}/{timeframe} with fast={fast}, slow={slow}, signal={signal}")
    try:
        # Create cache key
        params = {"fast": fast, "slow": slow, "signal": signal, "column": column}
        cache_key = generate_indicator_cache_key("macd", params, symbol, timeframe)
        logger.debug(f"Generated cache key: {cache_key}")
        
        # Try to get from cache if use_cache is True
        if use_cache:
            logger.debug(f"Attempting to retrieve from cache with key: {cache_key}")
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit! Using cached MACD for {symbol}/{timeframe}")
                return cached_result
            logger.debug("Cache miss. Calculating MACD from scratch.")
        else:
            logger.debug("Cache usage disabled. Calculating MACD from scratch.")
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if fast <= 0 or slow <= 0 or signal <= 0:
            logger.error(f"Invalid parameters: fast={fast}, slow={slow}, signal={signal} (all must be positive)")
            raise ValueError(f"Fast, slow, and signal periods must be positive")
        
        if fast >= slow:
            logger.error(f"Invalid parameters: fast={fast}, slow={slow} (fast must be less than slow)")
            raise ValueError(f"Fast period must be less than slow period")
        
        if column not in df.columns:
            logger.error(f"Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate MACD
        logger.debug(f"Calculating MACD with fast={fast}, slow={slow}, signal={signal} on column '{column}'")
        macd_values = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
        
        # Format response
        logger.debug(f"Formatting response for MACD with {len(macd_values)} rows")
        response = format_indicator_response("macd", macd_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            logger.debug(f"Storing MACD result in cache with key: {cache_key}")
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="complex"
            )
            logger.info(f"Successfully cached MACD result for {symbol}/{timeframe}")
        
        logger.info(f"Successfully calculated MACD for {symbol}/{timeframe}")
        return response
        
    except Exception as e:
        logger.error(f"Error calculating MACD for {symbol}/{timeframe}: {str(e)}", exc_info=True)
        raise 