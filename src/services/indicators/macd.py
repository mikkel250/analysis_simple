"""
Moving Average Convergence Divergence (MACD) Indicator Module

This module provides functions for calculating the Moving Average Convergence Divergence (MACD)
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
        fast: Fast period length
        slow: Slow period length
        signal: Signal period length
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with MACD values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"fast": fast, "slow": slow, "signal": signal, "column": column}
        cache_key = generate_indicator_cache_key("macd", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached MACD result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError(f"Period lengths must be positive. Got fast={fast}, slow={slow}, signal={signal}")
        
        if fast >= slow:
            raise ValueError(f"Fast period must be less than slow period. Got fast={fast}, slow={slow}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate MACD
        macd_result = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
        
        # Format response
        response = format_indicator_response("macd", macd_result, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="trend"
            )
            logger.debug(f"Stored MACD result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        raise 