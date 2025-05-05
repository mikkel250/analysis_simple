"""
Commodity Channel Index (CCI) Indicator Module

This module provides functions for calculating the Commodity Channel Index (CCI)
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


def calculate_cci(
    df: pd.DataFrame,
    length: int = 20,
    constant: float = 0.015,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Commodity Channel Index (CCI) for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for CCI calculation
        constant: Scaling constant (typically 0.015)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with CCI values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "constant": constant}
        cache_key = generate_indicator_cache_key("cci", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached CCI result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if constant <= 0:
            raise ValueError(f"Constant must be positive, got {constant}")
        
        # Calculate CCI
        cci_values = ta.cci(df['high'], df['low'], df['close'], length=length, c=constant)
        
        # Format response
        response = format_indicator_response("cci", cci_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="oscillator"
            )
            logger.debug(f"Stored CCI result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating CCI: {str(e)}")
        raise 