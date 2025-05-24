"""
Rate of Change (ROC) Indicator Module

This module provides functions for calculating the Rate of Change (ROC)
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


def calculate_roc(
    df: pd.DataFrame,
    length: int = 12,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Rate of Change (ROC) for the given DataFrame.
    
    The Rate of Change (ROC) indicator measures the percentage change in price 
    from one period to the next. It's a pure momentum oscillator that shows 
    the velocity of price movement and can help identify overbought/oversold 
    conditions and momentum shifts.
    
    Args:
        df: DataFrame with OHLC price data
        length: Number of periods for ROC calculation (default: 12)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with ROC values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("roc", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached ROC result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        required_columns = ['close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate ROC using pandas-ta
        roc_values = ta.roc(df['close'], length=length)
        
        # Format response
        response = format_indicator_response("roc", roc_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored ROC result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating ROC: {str(e)}")
        raise 