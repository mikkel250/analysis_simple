"""
Ichimoku Cloud Indicator Module

This module provides functions for calculating the Ichimoku Cloud
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


def calculate_ichimoku(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou: int = 52,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Ichimoku Cloud for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        tenkan: Tenkan-sen (Conversion Line) period
        kijun: Kijun-sen (Base Line) period
        senkou: Senkou Span B (Leading Span B) period
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Ichimoku Cloud values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"tenkan": tenkan, "kijun": kijun, "senkou": senkou}
        cache_key = generate_indicator_cache_key("ichimoku", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Ichimoku result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if tenkan <= 0 or kijun <= 0 or senkou <= 0:
            raise ValueError(f"Periods must be positive, got tenkan={tenkan}, kijun={kijun}, senkou={senkou}")
        
        if tenkan >= kijun:
            raise ValueError(f"Tenkan period should be less than Kijun period, got tenkan={tenkan}, kijun={kijun}")
        
        if kijun >= senkou:
            raise ValueError(f"Kijun period should be less than Senkou period, got kijun={kijun}, senkou={senkou}")
        
        # Calculate Ichimoku Cloud
        ichimoku = ta.ichimoku(df['high'], df['low'], df['close'], 
                              tenkan=tenkan, kijun=kijun, senkou=senkou)
        
        # Format response
        response = format_indicator_response("ichimoku", ichimoku, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="trend"
            )
            logger.debug(f"Stored Ichimoku result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
        raise 