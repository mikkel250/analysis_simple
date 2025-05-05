"""
Indicator Selection Module

This module provides utility functions for selecting and invalidating indicators.
"""

import logging
from typing import Dict, Any, Optional

import pandas as pd

from .sma import calculate_sma
from .ema import calculate_ema
from .rsi import calculate_rsi
from .macd import calculate_macd
from .bbands import calculate_bbands
from .stochastic import calculate_stochastic
from .adx import calculate_adx
from .atr import calculate_atr
from .cci import calculate_cci
from .obv import calculate_obv
from .ichimoku import calculate_ichimoku

from ..cache_service import invalidate_cache, generate_indicator_cache_key

# Configure logging
logger = logging.getLogger(__name__)


def get_indicator(
    df: pd.DataFrame,
    indicator: str,
    params: Dict[str, Any] = None,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate the specified indicator by delegating to the appropriate function.
    
    Args:
        df: DataFrame with price data
        indicator: Name of the indicator to calculate
        params: Parameters for the indicator calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with indicator values
        
    Raises:
        ValueError: If the indicator is not supported
    """
    if params is None:
        params = {}
    
    # Map indicator names to their calculation functions
    indicator_map = {
        'sma': calculate_sma,
        'ema': calculate_ema,
        'rsi': calculate_rsi,
        'macd': calculate_macd,
        'bbands': calculate_bbands,
        'stoch': calculate_stochastic,
        'adx': calculate_adx,
        'atr': calculate_atr,
        'cci': calculate_cci,
        'obv': calculate_obv,
        'ichimoku': calculate_ichimoku
    }
    
    # Get the appropriate function based on indicator name
    indicator_func = indicator_map.get(indicator.lower())
    
    if indicator_func is None:
        supported = ', '.join(indicator_map.keys())
        raise ValueError(f"Unsupported indicator: '{indicator}'. Supported indicators: {supported}")
    
    # Call the indicator function with the provided parameters
    try:
        return indicator_func(df=df, **params, symbol=symbol, timeframe=timeframe, use_cache=use_cache)
    except Exception as e:
        logger.error(f"Error calculating {indicator}: {str(e)}")
        raise


def invalidate_indicator_cache(
    indicator: str,
    params: Dict[str, Any] = None,
    symbol: str = 'BTC',
    timeframe: str = '1d'
) -> bool:
    """
    Invalidate the cache for a specific indicator.
    
    Args:
        indicator: Name of the indicator
        params: Parameters used for the calculation
        symbol: Symbol being analyzed
        timeframe: Timeframe of the data
        
    Returns:
        bool: True if cache was invalidated, False otherwise
    """
    if params is None:
        params = {}
    
    try:
        # Generate the cache key for the specified indicator and parameters
        cache_key = generate_indicator_cache_key(indicator, params, symbol, timeframe)
        
        # Invalidate the cache entry
        result = invalidate_cache(cache_key)
        
        if result:
            logger.info(f"Invalidated cache for {indicator} with params {params}")
        else:
            logger.info(f"No cache entry found for {indicator} with params {params}")
        
        return result
    except Exception as e:
        logger.error(f"Error invalidating cache for {indicator}: {str(e)}")
        return False 