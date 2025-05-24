"""
Aroon Indicator Module

This module provides functions for calculating the Aroon
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


def calculate_aroon(
    df: pd.DataFrame,
    length: int = 25,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Aroon indicator for the given DataFrame.
    
    The Aroon indicator identifies when trends are likely to change direction.
    Aroon Up measures how long it has been since the highest high.
    Aroon Down measures how long it has been since the lowest low.
    Both oscillate between 0 and 100.
    
    Args:
        df: DataFrame with OHLC price data
        length: Number of periods for Aroon calculation (default: 25)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Aroon Up and Aroon Down values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("aroon", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Aroon result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        required_columns = ['high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate Aroon using pandas-ta
        aroon_result = ta.aroon(df['high'], df['low'], length=length)
        
        # aroon_result is a DataFrame with AROONU_X and AROOND_X columns
        if aroon_result is not None and not aroon_result.empty:
            # Get the column names (they include the length parameter)
            aroon_up_col = f'AROONU_{length}'
            aroon_down_col = f'AROOND_{length}'
            
            if aroon_up_col in aroon_result.columns and aroon_down_col in aroon_result.columns:
                aroon_data = {
                    'AROON_UP': aroon_result[aroon_up_col],
                    'AROON_DOWN': aroon_result[aroon_down_col],
                    'AROON_OSC': aroon_result[aroon_up_col] - aroon_result[aroon_down_col]
                }
            else:
                # Fallback: use the available columns
                aroon_data = aroon_result
        else:
            # Manual calculation if pandas-ta fails
            aroon_up = []
            aroon_down = []
            
            for i in range(len(df)):
                if i < length - 1:
                    aroon_up.append(None)
                    aroon_down.append(None)
                else:
                    # Get the window data
                    high_window = df['high'].iloc[i-length+1:i+1]
                    low_window = df['low'].iloc[i-length+1:i+1]
                    
                    # Find periods since highest high and lowest low
                    high_max_pos = high_window.values.argmax()
                    low_min_pos = low_window.values.argmin()
                    
                    periods_since_high = length - 1 - high_max_pos
                    periods_since_low = length - 1 - low_min_pos
                    
                    aroon_up_val = ((length - periods_since_high) / length) * 100
                    aroon_down_val = ((length - periods_since_low) / length) * 100
                    
                    aroon_up.append(aroon_up_val)
                    aroon_down.append(aroon_down_val)
            
            aroon_data = {
                'AROON_UP': pd.Series(aroon_up, index=df.index),
                'AROON_DOWN': pd.Series(aroon_down, index=df.index),
                'AROON_OSC': pd.Series([
                    up - down if up is not None and down is not None else None 
                    for up, down in zip(aroon_up, aroon_down)
                ], index=df.index)
            }
        
        # Format response - convert dict to DataFrame first
        if isinstance(aroon_data, dict):
            aroon_df = pd.DataFrame(aroon_data, index=df.index)
        else:
            aroon_df = aroon_data
            
        response = format_indicator_response("aroon", aroon_df, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="trend"
            )
            logger.debug(f"Stored Aroon result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Aroon: {str(e)}")
        raise 