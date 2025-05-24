"""
Fisher Transform Indicator Module

This module provides functions for calculating the Fisher Transform
technical indicator using the pandas-ta library.
"""

import logging
from typing import Dict, Any

import pandas as pd
import pandas_ta as ta
import numpy as np

from .utils import validate_dataframe, format_indicator_response
from ..cache_service import (
    store_indicator,
    get_cached_indicator,
    generate_indicator_cache_key
)

# Configure logging
logger = logging.getLogger(__name__)


def calculate_fisher_transform(
    df: pd.DataFrame,
    length: int = 10,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Fisher Transform for the given DataFrame.
    
    The Fisher Transform converts prices into a Gaussian normal distribution,
    making it easier to identify turning points in price movements. It
    emphasizes when prices have moved to an extreme relative to recent prices.
    
    Args:
        df: DataFrame with OHLC price data
        length: Number of periods for calculation (default: 10)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Fisher Transform values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("fisher_transform", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Fisher Transform result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        required_columns = ['high', 'low']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate Fisher Transform
        # Step 1: Calculate the midpoint (HL2)
        midpoint = (df['high'] + df['low']) / 2
        
        # Step 2: Normalize the midpoint to a range between -1 and 1
        highest_high = midpoint.rolling(window=length).max()
        lowest_low = midpoint.rolling(window=length).min()
        
        # Avoid division by zero
        range_val = highest_high - lowest_low
        range_val = range_val.replace(0, 0.001)
        
        normalized = 2 * ((midpoint - lowest_low) / range_val) - 1
        
        # Ensure values are within bounds to avoid math domain errors
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Step 3: Apply smoothing (EMA)
        smoothed = normalized.ewm(span=5).mean()
        
        # Step 4: Apply Fisher Transform
        fisher = 0.5 * np.log((1 + smoothed) / (1 - smoothed))
        
        # Step 5: Apply smoothing to Fisher Transform for signal line
        fisher_signal = fisher.shift(1)
        
        # Prepare response data
        fisher_data = {
            'FISHER': fisher,
            'FISHER_SIGNAL': fisher_signal
        }
        
        # Format response - convert dict to DataFrame first  
        if isinstance(fisher_data, dict):
            fisher_df = pd.DataFrame(fisher_data, index=df.index)
        else:
            fisher_df = fisher_data
            
        response = format_indicator_response("fisher_transform", fisher_df, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored Fisher Transform result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Fisher Transform: {str(e)}")
        raise 