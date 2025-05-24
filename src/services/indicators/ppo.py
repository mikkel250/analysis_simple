"""
PPO (Percentage Price Oscillator) Indicator Module

This module provides functions for calculating PPO
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


def calculate_ppo(
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
    Calculate PPO (Percentage Price Oscillator) for the given DataFrame.
    
    The PPO is a momentum oscillator that measures the difference between 
    two moving averages as a percentage of the larger moving average.
    
    Args:
        df: DataFrame with price data
        fast: Fast period for EMA (default: 12)
        slow: Slow period for EMA (default: 26)
        signal: Signal line period (default: 9)
        column: Column name to use for calculation (default: 'close')
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with PPO values (PPO line, signal line, histogram)
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"fast": fast, "slow": slow, "signal": signal, "column": column}
        cache_key = generate_indicator_cache_key("ppo", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached PPO result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError("All periods must be positive")
        
        if fast >= slow:
            raise ValueError("Fast period must be less than slow period")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate PPO using pandas-ta
        ppo_result = ta.ppo(df[column], fast=fast, slow=slow, signal=signal)
        
        if ppo_result is not None and not ppo_result.empty:
            # The ppo function returns a DataFrame with multiple columns
            ppo_data = ppo_result
        else:
            # Manual calculation as fallback
            fast_ema = df[column].ewm(span=fast).mean()
            slow_ema = df[column].ewm(span=slow).mean()
            
            # Calculate PPO line
            ppo_line = ((fast_ema - slow_ema) / slow_ema) * 100
            
            # Calculate signal line
            ppo_signal = ppo_line.ewm(span=signal).mean()
            
            # Calculate histogram
            ppo_hist = ppo_line - ppo_signal
            
            ppo_data = pd.DataFrame({
                'PPO': ppo_line,
                'PPO_Signal': ppo_signal,
                'PPO_Histogram': ppo_hist
            }, index=df.index)
        
        # Format response
        response = format_indicator_response("ppo", ppo_data, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="momentum"
            )
            logger.debug(f"Stored PPO result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating PPO: {str(e)}")
        raise 