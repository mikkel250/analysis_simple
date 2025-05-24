"""
Vortex Indicator Module

This module provides functions for calculating the Vortex Indicator
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


def calculate_vortex(
    df: pd.DataFrame,
    length: int = 14,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Vortex Indicator for the given DataFrame.
    
    The Vortex Indicator consists of two oscillators that capture positive 
    and negative trend movement. A bullish signal triggers when the positive 
    trend indicator crosses above the negative trend indicator or a key level.
    
    Args:
        df: DataFrame with OHLC price data
        length: Number of periods for Vortex calculation (default: 14)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Vortex Indicator values (VI+ and VI-)
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("vortex", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Vortex result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
            
        # Calculate Vortex Indicator using pandas-ta
        vortex_result = ta.vortex(df['high'], df['low'], df['close'], length=length)
        
        if vortex_result is not None and not vortex_result.empty:
            # Extract the VIP and VIM columns
            vip_col = f'VIP_{length}'
            vim_col = f'VIM_{length}'
            
            if vip_col in vortex_result.columns and vim_col in vortex_result.columns:
                vortex_data = {
                    'VORTEX_POS': vortex_result[vip_col],
                    'VORTEX_NEG': vortex_result[vim_col]
                }
            else:
                # Use whatever columns are available
                cols = list(vortex_result.columns)
                if len(cols) >= 2:
                    vortex_data = {
                        'VORTEX_POS': vortex_result[cols[0]],
                        'VORTEX_NEG': vortex_result[cols[1]]
                    }
                else:
                    vortex_data = vortex_result
        else:
            # Manual calculation as fallback
            # Calculate True Range
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Vortex Movements
            vmp = (df['high'] - df['low'].shift(1)).abs()
            vmn = (df['low'] - df['high'].shift(1)).abs()
            
            # Calculate rolling sums
            tr_sum = tr.rolling(window=length).sum()
            vmp_sum = vmp.rolling(window=length).sum()
            vmn_sum = vmn.rolling(window=length).sum()
            
            # Calculate Vortex Indicators
            vip = vmp_sum / tr_sum
            vim = vmn_sum / tr_sum
            
            vortex_data = {
                'VORTEX_POS': vip,
                'VORTEX_NEG': vim
            }
        
        # Combine both indicators into a DataFrame
        vortex_df = pd.DataFrame(vortex_data)
        
        # Format response
        response = format_indicator_response("vortex", vortex_df, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="trend"
            )
            logger.debug(f"Stored Vortex result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Vortex Indicator: {str(e)}")
        raise 