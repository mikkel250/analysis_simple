"""
Enhanced Commodity Channel Index (CCI) Indicator Module

This module provides functions for calculating enhanced CCI variations
including multi-timeframe CCI, CCI with different smoothing methods,
and CCI divergence analysis using the pandas-ta library.
"""

import logging
from typing import Dict, Any, List, Optional

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


def calculate_cci_enhanced(
    df: pd.DataFrame,
    length: int = 20,
    constant: float = 0.015,
    smoothing: str = 'sma',
    multi_timeframe: bool = False,
    timeframes: List[int] = None,
    divergence_analysis: bool = False,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Enhanced CCI with multiple variations for the given DataFrame.
    
    Enhanced CCI includes:
    - Standard CCI with different smoothing methods (SMA, EMA, WMA)
    - Multi-timeframe CCI analysis
    - CCI divergence detection
    - Overbought/oversold level analysis
    
    Args:
        df: DataFrame with OHLC price data
        length: Number of periods for CCI calculation (default: 20)
        constant: Scaling constant (default: 0.015)
        smoothing: Smoothing method ('sma', 'ema', 'wma') (default: 'sma')
        multi_timeframe: Whether to calculate multiple timeframes (default: False)
        timeframes: List of timeframes for multi-timeframe analysis (default: [14, 20, 50])
        divergence_analysis: Whether to perform divergence analysis (default: False)
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Enhanced CCI values and analysis
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Set default timeframes if not provided
        if timeframes is None:
            timeframes = [14, 20, 50]
        
        # Create cache key
        params = {
            "length": length, 
            "constant": constant,
            "smoothing": smoothing,
            "multi_timeframe": multi_timeframe,
            "timeframes": timeframes,
            "divergence_analysis": divergence_analysis
        }
        cache_key = generate_indicator_cache_key("cci_enhanced", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Enhanced CCI result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if constant <= 0:
            raise ValueError(f"Constant must be positive, got {constant}")
        
        if smoothing not in ['sma', 'ema', 'wma']:
            raise ValueError(f"Smoothing must be 'sma', 'ema', or 'wma', got {smoothing}")
        
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
        
        # Calculate typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate CCI with different smoothing methods
        if smoothing == 'sma':
            tp_ma = typical_price.rolling(window=length).mean()
        elif smoothing == 'ema':
            tp_ma = typical_price.ewm(span=length).mean()
        elif smoothing == 'wma':
            weights = np.arange(1, length + 1)
            tp_ma = typical_price.rolling(window=length).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # Calculate mean deviation
        mean_deviation = typical_price.rolling(window=length).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        
        # Calculate enhanced CCI
        cci_enhanced = (typical_price - tp_ma) / (constant * mean_deviation)
        
        # Prepare response data
        response_data = {
            'cci_enhanced': cci_enhanced,
            'typical_price': typical_price,
            'tp_ma': tp_ma,
            'mean_deviation': mean_deviation
        }
        
        # Multi-timeframe analysis
        if multi_timeframe:
            mtf_data = {}
            for tf in timeframes:
                if tf != length:  # Avoid duplicate calculation
                    if smoothing == 'sma':
                        tf_ma = typical_price.rolling(window=tf).mean()
                    elif smoothing == 'ema':
                        tf_ma = typical_price.ewm(span=tf).mean()
                    elif smoothing == 'wma':
                        weights = np.arange(1, tf + 1)
                        tf_ma = typical_price.rolling(window=tf).apply(
                            lambda x: np.dot(x, weights) / weights.sum(), raw=True
                        )
                    
                    tf_mean_dev = typical_price.rolling(window=tf).apply(
                        lambda x: np.mean(np.abs(x - x.mean())), raw=True
                    )
                    
                    mtf_cci = (typical_price - tf_ma) / (constant * tf_mean_dev)
                    mtf_data[f'cci_{tf}'] = mtf_cci
            
            response_data.update(mtf_data)
        
        # Divergence analysis
        if divergence_analysis:
            # Calculate price peaks and troughs
            price_peaks = df['close'].rolling(window=5, center=True).max() == df['close']
            price_troughs = df['close'].rolling(window=5, center=True).min() == df['close']
            
            # Calculate CCI peaks and troughs
            cci_peaks = cci_enhanced.rolling(window=5, center=True).max() == cci_enhanced
            cci_troughs = cci_enhanced.rolling(window=5, center=True).min() == cci_enhanced
            
            # Identify potential divergences
            bullish_divergence = price_troughs & ~cci_troughs
            bearish_divergence = price_peaks & ~cci_peaks
            
            response_data.update({
                'price_peaks': price_peaks,
                'price_troughs': price_troughs,
                'cci_peaks': cci_peaks,
                'cci_troughs': cci_troughs,
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence
            })
        
        # Add overbought/oversold levels
        response_data.update({
            'overbought_level': pd.Series([100] * len(df), index=df.index),
            'oversold_level': pd.Series([-100] * len(df), index=df.index),
            'extreme_overbought': pd.Series([200] * len(df), index=df.index),
            'extreme_oversold': pd.Series([-200] * len(df), index=df.index)
        })
        
        # Format response - convert dict to DataFrame first
        if isinstance(response_data, dict):
            response_df = pd.DataFrame(response_data, index=df.index)
        else:
            response_df = response_data
            
        response = format_indicator_response("cci_enhanced", response_df, params)
        
        # Add interpretation
        latest_cci = cci_enhanced.iloc[-1] if not pd.isna(cci_enhanced.iloc[-1]) else None
        if latest_cci is not None:
            if latest_cci > 200:
                signal = "Extremely Overbought"
            elif latest_cci > 100:
                signal = "Overbought"
            elif latest_cci < -200:
                signal = "Extremely Oversold"
            elif latest_cci < -100:
                signal = "Oversold"
            else:
                signal = "Neutral"
            
            response['interpretation'] = {
                'signal': signal,
                'latest_value': latest_cci,
                'smoothing_method': smoothing
            }
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="oscillator"
            )
            logger.debug(f"Stored Enhanced CCI result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Enhanced CCI: {str(e)}")
        raise 