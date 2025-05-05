"""
Technical Indicators Module

This module provides wrapper functions for calculating various technical indicators
using the pandas-ta library. It handles input validation, calculation, and formatting
the output to match the expected API response format.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np

# Fix for pandas-ta compatibility with newer numpy versions
# The module tries to import NaN from numpy but newer versions use np.nan
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Now import pandas_ta
import pandas_ta as ta

# Import caching functions
from .cache_service import (
    store_indicator, 
    get_cached_indicator,
    generate_indicator_cache_key,
    invalidate_cache
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate if the DataFrame has the required columns for technical analysis.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        bool: True if DataFrame is valid, False otherwise
        
    Raises:
        ValueError: If DataFrame is missing required columns
    """
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
    
    return True


def format_indicator_response(
    indicator_name: str,
    values: Union[pd.Series, pd.DataFrame],
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Format indicator values to match the expected API response structure.
    
    Args:
        indicator_name: Name of the indicator
        values: Calculated indicator values
        params: Parameters used for the calculation
        
    Returns:
        Dict: Formatted response with indicator values and metadata
    """
    # Convert to dictionary with timestamps as keys
    if isinstance(values, pd.Series):
        result_dict = {
            str(idx): round(val, 6) if not pd.isna(val) else None 
            for idx, val in values.items()
        }
    elif isinstance(values, pd.DataFrame):
        # For indicators that return multiple series (like MACD)
        result_dict = {}
        for col in values.columns:
            result_dict[col] = {
                str(idx): round(val, 6) if not pd.isna(val) else None 
                for idx, val in values[col].items()
            }
    else:
        result_dict = {}
    
    # Create response structure
    response = {
        "indicator": indicator_name,
        "values": result_dict,
        "metadata": {
            "params": params or {},
            "count": len(values) if hasattr(values, "__len__") else 0,
            "indicator_name": indicator_name,
            "calculation_time": pd.Timestamp.now().isoformat(),
        }
    }
    
    return response


def calculate_sma(
    df: pd.DataFrame,
    length: int = 20,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Simple Moving Average (SMA) for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for SMA calculation
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with SMA values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "column": column}
        cache_key = generate_indicator_cache_key("sma", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached SMA result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate SMA
        sma_values = ta.sma(df[column], length=length)
        
        # Format response
        response = format_indicator_response("sma", sma_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="simple"
            )
            logger.debug(f"Stored SMA result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}")
        raise


def calculate_ema(
    df: pd.DataFrame,
    length: int = 20,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Exponential Moving Average (EMA) for the given DataFrame.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for EMA calculation
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with EMA values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "column": column}
        cache_key = generate_indicator_cache_key("ema", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached EMA result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate EMA
        ema_values = ta.ema(df[column], length=length)
        
        # Format response
        response = format_indicator_response("ema", ema_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="simple"
            )
            logger.debug(f"Stored EMA result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating EMA: {str(e)}")
        raise


def calculate_rsi(
    df: pd.DataFrame,
    length: int = 14,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Relative Strength Index (RSI) for the given DataFrame.
    
    RSI is a momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, RSI is considered overbought when 
    above 70 and oversold when below 30.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for RSI calculation
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with RSI values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "column": column}
        cache_key = generate_indicator_cache_key("rsi", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached RSI result for {symbol} {timeframe} with length {length}")
                return cached_result
        
        # Validate input if not using cache or cache miss
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate RSI
        rsi_values = ta.rsi(df[column], length=length)
        
        # Format response
        response = format_indicator_response("rsi", rsi_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="simple"
            )
            logger.debug(f"Stored RSI result in cache for {symbol} {timeframe} with length {length}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}")
        raise


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
    
    MACD is calculated by subtracting the long-term EMA (26 periods) from the short-term EMA (12 periods).
    The signal line is a 9-period EMA of the MACD line.
    
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
        
        # Validate input
        validate_dataframe(df)
        
        if fast <= 0 or slow <= 0 or signal <= 0:
            raise ValueError(f"Period lengths must be positive, got fast={fast}, slow={slow}, signal={signal}")
        
        if fast >= slow:
            raise ValueError(f"Fast period must be smaller than slow period, got fast={fast}, slow={slow}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate MACD
        macd_values = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
        
        # Format response
        response = format_indicator_response("macd", macd_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="complex"
            )
            logger.debug(f"Stored MACD result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}")
        raise


def calculate_bbands(
    df: pd.DataFrame,
    length: int = 20,
    std: float = 2.0,
    column: str = 'close',
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Bollinger Bands for the given DataFrame.
    
    Bollinger Bands consist of a middle band (SMA), an upper band (SMA + std*StdDev),
    and a lower band (SMA - std*StdDev).
    
    Args:
        df: DataFrame with price data
        length: Number of periods for calculation
        std: Number of standard deviations for upper and lower bands
        column: Column name to use for calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Bollinger Bands values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length, "std": std, "column": column}
        cache_key = generate_indicator_cache_key("bbands", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Bollinger Bands result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
        
        if std <= 0:
            raise ValueError(f"Standard deviation multiplier must be positive, got {std}")
        
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        # Calculate Bollinger Bands
        bbands = ta.bbands(df[column], length=length, std=std)
        
        # Format response
        response = format_indicator_response("bbands", bbands, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="complex"
            )
            logger.debug(f"Stored Bollinger Bands result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {str(e)}")
        raise


def calculate_stochastic(
    df: pd.DataFrame,
    k: int = 14,
    d: int = 3,
    smooth_k: int = 3,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Stochastic Oscillator for the given DataFrame.
    
    The Stochastic Oscillator is a momentum indicator that shows the location of the close
    relative to high-low range over a set number of periods. It consists of two lines:
    %K (fast) and %D (slow), which oscillate between 0 and 100.
    
    Args:
        df: DataFrame with price data
        k: Number of periods for %K calculation
        d: Number of periods for %D moving average
        smooth_k: Number of periods for %K smoothing
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with Stochastic Oscillator values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"k": k, "d": d, "smooth_k": smooth_k}
        cache_key = generate_indicator_cache_key("stoch", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached Stochastic Oscillator result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input
        validate_dataframe(df)
        
        if k <= 0 or d <= 0 or smooth_k <= 0:
            raise ValueError(f"Period lengths must be positive, got k={k}, d={d}, smooth_k={smooth_k}")
            
        # Calculate Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'], k=k, d=d, smooth_k=smooth_k)
        
        # Format response
        response = format_indicator_response("stoch", stoch, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="complex"
            )
            logger.debug(f"Stored Stochastic Oscillator result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Stochastic Oscillator: {str(e)}")
        raise


def calculate_adx(
    df: pd.DataFrame,
    length: int = 14,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Average Directional Index (ADX) for the given DataFrame.
    
    ADX measures the strength of a trend, regardless of whether it's an uptrend or downtrend.
    It's derived from the DMI (Directional Movement Index) and ranges from 0 to 100.
    Values above 25 indicate strong trends.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for ADX calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with ADX values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("adx", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached ADX result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
            
        # Calculate ADX
        adx_values = ta.adx(df['high'], df['low'], df['close'], length=length)
        
        # Format response
        response = format_indicator_response("adx", adx_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="simple"
            )
            logger.debug(f"Stored ADX result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating ADX: {str(e)}")
        raise


def calculate_atr(
    df: pd.DataFrame,
    length: int = 14,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate Average True Range (ATR) for the given DataFrame.
    
    ATR is a volatility indicator that measures market volatility by decomposing the entire
    range of an asset price for the period.
    
    Args:
        df: DataFrame with price data
        length: Number of periods for ATR calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with ATR values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {"length": length}
        cache_key = generate_indicator_cache_key("atr", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached ATR result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input
        validate_dataframe(df)
        
        if length <= 0:
            raise ValueError(f"Length must be positive, got {length}")
            
        # Calculate ATR
        atr_values = ta.atr(df['high'], df['low'], df['close'], length=length)
        
        # Format response
        response = format_indicator_response("atr", atr_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="simple"
            )
            logger.debug(f"Stored ATR result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating ATR: {str(e)}")
        raise


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
    
    CCI measures the current price level relative to an average price level over a given period.
    CCI is relatively high when prices are far above their average, and relatively low when 
    prices are far below their average.
    
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
        
        # Validate input
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
                indicator_type="simple"
            )
            logger.debug(f"Stored CCI result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating CCI: {str(e)}")
        raise


def calculate_obv(
    df: pd.DataFrame,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculate On Balance Volume (OBV) for the given DataFrame.
    
    OBV measures buying and selling pressure as a cumulative indicator, adding volume
    on up days and subtracting it on down days.
    
    Args:
        df: DataFrame with price data
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with OBV values
        
    Raises:
        ValueError: If input parameters are invalid
    """
    try:
        # Create cache key
        params = {}
        cache_key = generate_indicator_cache_key("obv", params, symbol, timeframe)
        
        # Try to get from cache if use_cache is True
        if use_cache:
            cached_result = get_cached_indicator(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached OBV result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input
        validate_dataframe(df)
        
        # Check if volume is available
        if 'volume' not in df.columns:
            raise ValueError("Volume data is required for OBV calculation")
            
        # Calculate OBV
        obv_values = ta.obv(df['close'], df['volume'])
        
        # Format response
        response = format_indicator_response("obv", obv_values, params)
        
        # Store in cache if use_cache is True
        if use_cache:
            store_indicator(
                cache_key, 
                response, 
                metadata={"symbol": symbol, "timeframe": timeframe},
                timeframe=timeframe,
                indicator_type="simple"
            )
            logger.debug(f"Stored OBV result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating OBV: {str(e)}")
        raise


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
    
    Ichimoku Cloud is a collection of technical indicators that show support and resistance
    levels, momentum, and trend direction. It consists of five lines and a "cloud" area.
    
    Args:
        df: DataFrame with price data
        tenkan: Number of periods for Tenkan-sen (Conversion Line)
        kijun: Number of periods for Kijun-sen (Base Line)
        senkou: Number of periods for Senkou Span B (Leading Span B)
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
                logger.debug(f"Using cached Ichimoku Cloud result for {symbol} {timeframe}")
                return cached_result
        
        # Validate input
        validate_dataframe(df)
        
        if tenkan <= 0 or kijun <= 0 or senkou <= 0:
            raise ValueError(f"Period lengths must be positive, got tenkan={tenkan}, kijun={kijun}, senkou={senkou}")
            
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
                indicator_type="complex"
            )
            logger.debug(f"Stored Ichimoku Cloud result in cache for {symbol} {timeframe}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
        raise


def get_indicator(
    df: pd.DataFrame,
    indicator: str,
    params: Dict[str, Any] = None,
    symbol: str = 'BTC',
    timeframe: str = '1d',
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Get indicator values for the given DataFrame and parameters.
    
    This is a convenience function that routes the request to the appropriate indicator function.
    
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
    params = params or {}
    
    # Route to the appropriate indicator function
    if indicator == 'sma':
        return calculate_sma(
            df, 
            length=params.get('length', 20), 
            column=params.get('column', 'close'),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'ema':
        return calculate_ema(
            df, 
            length=params.get('length', 20), 
            column=params.get('column', 'close'),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'rsi':
        return calculate_rsi(
            df, 
            length=params.get('length', 14), 
            column=params.get('column', 'close'),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'macd':
        return calculate_macd(
            df, 
            fast=params.get('fast', 12), 
            slow=params.get('slow', 26), 
            signal=params.get('signal', 9), 
            column=params.get('column', 'close'),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'bbands':
        return calculate_bbands(
            df, 
            length=params.get('length', 20), 
            std=params.get('std', 2.0), 
            column=params.get('column', 'close'),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'stoch':
        return calculate_stochastic(
            df, 
            k=params.get('k', 14), 
            d=params.get('d', 3), 
            smooth_k=params.get('smooth_k', 3),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'adx':
        return calculate_adx(
            df, 
            length=params.get('length', 14),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'atr':
        return calculate_atr(
            df, 
            length=params.get('length', 14),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'cci':
        return calculate_cci(
            df, 
            length=params.get('length', 20), 
            constant=params.get('constant', 0.015),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'obv':
        return calculate_obv(
            df,
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    elif indicator == 'ichimoku':
        return calculate_ichimoku(
            df, 
            tenkan=params.get('tenkan', 9), 
            kijun=params.get('kijun', 26), 
            senkou=params.get('senkou', 52),
            symbol=symbol,
            timeframe=timeframe,
            use_cache=use_cache
        )
    else:
        raise ValueError(f"Unsupported indicator: {indicator}")


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
        params: Parameters used for the indicator calculation
        symbol: Symbol of the asset
        timeframe: Timeframe of the data
        
    Returns:
        bool: True if cache was invalidated, False otherwise
    """
    params = params or {}
    cache_key = generate_indicator_cache_key(indicator, params, symbol, timeframe)
    return invalidate_cache(cache_key, cache_type="indicator") 