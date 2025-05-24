"""
Legacy Indicators Package

This package contains implementations of various technical indicators
using pandas-ta. It provides backwards compatibility with older code.
"""

from typing import Dict, Any, List, Optional

import pandas as pd
from pandas import DataFrame

# Import the centralized logging configuration
from src.config.logging_config import get_logger

# Configure logger for this module
logger = get_logger(__name__)

# Import all indicator modules
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

# Map of indicator names to calculation functions
INDICATOR_FUNCTIONS = {
    "sma": calculate_sma,
    "ema": calculate_ema,
    "rsi": calculate_rsi,
    "macd": calculate_macd,
    "bbands": calculate_bbands,
    "stoch": calculate_stochastic,
    "adx": calculate_adx,
    "atr": calculate_atr,
    "cci": calculate_cci,
    "obv": calculate_obv,
    "ichimoku": calculate_ichimoku,
}

logger.debug(f"Legacy indicators package initialized with {len(INDICATOR_FUNCTIONS)} indicators")

def calculate_indicator(
    indicator_name: str,
    df: DataFrame,
    params: Optional[Dict[str, Any]] = None,
    symbol: str = "BTC",
    timeframe: str = "1d",
    use_cache: bool = True
) -> Dict[str, Any]:
    """
    Calculates the specified indicator using the appropriate function.
    
    Args:
        indicator_name: Name of the indicator to calculate
        df: DataFrame with price data
        params: Parameters for the indicator calculation
        symbol: Symbol being analyzed (for caching)
        timeframe: Timeframe of the data (for caching)
        use_cache: Whether to use cached results if available
        
    Returns:
        Dict: Formatted response with indicator values
        
    Raises:
        ValueError: If the indicator is not supported
    """
    logger.info(f"Calculating legacy indicator: {indicator_name} for {symbol}/{timeframe}")
    
    indicator_name = indicator_name.lower()
    
    if indicator_name not in INDICATOR_FUNCTIONS:
        logger.error(f"Unsupported indicator requested: {indicator_name}")
        supported = list(INDICATOR_FUNCTIONS.keys())
        raise ValueError(f"Unsupported indicator '{indicator_name}'. Supported indicators: {supported}")
    
    # Get the indicator function
    indicator_func = INDICATOR_FUNCTIONS[indicator_name]
    
    # Set default params if none provided
    if params is None:
        params = {}
    
    logger.debug(f"Calling {indicator_name} calculation with params: {params}")
    
    try:
        # Call the indicator function with parameters
        result = indicator_func(df=df, symbol=symbol, timeframe=timeframe, use_cache=use_cache, **params)
        logger.info(f"Successfully calculated {indicator_name} for {symbol}/{timeframe}")
        return result
    except Exception as e:
        logger.error(f"Error calculating {indicator_name} for {symbol}/{timeframe}: {str(e)}", exc_info=True)
        raise

import numpy as np

def forecast_volatility(df, horizon="24h"):
    """
    Forecast volatility for the given horizon using EWMA.
    Args:
        df (pd.DataFrame): DataFrame with 'close' prices.
        horizon (str): One of '24h', '4h', '1h'.
    Returns:
        dict: {'horizon': str, 'forecast': float, 'confidence': str}
    """
    # Validate input
    if df is None or 'close' not in df or len(df['close'].dropna()) < 20:
        return {"horizon": horizon, "forecast": None, "confidence": "low"}
    closes = df['close'].dropna().values
    if np.allclose(closes, closes[0]):
        return {"horizon": horizon, "forecast": 0.0, "confidence": "low"}
    # Map horizon to EWMA span (approximate)
    span_map = {"24h": 20, "4h": 5, "1h": 2}
    span = span_map.get(horizon, 20)
    # Calculate EWMA volatility (annualized, then scaled to percent)
    returns = pd.Series(closes).pct_change().dropna()
    ewma_vol = returns.ewm(span=span).std().iloc[-1]
    # Convert to percent (daily volatility * sqrt(365) for annualized, but here just percent)
    forecast = float(ewma_vol * 100)
    # Confidence: high if enough data and volatility > 1%, medium if 0.5-1%, low otherwise
    if len(closes) >= 50 and forecast > 1:
        confidence = "high"
    elif forecast > 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    return {"horizon": horizon, "forecast": forecast, "confidence": confidence} 