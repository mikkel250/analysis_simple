"""
Technical Indicators Package

This package provides modules for calculating various technical indicators
using the pandas-ta library. Each indicator is implemented in its own module
to promote code modularity and maintainability.
"""

# Import utility functions
from .utils import validate_dataframe, format_indicator_response

# Import individual indicators
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

# Import the get_indicator function
from .get_indicator import get_indicator, invalidate_indicator_cache

# Define __all__ to control what gets imported with "from indicators import *"
__all__ = [
    'validate_dataframe',
    'format_indicator_response',
    'calculate_sma',
    'calculate_ema',
    'calculate_rsi',
    'calculate_macd',
    'calculate_bbands',
    'calculate_stochastic',
    'calculate_adx',
    'calculate_atr',
    'calculate_cci',
    'calculate_obv',
    'calculate_ichimoku',
    'get_indicator',
    'invalidate_indicator_cache'
]

import numpy as np
import pandas as pd

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