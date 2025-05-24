"""
Trend analysis module for market data.

This module provides functions for analyzing market trends based on 
technical indicators, price action, and other market data.
"""

from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np
from src.config.logging_config import get_logger
from src.analysis.error_handling import (
    ValidationError, IndicatorError, validate_dataframe, safe_operation, safe_dataframe_operation
)

# Set up logger
logger = get_logger(__name__)

def determine_trend(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze price data to determine trend direction, strength, and key levels.
    
    Args:
        data: DataFrame with OHLCV and technical indicator data
        
    Returns:
        Dict containing trend analysis results
        
    Raises:
        ValidationError: If the data doesn't meet requirements
        IndicatorError: If there's an error analyzing trends
    """
    logger.info("Analyzing market trend")
    
    try:
        # Validate input data
        required_columns = ['close', 'high', 'low']
        validate_dataframe(data, required_columns=required_columns, min_rows=20)
        
        # Initialize results dictionary
        result = {
            "overall_trend": "neutral",
            "confidence": "medium",
            "supporting_indicators": [],
            "contradicting_indicators": [],
            "price_action": {},
            "moving_averages": {},
            "momentum": {},
        }
        
        # Extract the last few rows for trend analysis
        recent_data = data.tail(20)
        logger.debug(f"Using {len(recent_data)} recent data points for trend analysis")

        # Find the last valid row for key indicators
        required_cols = ['rsi_14', 'MACD_12_26_9', 'MACDs_12_26_9']
        try:
            valid_recent = recent_data.dropna(subset=required_cols)
            if valid_recent.empty:
                logger.debug(f"All required columns {required_cols} are present but all values are NaN. Cannot determine trend.")
                raise IndicatorError(f"Cannot determine trend: all required columns {required_cols} are NaN.")
        except KeyError as e:
            error_msg = f"Error determining trend: {list(e.args[0])}"
            logger.error(error_msg)
            raise IndicatorError(error_msg)
        
        # Check if necessary indicators exist - just log warnings but continue
        additional_indicators = ['sma_20', 'sma_50', 'rsi_14', 'MACD_12_26_9', 'MACDs_12_26_9']
        missing_indicators = [col for col in additional_indicators if col not in data.columns]
        
        if missing_indicators:
            logger.warning(f"Missing indicators for trend analysis: {missing_indicators}")
        
        # 1. Moving Average Trend Analysis
        ma_trend = safe_operation(
            lambda: _analyze_moving_averages(data, valid_recent.iloc[-1]),
            fallback={
                "price_vs_sma20": "unavailable",
                "price_vs_sma50": "unavailable",
                "sma20_vs_sma50": "unavailable",
            },
            error_msg="Error analyzing moving average trend",
            logger_instance=logger
        )
        result["moving_averages"] = ma_trend
        
        # 2. Momentum Indicators Analysis
        momentum_trend = safe_operation(
            lambda: _analyze_momentum(data, valid_recent.iloc[-1]),
            fallback={
                "rsi": None,
                "rsi_condition": "unavailable",
                "MACD_12_26_9": None,
                "MACDs_12_26_9": None,
                "MACDh_12_26_9": None,
                "macd_trend": "unavailable"
            },
            error_msg="Error analyzing momentum indicators",
            logger_instance=logger
        )
        result["momentum"] = momentum_trend
        
        # 3. Price Action Analysis
        price_action_trend = safe_operation(
            lambda: _analyze_price_action(valid_recent),
            fallback={
                "recent_performance": "0.00%",
                "volatility": "0.00%",
                "direction": "neutral"
            },
            error_msg="Error analyzing price action",
            logger_instance=logger
        )
        result["price_action"] = price_action_trend
        
        # Combine trend signals to determine overall trend
        result = _determine_overall_trend(result)
        
        logger.info(f"Trend analysis complete. Result: {result['overall_trend']} with {result['confidence']} confidence")
        return result
    except ValidationError as e:
        logger.error(f"Validation error in determine_trend: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Error determining trend: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise IndicatorError(error_msg)

def _analyze_moving_averages(data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
    """
    Analyze price in relation to moving averages.
    
    Args:
        data: DataFrame with market data
        current: Series with most recent data point
        
    Returns:
        Dict with moving average analysis
        
    Raises:
        Exception: If required data is missing
    """
    # Check if required columns exist
    required_columns = ['close', 'sma_20', 'sma_50']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        logger.warning(f"Cannot analyze moving averages: missing columns {missing}")
        raise ValidationError(f"Missing required columns for MA analysis: {missing}")
    
    logger.debug("Analyzing price relative to moving averages")
    price = current['close']
    sma20 = current['sma_20']
    sma50 = current['sma_50']
    
    result = {
        'price_vs_sma20': 'above' if price > sma20 else 'below',
        'price_vs_sma50': 'above' if price > sma50 else 'below',
        'sma20_vs_sma50': 'above' if sma20 > sma50 else 'below',
    }
    
    return result

def _analyze_momentum(data: pd.DataFrame, current: pd.Series) -> Dict[str, Any]:
    """
    Analyze momentum indicators.
    
    Args:
        data: DataFrame with market data
        current: Series with most recent data point
        
    Returns:
        Dict with momentum indicator analysis
        
    Raises:
        Exception: If required data is missing
    """
    # Check if required columns exist
    required_columns = ['rsi_14', 'MACD_12_26_9', 'MACDs_12_26_9']
    if not all(col in data.columns for col in required_columns):
        missing = [col for col in required_columns if col not in data.columns]
        logger.warning(f"Cannot analyze momentum: missing columns {missing}")
        raise ValidationError(f"Missing required columns for momentum analysis: {missing}")
    
    logger.debug("Analyzing momentum indicators")
    rsi = current['rsi_14']
    macd = current['MACD_12_26_9']
    macd_signal = current['MACDs_12_26_9']

    # Debug print for NaN investigation
    print("[DEBUG] _analyze_momentum: current index:", getattr(current, 'name', None))
    print("[DEBUG] _analyze_momentum: current values:")
    print(current[['rsi_14', 'MACD_12_26_9', 'MACDs_12_26_9']])
    print("[DEBUG] _analyze_momentum: last 10 RSI values:")
    print(data['rsi_14'].tail(10))
    print("[DEBUG] _analyze_momentum: last 10 MACD values:")
    print(data['MACD_12_26_9'].tail(10))
    print("[DEBUG] _analyze_momentum: last 10 MACD signal values:")
    print(data['MACDs_12_26_9'].tail(10))

    # Check for NaN values
    if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal):
        logger.warning("NaN values detected in momentum indicators")
        raise ValidationError("NaN values detected in momentum indicators")
    
    result = {
        'rsi': rsi,
        'rsi_condition': 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral',
        'MACD_12_26_9': macd,
        'MACDs_12_26_9': macd_signal,
        'MACDh_12_26_9': macd - macd_signal,
        'macd_trend': 'bullish' if macd > macd_signal else 'bearish'
    }
    
    return result

def _analyze_price_action(recent_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze recent price action.
    
    Args:
        recent_data: DataFrame with recent market data
        
    Returns:
        Dict with price action analysis
        
    Raises:
        Exception: If calculation fails
    """
    # Validate input
    if 'close' not in recent_data.columns:
        raise ValidationError("Missing 'close' column for price action analysis")
    
    if len(recent_data) < 2:
        raise ValidationError("Insufficient data points for price action analysis")
    
    logger.debug("Analyzing recent price action")
    recent_closes = recent_data['close'].values
    
    # Check for NaN values
    if np.isnan(recent_closes).any():
        logger.warning("NaN values detected in price data")
        # Fill NaN values with forward fill method
        recent_closes = pd.Series(recent_closes).fillna(method='ffill').values
    
    # Calculate price changes
    price_changes = np.diff(recent_closes) / recent_closes[:-1]
    
    # Calculate average daily change and volatility
    avg_change = np.mean(price_changes) * 100  # as percentage
    volatility = np.std(price_changes) * 100  # as percentage
    
    result = {
        'recent_performance': f"{avg_change:.2f}%",
        'volatility': f"{volatility:.2f}%",
        'direction': 'up' if avg_change > 0 else 'down' if avg_change < 0 else 'neutral',
    }
    
    return result

def _determine_overall_trend(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine the overall trend based on individual indicators.
    
    Args:
        result: Dict with trend analysis components
        
    Returns:
        Updated result dict with overall trend and confidence
    """
    try:
        # Initialize counters for trend signals
        bullish_signals = 0
        bearish_signals = 0
        neutral_signals = 0
        
        # Process moving averages
        ma_data = result.get('moving_averages', {})
        if ma_data.get('price_vs_sma20') == 'above' and ma_data.get('price_vs_sma50') == 'above':
            result['supporting_indicators'].append('price above both MAs, bullish alignment')
            bullish_signals += 1
        elif ma_data.get('price_vs_sma20') == 'below' and ma_data.get('price_vs_sma50') == 'below':
            result['supporting_indicators'].append('price below both MAs, bearish alignment')
            bearish_signals += 1
        elif ma_data.get('price_vs_sma20') == 'above':
            result['supporting_indicators'].append('price above short-term MA')
            bullish_signals += 0.5
        elif ma_data.get('price_vs_sma20') == 'below':
            result['contradicting_indicators'].append('price below short-term MA')
            bearish_signals += 0.5
            
        # Process momentum indicators
        momentum_data = result.get('momentum', {})
        
        # RSI analysis
        rsi = momentum_data.get('rsi')
        if rsi is not None:
            if rsi > 60:
                result['supporting_indicators'].append(f'RSI showing strength ({rsi:.2f})')
                bullish_signals += 1
            elif rsi < 40:
                result['supporting_indicators'].append(f'RSI showing weakness ({rsi:.2f})')
                bearish_signals += 1
        
        # MACD analysis
        macd_trend = momentum_data.get('macd_trend')
        if macd_trend == 'bullish':
            result['supporting_indicators'].append('MACD above signal line')
            bullish_signals += 1
        elif macd_trend == 'bearish':
            result['supporting_indicators'].append('MACD below signal line')
            bearish_signals += 1
            
        # Process price action
        price_action = result.get('price_action', {})
        direction = price_action.get('direction')
        if direction == 'up':
            bullish_signals += 0.5
        elif direction == 'down':
            bearish_signals += 0.5
            
        # Final trend determination
        if bullish_signals > bearish_signals + 0.5:
            result['overall_trend'] = 'bullish'
        elif bearish_signals > bullish_signals + 0.5:
            result['overall_trend'] = 'bearish'
        else:
            result['overall_trend'] = 'neutral'
            
        # Determine confidence level
        supporting_count = len(result['supporting_indicators'])
        contradicting_count = len(result['contradicting_indicators'])
        
        # Adjust confidence based on agreement among indicators
        if supporting_count > 3 and contradicting_count == 0:
            result['confidence'] = 'high'
        elif contradicting_count > supporting_count:
            result['confidence'] = 'low'
        else:
            result['confidence'] = 'medium'
            
        return result
    except Exception as e:
        logger.error(f"Error determining overall trend: {str(e)}", exc_info=True)
        # Provide a fallback if trend determination fails
        result['overall_trend'] = 'neutral'
        result['confidence'] = 'low'
        return result 