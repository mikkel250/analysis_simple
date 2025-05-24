"""
Technical indicator summarization module.

This module provides functions for summarizing and interpreting
technical indicators from market data.
"""

from typing import Dict, Any, Union, List, Optional, Tuple
import pandas as pd
import numpy as np
from src.config.logging_config import get_logger
from src.analysis.error_handling import (
    ValidationError, IndicatorError, validate_dataframe, safe_operation, safe_dataframe_operation
)

# Set up logger
logger = get_logger(__name__)

def summarize_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of technical indicators from market data.
    
    Args:
        data: DataFrame with market data and technical indicators
        
    Returns:
        Dict with indicator summaries and interpretations
        
    Raises:
        ValidationError: If the data doesn't meet requirements
        IndicatorError: If there's an error analyzing indicators
    """
    logger.info("Summarizing technical indicators")
    
    try:
        # Validate input data
        required_columns = ['close']  # Minimal requirement
        validate_dataframe(data, required_columns=required_columns, min_rows=10)
        
        # Log data shape and available columns for debugging
        logger.debug(f"Data shape: {data.shape}")
        logger.debug(f"Available columns: {data.columns.tolist()}")
        
        # Find the last valid row for key indicators
        required_cols = ['rsi_14', 'MACD_12_26_9', 'MACDs_12_26_9']
        valid_data = data.dropna(subset=required_cols)
        if valid_data.empty:
            logger.debug(f"All required columns {required_cols} are present but all values are NaN. Cannot summarize indicators.")
            raise IndicatorError(f"Cannot summarize indicators: all required columns {required_cols} are NaN.")
        else:
            current = valid_data.iloc[-1]
        
        # Initialize results dictionary
        result = {
            "moving_averages": {},
            "oscillators": {},
            "volume": {},
            "trends": {},
            "patterns": {},
            "summary": {
                "bullish_indicators": 0,
                "bearish_indicators": 0,
                "neutral_indicators": 0,
                "total_indicators": 0,
                "direction": "neutral",
                "strength": "weak"
            }
        }
        
        # Analyze Moving Averages using safe operation
        logger.debug("Analyzing moving averages")
        ma_indicators = safe_operation(
            lambda: _analyze_moving_averages(data),
            fallback={"indicators": [], "summary": {}, "overall": "neutral"},
            error_msg="Error analyzing moving averages, using fallback",
            logger_instance=logger
        )
        result["moving_averages"] = ma_indicators
        
        # Analyze Oscillators using safe operation
        logger.debug("Analyzing oscillators")
        osc_indicators = safe_operation(
            lambda: _analyze_oscillators(data),
            fallback={"indicators": [], "summary": {}, "overall": "neutral"},
            error_msg="Error analyzing oscillators, using fallback",
            logger_instance=logger
        )
        result["oscillators"] = osc_indicators
        
        # Analyze Volume using safe operation
        logger.debug("Analyzing volume indicators")
        vol_indicators = safe_operation(
            lambda: _analyze_volume(data),
            fallback={"indicators": [], "summary": {}, "overall": "neutral"},
            error_msg="Error analyzing volume indicators, using fallback",
            logger_instance=logger
        )
        result["volume"] = vol_indicators
        
        # Analyze Trends using safe operation
        logger.debug("Analyzing trend indicators")
        trend_indicators = safe_operation(
            lambda: _analyze_trends(data),
            fallback={"indicators": [], "summary": {}, "overall": "neutral"},
            error_msg="Error analyzing trend indicators, using fallback",
            logger_instance=logger
        )
        result["trends"] = trend_indicators
        
        # Analyze Patterns using safe operation
        logger.debug("Analyzing chart patterns")
        pattern_indicators = safe_operation(
            lambda: _analyze_patterns(data),
            fallback={"indicators": [], "summary": {}, "overall": "neutral"},
            error_msg="Error analyzing chart patterns, using fallback",
            logger_instance=logger
        )
        result["patterns"] = pattern_indicators
        
        # Count indicators by direction
        all_indicators = []
        for section in [ma_indicators, osc_indicators, vol_indicators, trend_indicators, pattern_indicators]:
            all_indicators.extend(section.get("indicators", []))
        
        for indicator in all_indicators:
            result["summary"]["total_indicators"] += 1
            if indicator.get("direction") == "bullish":
                result["summary"]["bullish_indicators"] += 1
            elif indicator.get("direction") == "bearish":
                result["summary"]["bearish_indicators"] += 1
            else:
                result["summary"]["neutral_indicators"] += 1
        
        # Determine overall direction
        if result["summary"]["bullish_indicators"] > result["summary"]["bearish_indicators"]:
            result["summary"]["direction"] = "bullish"
        elif result["summary"]["bearish_indicators"] > result["summary"]["bullish_indicators"]:
            result["summary"]["direction"] = "bearish"
        else:
            result["summary"]["direction"] = "neutral"
        
        # Determine strength of signal
        total = result["summary"]["total_indicators"]
        if total > 0:
            dominant_ratio = max(
                result["summary"]["bullish_indicators"] / total,
                result["summary"]["bearish_indicators"] / total
            )
            
            if dominant_ratio >= 0.7:
                result["summary"]["strength"] = "strong"
            elif dominant_ratio >= 0.55:
                result["summary"]["strength"] = "moderate"
            else:
                result["summary"]["strength"] = "weak"
        
        logger.info(f"Indicator summary complete: {result['summary']['direction']} ({result['summary']['strength']})")
        return result
        
    except ValidationError as e:
        logger.error(f"Validation error in summarize_indicators: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Error summarizing indicators: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise IndicatorError(error_msg)

def _analyze_moving_averages(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze moving average indicators from the data."""
    try:
        # Validate required data
        required_columns = ['close']
        validate_dataframe(data, required_columns=required_columns)
        
        # Get the most recent data
        current = data.iloc[-1]
        
        # Initialize results
        result = {
            "indicators": [],
            "summary": {
                "price_above_mas": 0,
                "price_below_mas": 0,
                "ma_trends_up": 0,
                "ma_trends_down": 0,
                "total_mas": 0
            },
            "overall": "neutral"
        }
        
        # Check commonly used MAs
        ma_columns = [col for col in data.columns if col.startswith(('sma_', 'ema_'))]
        logger.debug(f"Found {len(ma_columns)} moving average columns")
        
        if not ma_columns:
            logger.warning("No moving average columns found in data")
            return result
            
        # Ensure close price is valid
        close = current['close']
        if pd.isna(close):
            logger.warning("Current close price is NaN, cannot analyze moving averages")
            return result
        
        # Analyze each MA
        for ma_col in ma_columns:
            try:
                # Skip if MA value is not available
                if pd.isna(current[ma_col]):
                    logger.debug(f"Skipping {ma_col} due to NaN value")
                    continue
                    
                ma_value = current[ma_col]
                ma_slope = _calculate_slope(data[ma_col].tail(5))
                
                # Determine if price is above or below MA
                price_vs_ma = "above" if close > ma_value else "below"
                
                # Determine if MA is trending up or down
                ma_trend = "up" if ma_slope > 0 else "down" if ma_slope < 0 else "sideways"
                
                # Determine bullish/bearish indication
                if price_vs_ma == "above" and ma_trend == "up":
                    direction = "bullish"
                elif price_vs_ma == "below" and ma_trend == "down":
                    direction = "bearish"
                else:
                    # Mixed signals
                    direction = "neutral"
                
                # Add to indicator list
                result["indicators"].append({
                    "name": ma_col,
                    "value": ma_value,
                    "price_relation": price_vs_ma,
                    "trend": ma_trend,
                    "direction": direction
                })
                
                # Update summary counts
                result["summary"]["total_mas"] += 1
                if price_vs_ma == "above":
                    result["summary"]["price_above_mas"] += 1
                else:
                    result["summary"]["price_below_mas"] += 1
                    
                if ma_trend == "up":
                    result["summary"]["ma_trends_up"] += 1
                elif ma_trend == "down":
                    result["summary"]["ma_trends_down"] += 1
            except Exception as e:
                logger.warning(f"Error analyzing moving average {ma_col}: {str(e)}")
                continue  # Skip to next MA on error
        
        # Determine overall MA signal
        if result["summary"]["total_mas"] > 0:
            if result["summary"]["price_above_mas"] > result["summary"]["price_below_mas"] and \
               result["summary"]["ma_trends_up"] > result["summary"]["ma_trends_down"]:
                result["overall"] = "bullish"
            elif result["summary"]["price_below_mas"] > result["summary"]["price_above_mas"] and \
                 result["summary"]["ma_trends_down"] > result["summary"]["ma_trends_up"]:
                result["overall"] = "bearish"
            else:
                result["overall"] = "neutral"
        else:
            logger.warning("No valid moving averages found for analysis")
                
        return result
    except ValidationError as e:
        logger.error(f"Validation error analyzing moving averages: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error analyzing moving averages: {str(e)}", exc_info=True)
        return {"indicators": [], "summary": {}, "overall": "error"}

def _analyze_oscillators(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze oscillator indicators from the data."""
    try:
        # Get the most recent data
        current = data.iloc[-1]
        
        # Initialize results
        result = {
            "indicators": [],
            "summary": {
                "bullish_signals": 0,
                "bearish_signals": 0,
                "neutral_signals": 0,
                "overbought_signals": 0,
                "oversold_signals": 0,
                "total_oscillators": 0
            },
            "overall": "neutral"
        }
        
        # RSI analysis
        if 'rsi_14' in data.columns and not pd.isna(current['rsi_14']):
            rsi = current['rsi_14']
            
            # Determine RSI condition
            if rsi > 70:
                rsi_condition = "overbought"
                direction = "bearish"
                result["summary"]["overbought_signals"] += 1
            elif rsi < 30:
                rsi_condition = "oversold"
                direction = "bullish"
                result["summary"]["oversold_signals"] += 1
            elif rsi > 50:
                rsi_condition = "neutral_bullish"
                direction = "bullish"
            else:
                rsi_condition = "neutral_bearish"
                direction = "bearish"
                
            # Add to indicators
            result["indicators"].append({
                "name": "RSI(14)",
                "value": rsi,
                "condition": rsi_condition,
                "direction": direction
            })
            
            # Update summary
            result["summary"]["total_oscillators"] += 1
            if direction == "bullish":
                result["summary"]["bullish_signals"] += 1
            elif direction == "bearish":
                result["summary"]["bearish_signals"] += 1
            else:
                result["summary"]["neutral_signals"] += 1
        
        # MACD analysis
        if all(col in data.columns for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
            if not pd.isna(current['MACD_12_26_9']) and not pd.isna(current['MACDs_12_26_9']):
                signal = current['MACDs_12_26_9']
                histogram = current['MACD_12_26_9'] - signal
                
                # Trend of histogram
                hist_trend = _calculate_slope(data['MACD_12_26_9'].tail(5) - data['MACDs_12_26_9'].tail(5))
                
                # Determine MACD condition
                if current['MACD_12_26_9'] > signal and histogram > 0:
                    if hist_trend > 0:
                        macd_condition = "bullish_strengthening"
                    else:
                        macd_condition = "bullish_weakening"
                    direction = "bullish"
                elif current['MACD_12_26_9'] < signal and histogram < 0:
                    if hist_trend < 0:
                        macd_condition = "bearish_strengthening"
                    else:
                        macd_condition = "bearish_weakening"
                    direction = "bearish"
                else:
                    macd_condition = "crossing"
                    direction = "neutral"
                
                # Add to indicators
                result["indicators"].append({
                    "name": "MACD",
                    "value": {
                        "MACD_12_26_9": current['MACD_12_26_9'],
                        "MACDs_12_26_9": signal,
                        "MACDh_12_26_9": histogram
                    },
                    "condition": macd_condition,
                    "direction": direction
                })
                
                # Update summary
                result["summary"]["total_oscillators"] += 1
                if direction == "bullish":
                    result["summary"]["bullish_signals"] += 1
                elif direction == "bearish":
                    result["summary"]["bearish_signals"] += 1
                else:
                    result["summary"]["neutral_signals"] += 1
        
        # Add more oscillators as needed...
        
        # Determine overall oscillator signal
        if result["summary"]["bullish_signals"] > result["summary"]["bearish_signals"]:
            result["overall"] = "bullish"
        elif result["summary"]["bearish_signals"] > result["summary"]["bullish_signals"]:
            result["overall"] = "bearish"
        else:
            result["overall"] = "neutral"
            
        return result
    except Exception as e:
        logger.error(f"Error analyzing oscillators: {str(e)}", exc_info=True)
        return {"indicators": [], "summary": {}, "overall": "error"}

def _analyze_volume(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze volume indicators from the data."""
    try:
        # Initialize results
        result = {
            "indicators": [],
            "summary": {
                "volume_trend": "neutral",
                "bullish_volume": 0,
                "bearish_volume": 0,
                "neutral_volume": 0,
                "total_indicators": 0
            },
            "overall": "neutral"
        }
        
        # Need at least 10 data points for meaningful volume analysis
        if len(data) < 10 or 'volume' not in data.columns:
            return result
            
        # Get recent volume data
        recent_data = data.tail(10)
        current = recent_data.iloc[-1]
        
        # Calculate volume metrics
        avg_volume = recent_data['volume'].mean()
        current_volume = current['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Determine if volume is increasing or decreasing
        volume_trend = _calculate_slope(recent_data['volume'])
        
        # Volume and price relationship
        price_change = current['close'] - recent_data.iloc[-2]['close']
        
        # Analyze basic volume
        volume_condition = "high" if volume_ratio > 1.5 else "low" if volume_ratio < 0.7 else "normal"
        
        if volume_ratio > 1.5 and price_change > 0:
            # High volume on price increase = bullish
            direction = "bullish"
            result["summary"]["bullish_volume"] += 1
        elif volume_ratio > 1.5 and price_change < 0:
            # High volume on price decrease = bearish
            direction = "bearish"
            result["summary"]["bearish_volume"] += 1
        elif volume_ratio < 0.7:
            # Low volume is usually indecisive
            direction = "neutral"
            result["summary"]["neutral_volume"] += 1
        else:
            direction = "neutral"
            result["summary"]["neutral_volume"] += 1
            
        # Add volume indicator
        result["indicators"].append({
            "name": "Volume",
            "value": current_volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio,
            "condition": volume_condition,
            "direction": direction
        })
        
        result["summary"]["total_indicators"] += 1
        result["summary"]["volume_trend"] = "up" if volume_trend > 0 else "down" if volume_trend < 0 else "neutral"
        
        # Overall volume signal
        if result["summary"]["bullish_volume"] > result["summary"]["bearish_volume"]:
            result["overall"] = "bullish"
        elif result["summary"]["bearish_volume"] > result["summary"]["bullish_volume"]:
            result["overall"] = "bearish"
        else:
            result["overall"] = "neutral"
            
        return result
    except Exception as e:
        logger.error(f"Error analyzing volume: {str(e)}", exc_info=True)
        return {"indicators": [], "summary": {}, "overall": "error"}

def _analyze_trends(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze trend indicators from the data."""
    try:
        # Initialize results
        result = {
            "indicators": [],
            "summary": {
                "bullish_trends": 0,
                "bearish_trends": 0,
                "neutral_trends": 0,
                "total_indicators": 0
            },
            "overall": "neutral"
        }
        
        # Need enough data points for trend analysis
        if len(data) < 20:
            return result
            
        # Get recent data
        recent_data = data.tail(20)
        current = recent_data.iloc[-1]
        
        # ADX analysis (Trend Strength)
        if 'adx' in data.columns and not pd.isna(current['adx']):
            adx = current['adx']
            
            # ADX interpretation
            if adx > 25:
                if adx > 50:
                    adx_strength = "very_strong"
                else:
                    adx_strength = "strong"
            elif adx > 15:
                adx_strength = "moderate"
            else:
                adx_strength = "weak"
                
            # For ADX, we need +DI and -DI to determine direction
            if 'plus_di' in data.columns and 'minus_di' in data.columns:
                plus_di = current['plus_di']
                minus_di = current['minus_di']
                
                if plus_di > minus_di:
                    direction = "bullish"
                    result["summary"]["bullish_trends"] += 1
                else:
                    direction = "bearish"
                    result["summary"]["bearish_trends"] += 1
            else:
                direction = "neutral"  # Can't determine without +DI/-DI
                result["summary"]["neutral_trends"] += 1
                
            # Add ADX indicator
            result["indicators"].append({
                "name": "ADX",
                "value": adx,
                "strength": adx_strength,
                "direction": direction
            })
            
            result["summary"]["total_indicators"] += 1
        
        # Overall trend signal
        if result["summary"]["bullish_trends"] > result["summary"]["bearish_trends"]:
            result["overall"] = "bullish"
        elif result["summary"]["bearish_trends"] > result["summary"]["bullish_trends"]:
            result["overall"] = "bearish"
        else:
            result["overall"] = "neutral"
            
        return result
    except Exception as e:
        logger.error(f"Error analyzing trends: {str(e)}", exc_info=True)
        return {"indicators": [], "summary": {}, "overall": "error"}

def _analyze_patterns(data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze chart patterns from the data."""
    # This is a simplified placeholder - real pattern recognition would be more complex
    return {
        "indicators": [],
        "summary": {
            "bullish_patterns": 0,
            "bearish_patterns": 0,
            "neutral_patterns": 0,
            "total_patterns": 0
        },
        "overall": "neutral"
    }

def _calculate_slope(series, periods=5):
    """
    Calculate the slope of a time series.
    
    Args:
        series: The time series data
        periods: Number of periods to use for slope calculation
        
    Returns:
        float: The calculated slope (positive = uptrend, negative = downtrend)
    """
    try:
        if len(series) < periods:
            logger.warning(f"Series too short for slope calculation. Required: {periods}, Found: {len(series)}")
            return 0.0
            
        # Use last n periods
        recent_values = series.tail(periods).values
        if np.isnan(recent_values).any():
            logger.warning("NaN values detected in slope calculation, filling with previous values")
            recent_values = pd.Series(recent_values).fillna(method='ffill').values
            
        # Simple linear slope approximation
        x = np.arange(len(recent_values))
        y = recent_values
        
        # Check if we have all NaN values after filling
        if np.isnan(y).all():
            logger.warning("All NaN values in series, cannot calculate slope")
            return 0.0
            
        # Filter out any remaining NaN values
        mask = ~np.isnan(y)
        if sum(mask) < 2:
            logger.warning("Insufficient valid data points for slope calculation")
            return 0.0
            
        x = x[mask]
        y = y[mask]
        
        # Calculate slope using numpy's polyfit
        slope = np.polyfit(x, y, 1)[0]
        return slope
    except Exception as e:
        logger.error(f"Error calculating slope: {str(e)}", exc_info=True)
        return 0.0  # Return neutral slope on error 