"""
Advanced trading strategies for market analysis.

This module provides functions to:
1. Detect market conditions (trend, ranging, volatility)
2. Identify support and resistance levels
3. Generate price targets based on technical analysis
4. Recommend specific trading strategies with entry/exit points
5. Provide risk assessment and position sizing recommendations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import trendln
from scipy import stats

# Set up logging
logger = logging.getLogger(__name__)

def detect_market_condition(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Detect the current market condition (trending, ranging, volatile).
    
    Args:
        data: DataFrame with market data including technical indicators
        
    Returns:
        Dictionary with market condition information:
        - 'condition': Main market condition ('trending', 'ranging', 'volatile')
        - 'sub_condition': More specific condition (e.g., 'strong_uptrend', 'tight_range')
        - 'confidence': Confidence level in the detected condition
        - 'metrics': Metrics used to determine the condition
    """
    if data is None or data.empty or len(data) < 20:
        return {
            'condition': 'unknown',
            'sub_condition': 'insufficient_data',
            'confidence': 'low',
            'metrics': {}
        }
    
    # Extract price data
    close_values = data['close'].dropna().values
    if len(close_values) < 20:
        return {
            'condition': 'unknown',
            'sub_condition': 'insufficient_data',
            'confidence': 'low',
            'metrics': {}
        }
    
    # Calculate metrics for market condition detection
    try:
        # Get the latest values
        latest = data.iloc[-1]
        
        # Linear regression for trend strength - safe sequence handling
        # Take up to last 20 values (for cases with insufficient data)
        n_points = min(20, len(close_values))
        y = close_values[-n_points:]
        x = np.arange(n_points)
        
        # Make sure x and y have the same length (explicit safety check)
        x = x[:len(y)]
        
        # Perform linear regression
        if len(x) > 1:  # We need at least 2 points for linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            if np.isfinite(y.mean()) and y.mean() != 0:
                normalized_slope = slope / y.mean() * 100  # Percentage trend per period
            else:
                normalized_slope = slope * 100  # Default to just the slope without normalization
            trend_strength = abs(r_value)  # R-squared as trend strength
        else:
            # Not enough data for regression
            slope = 0
            normalized_slope = 0
            trend_strength = 0
        
        # Volatility calculation - safe sequence handling
        if len(close_values) >= 21:
            # Calculate price differences
            price_diffs = np.diff(close_values[-21:])
            price_bases = close_values[-21:-1]  # One shorter than the original
            
            # Ensure the arrays have the same shape and no zeros to avoid division by zero
            price_diffs = price_diffs[:len(price_bases)]
            
            # Filter out zeros to avoid division by zero
            valid_indices = price_bases != 0
            if np.any(valid_indices):
                rel_changes = price_diffs[valid_indices] / price_bases[valid_indices]
                volatility = np.std(rel_changes) * 100
            else:
                # Fallback if all values are zero
                volatility = np.std(price_diffs) * 100
        else:
            # Not enough data for volatility calculation, use standard deviation directly
            volatility = np.std(close_values) / np.mean(close_values) * 100 if np.mean(close_values) != 0 else 0
        
        # ADX for trend strength (if available)
        adx = latest.get('ADX_14', None)
        if adx is not None and not pd.isna(adx):
            trend_strength = max(trend_strength, adx / 100)  # Scale ADX to 0-1
        
        # Bollinger Band width for range detection
        bb_upper_key = 'BBU_20_2.0'
        bb_lower_key = 'BBL_20_2.0'
        bb_upper = latest.get(bb_upper_key, None)
        bb_lower = latest.get(bb_lower_key, None)
        
        band_width = None
        if bb_upper is not None and bb_lower is not None and not (pd.isna(bb_upper) or pd.isna(bb_lower)):
            # Safely calculate band width with a positive denominator
            close_price = latest.get('close', 1)
            if close_price > 0:
                band_width = (bb_upper - bb_lower) / close_price * 100  # As percentage of price
    
    except Exception as e:
        logger.error(f"Error calculating market condition metrics: {e}")
        return {
            'condition': 'unknown',
            'sub_condition': 'calculation_error',
            'confidence': 'low',
            'metrics': {'error': str(e)}
        }
    
    # Determine market condition based on metrics
    condition = 'ranging'  # Default condition
    sub_condition = 'sideways'
    confidence = 'medium'
    
    # Trending market detection
    if trend_strength > 0.7 or (adx is not None and adx > 25):
        condition = 'trending'
        if normalized_slope > 0:
            sub_condition = 'strong_uptrend' if trend_strength > 0.85 else 'uptrend'
        else:
            sub_condition = 'strong_downtrend' if trend_strength > 0.85 else 'downtrend'
        confidence = 'high' if trend_strength > 0.85 else 'medium'
    
    # Ranging market detection
    elif trend_strength < 0.3:
        condition = 'ranging'
        if band_width is not None:
            if band_width < 2:  # Tight range
                sub_condition = 'tight_range'
                confidence = 'high'
            else:
                sub_condition = 'wide_range'
                confidence = 'medium'
        else:
            sub_condition = 'sideways'
            confidence = 'medium'
    
    # Volatile market detection
    elif volatility > 3:  # More than 3% daily volatility
        condition = 'volatile'
        if normalized_slope > 0:
            sub_condition = 'volatile_uptrend'
        elif normalized_slope < 0:
            sub_condition = 'volatile_downtrend'
        else:
            sub_condition = 'choppy'
        confidence = 'medium'
    
    # Return condition with metrics
    return {
        'condition': condition,
        'sub_condition': sub_condition,
        'confidence': confidence,
        'metrics': {
            'trend_strength': trend_strength,
            'normalized_slope': normalized_slope,
            'volatility': volatility,
            'adx': adx,
            'band_width': band_width
        }
    }

def detect_support_resistance(data: pd.DataFrame, 
                             window_size: int = 50,
                             max_levels: int = 3) -> Dict[str, Any]:
    """
    Detect support and resistance levels using trendln library.
    
    Args:
        data: DataFrame with market data
        window_size: Number of periods to analyze for support/resistance
        max_levels: Maximum number of support/resistance levels to return
        
    Returns:
        Dictionary with support and resistance levels:
        - 'supports': List of support levels
        - 'resistances': List of resistance levels
        - 'key_level': The most significant level (closest to current price)
        - 'confidence': Confidence in detected levels
    """
    if data is None or data.empty or len(data) < window_size:
        return {
            'supports': [],
            'resistances': [],
            'key_level': None,
            'confidence': 'low'
        }
    
    try:
        # Get price data for analysis
        highs = data['high'].dropna().tail(window_size).values
        lows = data['low'].dropna().tail(window_size).values
        current_price = data['close'].iloc[-1]
        
        # Detect support and resistance levels using trendln's methods
        support_levels = []
        try:
            # Get horizontal support levels
            # Use try/except and fall back to manual support detection if trendln fails
            try:
                # Try with accuracy parameter (fixed even number) instead of mode
                minimaIdxs = trendln.get_extrema(lows, accuracy=8)
                
                # Handle tuple return format
                if isinstance(minimaIdxs, tuple) and len(minimaIdxs) > 0:
                    minimaIdxs = minimaIdxs[0]  # Get first element (minima indexes)
                
                minima = [lows[i] for i in minimaIdxs if i < len(lows)]
            except Exception as e:
                # If trendln fails, use a simple rolling min approach
                logger.warning(f"Falling back to simple support detection: {e}")
                window = min(20, len(lows) // 2)
                if window > 0:
                    # Find local minima using simple rolling window
                    minima = []
                    for i in range(window, len(lows) - window):
                        if all(lows[i] <= lows[j] for j in range(i-window, i+window+1) if j != i):
                            minima.append(lows[i])
                else:
                    minima = []
            
            # Filter supports below current price and sort them
            if minima:
                support_levels = [level for level in minima if level < current_price]
                support_levels = sorted(support_levels, reverse=True)[:max_levels]
        except Exception as e:
            logger.warning(f"Error detecting support levels: {e}")
        
        resistance_levels = []
        try:
            # Get horizontal resistance levels
            # Use try/except and fall back to manual resistance detection if trendln fails
            try:
                # Try with accuracy parameter (fixed even number) instead of mode
                maximaIdxs = trendln.get_extrema(highs, accuracy=8)
                
                # Handle tuple return format
                if isinstance(maximaIdxs, tuple) and len(maximaIdxs) > 1:
                    maximaIdxs = maximaIdxs[1]  # Get second element (maxima indexes)
                elif isinstance(maximaIdxs, tuple) and len(maximaIdxs) > 0:
                    maximaIdxs = maximaIdxs[0]  # Use first element if only one exists
                
                maxima = [highs[i] for i in maximaIdxs if i < len(highs)]
            except Exception as e:
                # If trendln fails, use a simple rolling max approach
                logger.warning(f"Falling back to simple resistance detection: {e}")
                window = min(20, len(highs) // 2)
                if window > 0:
                    # Find local maxima using simple rolling window
                    maxima = []
                    for i in range(window, len(highs) - window):
                        if all(highs[i] >= highs[j] for j in range(i-window, i+window+1) if j != i):
                            maxima.append(highs[i])
                else:
                    maxima = []
            
            # Filter resistances above current price and sort them
            if maxima:
                resistance_levels = [level for level in maxima if level > current_price]
                resistance_levels = sorted(resistance_levels)[:max_levels]
        except Exception as e:
            logger.warning(f"Error detecting resistance levels: {e}")
        
        # Determine the most significant (key) level
        key_level = None
        if support_levels and resistance_levels:
            # Find closest level to current price
            closest_support = support_levels[0]
            closest_resistance = resistance_levels[0]
            
            if (current_price - closest_support) < (closest_resistance - current_price):
                key_level = closest_support
            else:
                key_level = closest_resistance
        elif support_levels:
            key_level = support_levels[0]
        elif resistance_levels:
            key_level = resistance_levels[0]
        
        # Determine confidence based on number of detected levels
        confidence = 'low'
        if (len(support_levels) >= 2 or len(resistance_levels) >= 2):
            confidence = 'high'
        elif (len(support_levels) >= 1 or len(resistance_levels) >= 1):
            confidence = 'medium'
        
        return {
            'supports': support_levels,
            'resistances': resistance_levels,
            'key_level': key_level,
            'confidence': confidence
        }
        
    except Exception as e:
        logger.error(f"Error detecting support and resistance levels: {e}")
        return {
            'supports': [],
            'resistances': [],
            'key_level': None,
            'confidence': 'low',
            'error': str(e)
        }

def calculate_price_targets(data: pd.DataFrame, 
                           sr_levels: Dict[str, Any],
                           atr_multiplier: float = 2.0) -> Dict[str, Any]:
    """
    Calculate price targets using multiple methods.
    
    Args:
        data: DataFrame with market data
        sr_levels: Support and resistance levels
        atr_multiplier: Multiplier for ATR-based targets
        
    Returns:
        Dictionary with price targets:
        - 'next_target': The next price target
        - 'stop_loss': Recommended stop loss level
        - 'targets': List of additional targets in order
        - 'target_methods': Methods used to derive targets
    
    Stop Loss Logic:
        - For long trades (uptrend): stop_loss = support - max(ATR*0.5, 0.5% of support)
        - For short trades (downtrend): stop_loss = resistance + max(ATR*0.5, 0.5% of resistance)
        - If no support/resistance, fallback to ATR-based stop loss
        - ATR is calculated using pandas_ta if not already present in the DataFrame
    
    This ensures the stop loss is always a buffer away from entry/support, never equal to entry, and adapts to volatility.
    """
    if data is None or data.empty or len(data) < 20:
        return {
            'next_target': None,
            'stop_loss': None,
            'targets': [],
            'target_methods': []
        }
    
    try:
        # Get current price and trend direction
        current_price = data['close'].iloc[-1]
        
        # Calculate trend direction using linear regression
        close_values = data['close'].dropna().tail(20).values
        x = np.arange(len(close_values))
        slope, _, _, _, _ = stats.linregress(x, close_values)
        trend_direction = 'up' if slope > 0 else 'down'
        
        # Initialize targets and methods lists
        targets = []  # type: List[float]
        methods = []  # type: List[str]
        
        # 1. Support/Resistance based targets
        if sr_levels['key_level'] is not None:
            if trend_direction == 'up' and sr_levels['resistances']:
                for level in sr_levels['resistances']:
                    targets.append(float(level))
                    methods.append('resistance_level')
            elif trend_direction == 'down' and sr_levels['supports']:
                for level in sr_levels['supports']:
                    targets.append(float(level))
                    methods.append('support_level')
        
        # 2. ATR-based targets
        # Ensure ATR is available, calculate with pandas_ta if not
        if 'ATR_14' not in data.columns:
            try:
                import pandas_ta as ta
                data['ATR_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            except ImportError:
                raise ImportError("pandas_ta is required for ATR calculation.")
        atr_value = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else None
        if atr_value is not None and not pd.isna(atr_value):
            if trend_direction == 'up':
                atr_target = current_price + (atr_value * atr_multiplier)
                targets.append(float(atr_target))
                methods.append('atr_extension')
            else:
                atr_target = current_price - (atr_value * atr_multiplier)
                targets.append(float(atr_target))
                methods.append('atr_extension')
        
        # 3. Fibonacci extensions based on recent swing
        if len(data) >= 50:
            high_idx = data['high'].iloc[-50:].idxmax()
            low_idx = data['low'].iloc[-50:].idxmin()
            
            # Determine swing points based on which came first
            if high_idx < low_idx:  # Downward swing
                swing_high = data.loc[high_idx, 'high']
                swing_low = data.loc[low_idx, 'low']
                swing_diff = swing_high - swing_low
                
                if trend_direction == 'up':  # Possible reversal
                    fib_target = swing_low + (swing_diff * 0.618)  # 61.8% retracement
                    targets.append(float(fib_target))
                    methods.append('fibonacci_retracement')
                else:  # Continuation
                    fib_target = swing_low - (swing_diff * 0.618)  # 61.8% extension
                    targets.append(float(fib_target))
                    methods.append('fibonacci_extension')
            else:  # Upward swing
                swing_low = data.loc[low_idx, 'low']
                swing_high = data.loc[high_idx, 'high']
                swing_diff = swing_high - swing_low
                
                if trend_direction == 'up':  # Continuation
                    fib_target = swing_high + (swing_diff * 0.618)  # 61.8% extension
                    targets.append(float(fib_target))
                    methods.append('fibonacci_extension')
                else:  # Possible reversal
                    fib_target = swing_high - (swing_diff * 0.618)  # 61.8% retracement
                    targets.append(float(fib_target))
                    methods.append('fibonacci_retracement')
        
        # Sort targets based on trend direction
        sorted_targets_methods = []
        if targets and methods:
            if trend_direction == 'up':
                # For uptrend, sort targets in ascending order
                sorted_targets_methods = sorted(zip(targets, methods))
            else:
                # For downtrend, sort targets in descending order
                sorted_targets_methods = sorted(zip(targets, methods), reverse=True)
                
            # Safely unpack the sorted pairs
            if sorted_targets_methods:
                targets = [t for t, _ in sorted_targets_methods]
                methods = [m for _, m in sorted_targets_methods]
                
        # Determine stop loss based on support/resistance and ATR
        stop_loss = None
        # Long/Uptrend: stop loss below support
        if trend_direction == 'up' and sr_levels['supports']:
            support = float(sr_levels['supports'][0])
            atr_buffer = atr_value * 0.5 if atr_value is not None and not pd.isna(atr_value) else 0
            pct_buffer = support * 0.005  # 0.5% of support
            stop_loss = support - max(atr_buffer, pct_buffer)
        # Short/Downtrend: stop loss above resistance
        elif trend_direction == 'down' and sr_levels['resistances']:
            resistance = float(sr_levels['resistances'][0])
            atr_buffer = atr_value * 0.5 if atr_value is not None and not pd.isna(atr_value) else 0
            pct_buffer = resistance * 0.005  # 0.5% of resistance
            stop_loss = resistance + max(atr_buffer, pct_buffer)
        # If no support/resistance stop loss, use ATR
        if stop_loss is None and atr_value is not None and not pd.isna(atr_value):
            stop_loss = current_price - (atr_value * 2) if trend_direction == 'up' else current_price + (atr_value * 2)
        
        # Get next immediate target
        next_target = targets[0] if targets else None
        
        return {
            'next_target': next_target,
            'stop_loss': stop_loss,
            'targets': targets,
            'target_methods': methods,
            'trend_direction': trend_direction
        }
        
    except Exception as e:
        logger.error(f"Error calculating price targets: {e}")
        return {
            'next_target': None,
            'stop_loss': None,
            'targets': [],
            'target_methods': [],
            'error': str(e)
        }

def calculate_risk_metrics(data: pd.DataFrame, 
                          price_targets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate risk assessment metrics for trading.
    
    Args:
        data: DataFrame with market data
        price_targets: Dictionary with price targets
        
    Returns:
        Dictionary with risk assessment metrics:
        - 'risk_reward_ratio': Risk to reward ratio
        - 'position_size': Recommended position sizing (as % of capital)
        - 'max_loss_pct': Maximum loss percentage (risk per trade)
        - 'volatility_adjusted_risk': Risk adjusted for volatility
    
    Position Sizing Logic:
        - risk_per_trade is set by confidence:
            - High: 1% of capital
            - Medium: 0.5% of capital
            - Low: 0.25% of capital
        - position_size = min(10%, risk_per_trade / (risk per share))
        - All calculations use numpy for reliability
    
    This ensures position sizing is risk-aware, confidence-scaled, and never exceeds the maximum allowed.
    """
    if data is None or data.empty or price_targets['next_target'] is None or price_targets['stop_loss'] is None:
        return {
            'risk_reward_ratio': None,
            'position_size': None,
            'max_loss_pct': None,
            'volatility_adjusted_risk': None
        }
    
    try:
        # Get current price
        current_price = data['close'].iloc[-1]
        
        # Calculate risk (from current price to stop loss)
        risk = abs(current_price - price_targets['stop_loss'])
        risk_pct = (risk / current_price) * 100
        
        # Calculate reward (from current price to next target)
        reward = abs(current_price - price_targets['next_target'])
        reward_pct = (reward / current_price) * 100
        
        # Calculate risk-reward ratio
        risk_reward_ratio = reward / risk if risk > 0 else 0
        
        # Get ATR value for volatility
        atr_value = data['ATR_14'].iloc[-1] if 'ATR_14' in data.columns else None
        atr_pct = (atr_value / current_price) * 100 if atr_value is not None and not pd.isna(atr_value) else None
        
        # Determine confidence level from price_targets if available
        confidence = price_targets.get('confidence', 'medium')
        # Set risk per trade based on confidence
        if confidence == 'high':
            risk_per_trade = 0.01  # 1%
        elif confidence == 'medium':
            risk_per_trade = 0.005  # 0.5%
        else:
            risk_per_trade = 0.0025  # 0.25%
        max_position_pct = 0.10  # 10% cap
        # Calculate position size as a percentage of capital
        position_size = 0
        if risk > 0:
            raw_position = risk_per_trade / (risk / current_price)
            position_size = float(np.clip(raw_position, 0, max_position_pct)) * 100  # as percent
        
        # Adjust risk for volatility
        volatility_adjusted_risk = None
        if atr_pct is not None:
            # If market is more volatile than the risk, reduce position size
            volatility_factor = atr_pct / risk_pct if risk_pct > 0 else 0
            volatility_adjusted_risk = risk_pct * volatility_factor
        
        return {
            'risk_reward_ratio': risk_reward_ratio,
            'position_size': position_size,  # As a percentage of available capital
            'max_loss_pct': risk_per_trade,
            'volatility_adjusted_risk': volatility_adjusted_risk,
            'risk_amount': risk,
            'risk_pct': risk_pct,
            'reward_amount': reward,
            'reward_pct': reward_pct
        }
        
    except Exception as e:
        logger.error(f"Error calculating risk metrics: {e}")
        return {
            'risk_reward_ratio': None,
            'position_size': None,
            'max_loss_pct': None,
            'volatility_adjusted_risk': None,
            'error': str(e)
        }

def generate_trade_recommendation(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive trade recommendations with entry/exit points and risk assessment.
    
    Args:
        data: DataFrame with market data including technical indicators
        
    Returns:
        Dictionary with detailed trade recommendations:
        - 'market_condition': Market condition assessment
        - 'strategy': Recommended trading strategy
        - 'action': Specific action to take (buy, sell, hold)
        - 'entry_points': Recommended entry points with conditions
        - 'exit_points': Recommended exit points (targets and stop loss)
        - 'risk_assessment': Risk metrics for the trade
        - 'supportive_indicators': Indicators supporting the recommendation
        - 'contrary_indicators': Indicators contrary to the recommendation
        - 'confidence': Overall confidence in the recommendation
    """
    if data is None or data.empty or len(data) < 20:
        return {
            'market_condition': {'condition': 'unknown'},
            'strategy': 'insufficient_data',
            'action': 'hold',
            'entry_points': [],
            'exit_points': {'targets': [], 'stop_loss': None},
            'risk_assessment': {},
            'supportive_indicators': [],
            'contrary_indicators': [],
            'confidence': 'low'
        }
    
    try:
        # 1. Detect market condition
        market_condition = detect_market_condition(data)
        
        # 2. Detect support and resistance levels
        sr_levels = detect_support_resistance(data)
        
        # 3. Calculate price targets
        price_targets = calculate_price_targets(data, sr_levels)
        
        # 4. Calculate risk metrics
        risk_metrics = calculate_risk_metrics(data, price_targets)
        
        # 5. Determine supportive and contrary indicators
        latest = data.iloc[-1]
        supportive = []
        contrary = []
        
        # Check RSI
        if 'rsi_14' in latest:
            rsi = latest['rsi_14']
            if not pd.isna(rsi):
                if price_targets['trend_direction'] == 'up':
                    if rsi > 50:
                        supportive.append(('RSI', rsi, 'above_50'))
                    elif rsi < 30:
                        supportive.append(('RSI', rsi, 'oversold'))
                    else:
                        contrary.append(('RSI', rsi, 'below_50'))
                else:
                    if rsi < 50:
                        supportive.append(('RSI', rsi, 'below_50'))
                    elif rsi > 70:
                        supportive.append(('RSI', rsi, 'overbought'))
                    else:
                        contrary.append(('RSI', rsi, 'above_50'))
        
        # Check MACD
        if all(col in latest for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
            macd = latest['MACD_12_26_9']
            signal = latest['MACDs_12_26_9']
            if not (pd.isna(macd) or pd.isna(signal)):
                if price_targets['trend_direction'] == 'up':
                    if macd > signal:
                        supportive.append(('MACD', f"{macd:.4f} > {signal:.4f}", 'bullish_crossover'))
                    else:
                        contrary.append(('MACD', f"{macd:.4f} < {signal:.4f}", 'bearish_crossover'))
                else:
                    if macd < signal:
                        supportive.append(('MACD', f"{macd:.4f} < {signal:.4f}", 'bearish_crossover'))
                    else:
                        contrary.append(('MACD', f"{macd:.4f} > {signal:.4f}", 'bullish_crossover'))
        
        # Check Stochastic
        if all(col in latest for col in ['STOCHk_14_3_3', 'STOCHd_14_3_3']):
            k = latest['STOCHk_14_3_3']
            d = latest['STOCHd_14_3_3']
            if not (pd.isna(k) or pd.isna(d)):
                if price_targets['trend_direction'] == 'up':
                    if k > d:
                        supportive.append(('Stochastic', f"K({k:.1f}) > D({d:.1f})", 'bullish_crossover'))
                    elif k < 20 and d < 20:
                        supportive.append(('Stochastic', f"K({k:.1f}) D({d:.1f})", 'oversold'))
                    elif k > 80 and d > 80:
                        contrary.append(('Stochastic', f"K({k:.1f}) D({d:.1f})", 'overbought'))
                else:
                    if k < d:
                        supportive.append(('Stochastic', f"K({k:.1f}) < D({d:.1f})", 'bearish_crossover'))
                    elif k > 80 and d > 80:
                        supportive.append(('Stochastic', f"K({k:.1f}) D({d:.1f})", 'overbought'))
                    elif k < 20 and d < 20:
                        contrary.append(('Stochastic', f"K({k:.1f}) D({d:.1f})", 'oversold'))
        
        # 6. Determine recommended trading strategy based on market condition
        strategy = 'hold_cash'  # Default strategy
        action = 'hold'
        confidence = 'low'
        entry_points = []
        
        # Define strategies based on market conditions
        if market_condition['condition'] == 'trending':
            if market_condition['sub_condition'] in ['uptrend', 'strong_uptrend']:
                strategy = 'trend_following'
                action = 'buy'
                entry_points = [
                    {'price': sr_levels['supports'][0] if sr_levels['supports'] else None, 
                     'condition': 'pullback_to_support'},
                    {'price': data['close'].iloc[-1], 
                     'condition': 'immediate_entry'},
                ]
                confidence = 'high' if market_condition['sub_condition'] == 'strong_uptrend' and len(supportive) > len(contrary) else 'medium'
                
            elif market_condition['sub_condition'] in ['downtrend', 'strong_downtrend']:
                strategy = 'trend_following'
                action = 'sell'
                entry_points = [
                    {'price': sr_levels['resistances'][0] if sr_levels['resistances'] else None, 
                     'condition': 'pullback_to_resistance'},
                    {'price': data['close'].iloc[-1], 
                     'condition': 'immediate_entry'},
                ]
                confidence = 'high' if market_condition['sub_condition'] == 'strong_downtrend' and len(supportive) > len(contrary) else 'medium'
                
        elif market_condition['condition'] == 'ranging':
            if market_condition['sub_condition'] in ['tight_range', 'sideways']:
                # Range trading strategy
                strategy = 'range_trading'
                current_price = data['close'].iloc[-1]
                
                if sr_levels['supports'] and sr_levels['resistances']:
                    nearest_support = sr_levels['supports'][0]
                    nearest_resistance = sr_levels['resistances'][0]
                    
                    # Determine if we're closer to support or resistance
                    support_distance = current_price - nearest_support
                    resistance_distance = nearest_resistance - current_price
                    
                    if support_distance < resistance_distance:
                        # Closer to support, consider buying
                        action = 'buy'
                        entry_points = [
                            {'price': nearest_support, 'condition': 'buy_near_support'},
                        ]
                    else:
                        # Closer to resistance, consider selling
                        action = 'sell'
                        entry_points = [
                            {'price': nearest_resistance, 'condition': 'sell_near_resistance'},
                        ]
                    
                    confidence = 'medium' if market_condition['confidence'] == 'high' else 'low'
            
            elif market_condition['sub_condition'] == 'wide_range':
                # For wide range, suggest a more cautious approach
                strategy = 'wait_for_breakout'
                action = 'hold'
                
                if sr_levels['supports'] and sr_levels['resistances']:
                    entry_points = [
                        {'price': sr_levels['resistances'][0] * 1.01, 'condition': 'breakout_above_resistance'},
                        {'price': sr_levels['supports'][0] * 0.99, 'condition': 'breakdown_below_support'},
                    ]
                
                confidence = 'low'
        
        elif market_condition['condition'] == 'volatile':
            # For volatile markets, suggest reduced position size
            if market_condition['sub_condition'] == 'volatile_uptrend':
                strategy = 'pullback_entries'
                action = 'buy'
                entry_points = [
                    {'price': sr_levels['supports'][0] if sr_levels['supports'] else None, 
                     'condition': 'buy_on_dip'},
                ]
                confidence = 'medium' if len(supportive) > len(contrary) else 'low'
                
            elif market_condition['sub_condition'] == 'volatile_downtrend':
                strategy = 'rally_entries'
                action = 'sell'
                entry_points = [
                    {'price': sr_levels['resistances'][0] if sr_levels['resistances'] else None, 
                     'condition': 'sell_on_rally'},
                ]
                confidence = 'medium' if len(supportive) > len(contrary) else 'low'
                
            else:  # choppy
                strategy = 'reduce_exposure'
                action = 'hold'
                entry_points = []
                confidence = 'low'
        
        # Construct exit points
        exit_points = {
            'take_profit': price_targets['targets'],
            'stop_loss': price_targets['stop_loss']
        }
        
        # Compile the final recommendation
        recommendation = {
            'market_condition': market_condition,
            'strategy': strategy,
            'action': action,
            'entry_points': entry_points,
            'exit_points': exit_points,
            'risk_assessment': risk_metrics,
            'supportive_indicators': supportive,
            'contrary_indicators': contrary,
            'confidence': confidence
        }
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Error generating trade recommendation: {e}")
        return {
            'market_condition': {'condition': 'error', 'error': str(e)},
            'strategy': 'error',
            'action': 'hold',
            'entry_points': [],
            'exit_points': {'targets': [], 'stop_loss': None},
            'risk_assessment': {},
            'supportive_indicators': [],
            'contrary_indicators': [],
            'confidence': 'low'
        }

def detect_regime(df):
    """
    Detect volatility and trend regime using rolling std, ADX, and BBands width.
    Args:
        df (pd.DataFrame): DataFrame with price and indicator columns.
    Returns:
        dict: {'trend_regime': str, 'volatility_regime': str, 'confidence': str, 'metrics': dict}
    """
    metrics = {}
    if df is None or len(df) < 20 or 'close' not in df:
        return {
            'trend_regime': 'ambiguous',
            'volatility_regime': 'ambiguous',
            'confidence': 'low',
            'metrics': metrics
        }
    # Volatility regime: use rolling std and BBands width
    closes = df['close'].dropna()
    if len(closes) < 20:
        return {
            'trend_regime': 'ambiguous',
            'volatility_regime': 'ambiguous',
            'confidence': 'low',
            'metrics': metrics
        }
    rolling_std = closes.rolling(window=20).std().iloc[-1]
    price = closes.iloc[-1]
    # BBands width
    bb_upper = df['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in df else None
    bb_lower = df['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in df else None
    if bb_upper is not None and bb_lower is not None and price != 0:
        band_width = (bb_upper - bb_lower) / price * 100
    else:
        band_width = None
    # Volatility regime logic
    if band_width is not None and band_width > 4:
        volatility_regime = 'high_volatility'
    elif rolling_std / price * 100 > 2:
        volatility_regime = 'high_volatility'
    elif band_width is not None and band_width < 2:
        volatility_regime = 'low_volatility'
    elif rolling_std / price * 100 < 1:
        volatility_regime = 'low_volatility'
    else:
        volatility_regime = 'ambiguous'
    metrics['rolling_std_pct'] = rolling_std / price * 100 if price != 0 else None
    metrics['band_width'] = band_width
    # Trend regime: use ADX
    adx = df['ADX_14'].iloc[-1] if 'ADX_14' in df else None
    if adx is not None:
        metrics['adx'] = adx
        if adx > 25:
            trend_regime = 'trending'
        elif adx < 15:
            trend_regime = 'range_bound'
        else:
            trend_regime = 'ambiguous'
    else:
        trend_regime = 'ambiguous'
    # Confidence
    if trend_regime != 'ambiguous' and volatility_regime != 'ambiguous':
        confidence = 'high'
    elif trend_regime != 'ambiguous' or volatility_regime != 'ambiguous':
        confidence = 'medium'
    else:
        confidence = 'low'
    return {
        'trend_regime': trend_regime,
        'volatility_regime': volatility_regime,
        'confidence': confidence,
        'metrics': metrics
    }

def suggest_strategy_for_regime(regime):
    """
    Suggest a trading strategy based on detected regime for leveraged perpetual futures.
    Args:
        regime (dict): Output from detect_regime, or None.
    Returns:
        dict: {'strategy': str, 'educational_rationale': str, 'actionable_advice': str}
    """
    if regime is None or not isinstance(regime, dict):
        return {
            'strategy': 'insufficient_data',
            'educational_rationale': 'Not enough data to determine a safe or effective strategy. Wait for more price action and indicator signals before trading.',
            'actionable_advice': 'Do not open new positions. Monitor the market and wait for clearer signals.'
        }
    trend = regime.get('trend_regime', 'ambiguous')
    vol = regime.get('volatility_regime', 'ambiguous')
    confidence = regime.get('confidence', 'low')
    # Trending + High Volatility
    if trend == 'trending' and vol == 'high_volatility':
        return {
            'strategy': 'trend_following',
            'educational_rationale': 'Trending markets with high volatility are ideal for trend-following strategies. Use leverage judiciously to maximize gains, but be aware of increased risk and potential for large swings.',
            'actionable_advice': 'Consider entering in the direction of the trend on pullbacks. Use tight stop losses and moderate leverage (e.g., 2-5x). Scale in/out as volatility increases. Avoid overexposure.'
        }
    # Range-bound + Low Volatility
    if trend == 'range_bound' and vol == 'low_volatility':
        return {
            'strategy': 'range_trading',
            'educational_rationale': 'Range-bound, low volatility markets are best suited for mean-reversion and range-trading strategies. Leverage can be used, but with smaller position sizes due to limited price movement.',
            'actionable_advice': 'Buy near support, sell near resistance. Use tight stops and small to moderate leverage (e.g., 1-3x). Take profits quickly. Avoid chasing breakouts.'
        }
    # Ambiguous regime
    if trend == 'ambiguous' or vol == 'ambiguous' or confidence == 'low':
        return {
            'strategy': 'reduce_exposure',
            'educational_rationale': 'When the market regime is unclear or confidence is low, it is prudent to reduce exposure and avoid aggressive trading. Uncertain conditions increase the risk of whipsaws and losses.',
            'actionable_advice': 'Reduce position sizes, tighten stops, or stay on the sidelines. Avoid using high leverage. Wait for clearer signals before re-entering.'
        }
    # Fallback
    return {
        'strategy': 'insufficient_data',
        'educational_rationale': 'Not enough data to determine a safe or effective strategy. Wait for more price action and indicator signals before trading.',
        'actionable_advice': 'Do not open new positions. Monitor the market and wait for clearer signals.'
    }

def generate_watch_for_signals(regime, metrics):
    """
    Generate 'what to watch for' signals based on regime and indicator values.
    Args:
        regime (dict or None): Regime output from detect_regime, or None.
        metrics (dict): Indicator values (e.g., BBands width, ADX, RSI).
    Returns:
        list of str: Signals for what to watch for to trigger actionable strategies.
    """
    signals = []
    # If regime is None or strategy is 'insufficient_data', return generic signal
    if regime is None or not isinstance(regime, dict):
        return ["Watch for more price action and indicator signals to develop before trading."]
    trend = regime.get('trend_regime', 'ambiguous')
    vol = regime.get('volatility_regime', 'ambiguous')
    confidence = regime.get('confidence', 'low')
    # If actionable strategy, return empty list
    if (trend == 'trending' and vol == 'high_volatility' and confidence in ['high', 'medium']) or \
       (trend == 'range_bound' and vol == 'low_volatility' and confidence in ['high', 'medium']):
        return []
    # Otherwise, generate contextually relevant signals
    bbands_width = metrics.get('bbands_width') or metrics.get('band_width')
    adx = metrics.get('ADX_14') or metrics.get('adx')
    rsi = metrics.get('rsi_14') or metrics.get('RSI_14')
    # Volatility (BBands width)
    if bbands_width is not None and bbands_width < 4:
        signals.append(f"Watch for volatility (Bollinger Band width) to increase above 4% (currently {bbands_width:.1f}%).")
    # ADX (trend strength)
    if adx is not None and adx < 25:
        signals.append(f"Watch for ADX to rise above 25 (currently {adx:.1f}).")
    # RSI (momentum)
    if rsi is not None and 45 < rsi < 55:
        signals.append(f"Watch for RSI to move decisively away from 50 (currently {rsi:.1f}).")
    # If no specific signals, provide a generic one
    if not signals:
        signals.append("Watch for stronger trend, higher volatility, or clearer indicator signals before trading.")
    return signals 