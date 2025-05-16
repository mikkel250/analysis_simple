"""
Breakout Trading Strategy Module

This module provides functions to:
1. Analyze price data to identify support and resistance levels
2. Calculate entry points above resistance and below support for breakout trades
3. Determine appropriate stop loss levels based on volatility and risk management
4. Provide target price levels and risk/reward ratios for breakout trading strategies
5. Assess confidence level in the breakout trade recommendations
6. Implement crypto-specific features for explosive moves
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union

from src.services.trading_strategies import detect_support_resistance, calculate_price_targets

# Set up logging
logger = logging.getLogger(__name__)

class CryptoBreakoutStrategy:
    """Enhanced breakout strategy specifically for crypto markets."""
    
    def __init__(self, 
                 volatility_factor: float = 1.5,
                 min_risk_reward: float = 2.0,
                 max_position_size: float = 0.1,  # 10% max position size
                 base_leverage: float = 10.0,     # Higher base leverage
                 max_leverage: float = 30.0):     # Maximum leverage for strong setups
        """
        Initialize the crypto breakout strategy.
        
        Args:
            volatility_factor: Multiplier for ATR to determine entry buffers
            min_risk_reward: Minimum risk/reward ratio for valid trades
            max_position_size: Maximum position size as decimal (0.1 = 10%)
            base_leverage: Base leverage multiplier for position sizing
            max_leverage: Maximum allowed leverage for strong setups
        """
        self.volatility_factor = volatility_factor
        self.min_risk_reward = min_risk_reward
        self.max_position_size = max_position_size
        self.base_leverage = base_leverage
        self.max_leverage = max_leverage
    
    def calculate_breakout_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate breakout levels with multiple profit targets optimized for crypto."""
        if data is None or data.empty or len(data) < 20:
            return {
                'status': 'error',
                'message': 'Insufficient data for breakout analysis'
            }
        
        try:
            # Get current price
            current_price = data['close'].iloc[-1]

            # Get ATR, RSI, MACD from pre-calculated columns in the DataFrame
            atr = self._get_atr_from_df(data)
            momentum_indicators = self._get_momentum_from_df(data)

            if atr is None or momentum_indicators is None:
                return {
                    'status': 'error',
                    'message': 'Required indicators (ATR, RSI, MACD) not found or invalid in input data'
                }
            
            # Detect support and resistance levels
            levels = detect_support_resistance(data)
            if not levels or 'supports' not in levels or 'resistances' not in levels:
                return {
                    'status': 'error',
                    'message': 'Could not detect support/resistance levels'
                }
            
            # Adaptive buffer based on recent highest high/lowest low
            lookback = 20  # Configurable lookback window
            recent_high = data['high'].iloc[-lookback:].max()
            recent_low = data['low'].iloc[-lookback:].min()
            price_buffer = current_price * 0.002  # 0.2% of price
            buffer = min(atr * self.volatility_factor, price_buffer)
            
            # Long breakout: entry just above recent high + buffer
            long_entry = recent_high + buffer
            # Short breakout: entry just below recent low - buffer
            short_entry = recent_low - buffer
            
            # Calculate multiple profit targets
            long_targets = self._calculate_profit_targets(long_entry, atr, 'long')
            short_targets = self._calculate_profit_targets(short_entry, atr, 'short')
            
            # Calculate position sizes and leverage
            long_position = self._calculate_position_size(
                current_price=current_price,
                entry=long_entry,
                atr=atr,
                direction='long',
                momentum=momentum_indicators
            )
            short_position = self._calculate_position_size(
                current_price=current_price,
                entry=short_entry,
                atr=atr,
                direction='short',
                momentum=momentum_indicators
            )
            
            # Calculate dynamic stop losses
            long_stops = self._calculate_dynamic_stops(
                entry=long_entry,
                atr=atr,
                direction='long',
                momentum=momentum_indicators,
                leverage=long_position['leverage']
            )
            short_stops = self._calculate_dynamic_stops(
                entry=short_entry,
                atr=atr,
                direction='short',
                momentum=momentum_indicators,
                leverage=short_position['leverage']
            )
            
            # Check if market is consolidating
            is_consolidating = _is_market_consolidating(data)
            consolidation_strength = _calculate_consolidation_strength(data) if is_consolidating else 'low'
            
            # Calculate confidence levels
            long_confidence = _assign_confidence_level(
                is_consolidating=is_consolidating,
                consolidation_strength=consolidation_strength,
                price_to_level_ratio=current_price / recent_high,
                target_info=long_targets
            )
            short_confidence = _assign_confidence_level(
                is_consolidating=is_consolidating,
                consolidation_strength=consolidation_strength,
                price_to_level_ratio=current_price / recent_low,
                target_info=short_targets
            )
            
            return {
                'status': 'success',
                'current_price': current_price,
                'atr': atr,
                'momentum': momentum_indicators,
                'market_condition': 'consolidation' if is_consolidating else 'non_consolidation',
                'consolidation_strength': consolidation_strength,
                'supports': levels['supports'],
                'resistances': levels['resistances'],
                'long_breakout': {
                    'entry': long_entry,
                    'entry_distance': (long_entry - current_price) / current_price * 100,  # % above current price
                    'stop': long_stops['initial'],
                    'stop_distance': (long_entry - long_stops['initial']) / long_entry * 100,  # % below entry
                    'targets': [
                        {
                            'price': target['price'],
                            'r_r': target['atr_multiple'],
                            'pct_from_entry': target['pct_from_entry']
                        } for target in long_targets
                    ],
                    'position_size': long_position['size'],
                    'leverage': long_position['leverage'],
                    'risk_pct': long_position['risk_pct'],
                    'confidence': long_confidence
                },
                'short_breakout': {
                    'entry': short_entry,
                    'entry_distance': (current_price - short_entry) / current_price * 100,  # % below current price
                    'stop': short_stops['initial'],
                    'stop_distance': (short_stops['initial'] - short_entry) / short_entry * 100,  # % above entry
                    'targets': [
                        {
                            'price': target['price'],
                            'r_r': target['atr_multiple'],
                            'pct_from_entry': target['pct_from_entry']
                        } for target in short_targets
                    ],
                    'position_size': short_position['size'],
                    'leverage': short_position['leverage'],
                    'risk_pct': short_position['risk_pct'],
                    'confidence': short_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Error in breakout calculation: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _get_atr_from_df(self, data: pd.DataFrame) -> Optional[float]:
        """Get ATR value from DataFrame column 'ATR_14'."""
        if 'ATR_14' in data.columns and not pd.isna(data['ATR_14'].iloc[-1]):
            return data['ATR_14'].iloc[-1]
        logger.warning("ATR_14 not found or is NaN in DataFrame for breakout strategy.")
        return None

    def _get_momentum_from_df(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Get RSI and MACD histogram from DataFrame columns."""
        rsi_col = 'rsi_14' # Assuming this is the standard column name from add_technical_indicators
        macd_hist_col = 'MACDh_12_26_9' # Assuming this is the standard column name
        volume_col = 'volume' # Standard volume column name

        if not all(col in data.columns for col in [rsi_col, macd_hist_col, volume_col, 'close']):
            logger.warning("Required columns for momentum (RSI, MACD Histogram, Volume, Close) not in DataFrame.")
            return None

        rsi = data[rsi_col].iloc[-1]
        macd_hist = data[macd_hist_col].iloc[-1]
        
        if pd.isna(rsi) or pd.isna(macd_hist):
            logger.warning("RSI or MACD Histogram is NaN in DataFrame for breakout strategy.")
            return None

        # Volume trend (simplified, as original calculation was complex and local)
        # This part can be enhanced if a more sophisticated volume trend is needed from pre-calculated indicators
        volume_series = data[volume_col].rolling(window=20).mean()
        vol_change = 0.0
        if len(volume_series) >= 5 and not pd.isna(volume_series.iloc[-1]) and not pd.isna(volume_series.iloc[-5]) and volume_series.iloc[-5] != 0:
            vol_change = (volume_series.iloc[-1] / volume_series.iloc[-5] - 1) * 100
        else:
            logger.debug("Not enough data or NaN in volume series for vol_change calculation in breakout strategy.")

        return {
            'rsi': rsi,
            'macd_hist': macd_hist,
            'volume_trend': vol_change
        }
    
    def _calculate_profit_targets(self, entry: float, atr: float, direction: str) -> List[Dict[str, float]]:
        """
        Calculate multiple profit targets based on ATR and market volatility.
        
        Args:
            entry: Entry price
            atr: Average True Range
            direction: 'long' or 'short'
            
        Returns:
            List of targets with prices and percentages
        """
        # Define target distances in terms of ATR
        target_multipliers = [2.0, 3.5, 5.0]  # Conservative, moderate, aggressive
        targets = []
        
        for i, multiplier in enumerate(target_multipliers):
            target_distance = atr * multiplier
            
            if direction == 'long':
                price = entry + target_distance
                pct_gain = ((price - entry) / entry) * 100
            else:
                price = entry - target_distance
                pct_gain = ((entry - price) / entry) * 100
            
            targets.append({
                'price': price,
                'pct_from_entry': pct_gain,
                'atr_multiple': multiplier,
                'portion': 1.0 / len(target_multipliers)  # Equal position distribution
            })
        
        return targets
    
    def _calculate_position_size(
        self,
        current_price: float,
        entry: float,
        atr: float,
        direction: str,
        momentum: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate dynamic position size and leverage based on market conditions."""
        # Calculate distance to entry as percentage
        entry_distance = abs(entry - current_price) / current_price
        
        # Adjust position size based on distance to entry
        distance_factor = min(2.0, 1 + (entry_distance * 3))
        
        # Calculate momentum-based leverage multiplier
        momentum_score = self._calculate_momentum_score(momentum, direction)
        leverage_mult = 1.0 + (momentum_score * 2)  # Can double leverage for strong momentum
        
        # Calculate volatility-adjusted position size
        volatility_factor = min(1.0, 0.02 / (atr / current_price))
        
        # Base position size adjusted for distance and volatility
        position_size = self.max_position_size * distance_factor * volatility_factor
        
        # Calculate dynamic leverage based on momentum and volatility
        leverage = self.base_leverage * leverage_mult * volatility_factor
        leverage = min(leverage, self.max_leverage)  # Cap at max_leverage (30x)
        
        # Calculate risk percentage (will be managed by dynamic stops)
        risk_pct = (atr / current_price) * 100 * leverage
        
        return {
            'size': position_size,
            'leverage': leverage,
            'risk_pct': risk_pct
        }
    
    def _calculate_momentum_score(
        self,
        momentum: Dict[str, float],
        direction: str
    ) -> float:
        """Calculate momentum score (0.0 to 1.0) for leverage adjustment."""
        score = 0.0
        
        # RSI contribution (0.0 to 0.4)
        rsi = momentum['rsi']
        if direction == 'long':
            if rsi > 70:
                score += 0.4
            elif rsi > 60:
                score += 0.3
            elif rsi > 50:
                score += 0.2
        else:  # short
            if rsi < 30:
                score += 0.4
            elif rsi < 40:
                score += 0.3
            elif rsi < 50:
                score += 0.2
        
        # MACD contribution (0.0 to 0.4)
        macd_hist = momentum['macd_hist']
        if direction == 'long' and macd_hist > 0:
            score += min(0.4, abs(macd_hist) / 100)
        elif direction == 'short' and macd_hist < 0:
            score += min(0.4, abs(macd_hist) / 100)
        
        # Volume trend contribution (0.0 to 0.2)
        vol_change = momentum['volume_trend']
        if vol_change > 0:
            score += min(0.2, vol_change / 100)
        
        return score
    
    def _calculate_dynamic_stops(
        self,
        entry: float,
        atr: float,
        direction: str,
        momentum: Dict[str, float],
        leverage: float
    ) -> Dict[str, Union[float, List[Dict[str, float]]]]:
        """Calculate dynamic stop losses including breakeven and trailing stops."""
        momentum_score = self._calculate_momentum_score(momentum, direction)
        
        # Initial stop is tighter for higher leverage
        initial_stop_mult = max(0.2, 1.0 / leverage)  # Tighter stops for higher leverage
        initial_stop_distance = atr * initial_stop_mult
        
        if direction == 'long':
            initial_stop = entry - initial_stop_distance
            breakeven_stop = entry + (initial_stop_distance * 0.1)  # Small profit if stopped out
        else:
            initial_stop = entry + initial_stop_distance
            breakeven_stop = entry - (initial_stop_distance * 0.1)
        
        # Calculate trailing stops at different profit levels
        profit_levels = [0.5, 1.0, 2.0, 3.0, 5.0]  # ATR multiples
        trailing_stops = []
        
        for level in profit_levels:
            if direction == 'long':
                price_level = entry + (atr * level)
                trail_distance = atr * max(0.2, (1.0 - momentum_score) / 2)
                stop_level = price_level - trail_distance
            else:
                price_level = entry - (atr * level)
                trail_distance = atr * max(0.2, (1.0 - momentum_score) / 2)
                stop_level = price_level + trail_distance
            
            trailing_stops.append({
                'price_trigger': price_level,
                'stop_level': stop_level,
                'atr_multiple': level
            })
        
        return {
            'initial': initial_stop,
            'breakeven': breakeven_stop,
            'trailing': trailing_stops
        }

def analyze_breakout_strategy(data: pd.DataFrame, 
                            volatility_factor: float = 1.5, 
                            min_risk_reward: float = 2.0) -> Dict[str, Any]:
    """
    Legacy function maintained for backward compatibility.
    Uses the new CryptoBreakoutStrategy class internally.
    """
    strategy = CryptoBreakoutStrategy(
        volatility_factor=volatility_factor,
        min_risk_reward=min_risk_reward
    )
    return strategy.calculate_breakout_levels(data)

def _is_market_consolidating(data: pd.DataFrame, window: int = 20, threshold: float = 0.03) -> bool:
    """
    Determine if the market is in a consolidation or ranging phase.
    
    Args:
        data: DataFrame with price data
        window: Number of periods to analyze for consolidation
        threshold: Maximum percentage range for consolidation
        
    Returns:
        Boolean indicating if market is consolidating
    """
    if len(data) < window:
        return False
    
    recent_data = data.tail(window)
    
    # Calculate high and low bounds of the range
    highest = recent_data['high'].max()
    lowest = recent_data['low'].min()
    
    # Calculate range as percentage
    price_range = (highest - lowest) / lowest
    
    # Check ADX for trend strength if available
    has_weak_trend = True
    if 'ADX_14' in data.columns and not pd.isna(data['ADX_14'].iloc[-1]):
        adx = data['ADX_14'].iloc[-1]
        has_weak_trend = adx < 25  # ADX below 25 suggests weak trend
    
    # Market is consolidating if range is below threshold and ADX indicates weak trend
    return price_range <= threshold and has_weak_trend

def _calculate_consolidation_strength(data: pd.DataFrame, window: int = 20) -> str:
    """
    Calculate the strength of consolidation based on narrowness of range and duration.
    
    Args:
        data: DataFrame with price data
        window: Number of periods to analyze
        
    Returns:
        String indicating consolidation strength ('low', 'medium', 'high')
    """
    if len(data) < window:
        return 'low'
    
    recent_data = data.tail(window)
    
    # Calculate high and low bounds of the range
    highest = recent_data['high'].max()
    lowest = recent_data['low'].min()
    current = data['close'].iloc[-1]
    
    # Calculate range as percentage
    price_range = (highest - lowest) / lowest
    
    # Calculate volume decline (decreasing volume often indicates consolidation)
    volume_trend = -1
    if 'volume' in data.columns:
        try:
            volume_data = recent_data['volume'].dropna()
            if len(volume_data) >= 5:
                x = np.arange(len(volume_data))
                y = volume_data.values
                slope, _, _, _, _ = np.polyfit(x, y, 1, full=True)
                volume_trend = slope[0]
        except Exception:
            # If volume analysis fails, default to neutral
            volume_trend = 0
    
    # Higher score indicates stronger consolidation
    if price_range < 0.02 and volume_trend < 0:
        return 'high'
    elif price_range < 0.05 and volume_trend <= 0:
        return 'medium'
    else:
        return 'low'

def _calculate_long_breakout(current_price: float, 
                           resistance: float, 
                           atr: float,
                           volatility_factor: float,
                           min_risk_reward: float,
                           targets: List[float],
                           is_consolidating: bool,
                           consolidation_strength: str) -> Dict[str, Any]:
    """
    Calculate long breakout strategy parameters.
    
    Args:
        current_price: Current market price
        resistance: Nearest resistance level
        atr: Average True Range value
        volatility_factor: Factor for entry/stop calculation
        min_risk_reward: Minimum acceptable risk/reward ratio
        targets: List of potential price targets
        is_consolidating: Whether market is in consolidation
        consolidation_strength: Strength of the consolidation pattern
        
    Returns:
        Dictionary with long breakout strategy details
    """
    # Entry is above resistance by a buffer based on volatility
    entry_buffer = atr * volatility_factor
    entry_price = resistance + entry_buffer
    
    # Stop loss is below resistance by a buffer
    stop_buffer = atr * (volatility_factor * 2)  # Wider buffer for stop
    stop_price = resistance - stop_buffer
    
    # Risk calculation
    risk_amount = entry_price - stop_price
    risk_pct = (risk_amount / entry_price) * 100
    
    # Filter and sort targets (above entry)
    valid_targets = [t for t in targets if t > entry_price]
    valid_targets.sort()
    
    # If no valid targets, create targets based on risk multiples
    if not valid_targets:
        target1 = entry_price + (risk_amount * 1.5)  # 1.5:1 R:R
        target2 = entry_price + (risk_amount * 2.5)  # 2.5:1 R:R
        valid_targets = [target1, target2]
    
    # Calculate R:R for each target
    target_info = []
    for target in valid_targets[:3]:  # Limit to 3 targets
        reward_amount = target - entry_price
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        if rr_ratio >= min_risk_reward:
            target_info.append({
                'price': target,
                'r_r': rr_ratio,
                'pct_from_entry': (target - entry_price) / entry_price * 100
            })
    
    # Assign confidence level
    confidence = _assign_confidence_level(
        is_consolidating=is_consolidating,
        consolidation_strength=consolidation_strength,
        price_to_level_ratio=(current_price / resistance),
        target_info=target_info
    )
    
    return {
        'entry': entry_price,
        'entry_distance': (entry_price - current_price) / current_price * 100,  # % above current price
        'stop': stop_price,
        'stop_distance': (entry_price - stop_price) / entry_price * 100,  # % below entry
        'risk_pct': risk_pct,
        'targets': target_info,
        'confidence': confidence
    }

def _calculate_short_breakout(current_price: float, 
                            support: float, 
                            atr: float,
                            volatility_factor: float,
                            min_risk_reward: float,
                            targets: List[float],
                            is_consolidating: bool,
                            consolidation_strength: str) -> Dict[str, Any]:
    """
    Calculate short breakout strategy parameters.
    
    Args:
        current_price: Current market price
        support: Nearest support level
        atr: Average True Range value
        volatility_factor: Factor for entry/stop calculation
        min_risk_reward: Minimum acceptable risk/reward ratio
        targets: List of potential price targets
        is_consolidating: Whether market is in consolidation
        consolidation_strength: Strength of the consolidation pattern
        
    Returns:
        Dictionary with short breakout strategy details
    """
    # Entry is below support by a buffer based on volatility
    entry_buffer = atr * volatility_factor
    entry_price = support - entry_buffer
    
    # Stop loss is above support by a buffer
    stop_buffer = atr * (volatility_factor * 2)  # Wider buffer for stop
    stop_price = support + stop_buffer
    
    # Risk calculation
    risk_amount = stop_price - entry_price
    risk_pct = (risk_amount / entry_price) * 100
    
    # Filter and sort targets (below entry)
    valid_targets = [t for t in targets if t < entry_price]
    valid_targets.sort(reverse=True)  # Descending order for short targets
    
    # If no valid targets, create targets based on risk multiples
    if not valid_targets:
        target1 = entry_price - (risk_amount * 1.5)  # 1.5:1 R:R
        target2 = entry_price - (risk_amount * 2.5)  # 2.5:1 R:R
        valid_targets = [target1, target2]
    
    # Calculate R:R for each target
    target_info = []
    for target in valid_targets[:3]:  # Limit to 3 targets
        reward_amount = entry_price - target
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        if rr_ratio >= min_risk_reward:
            target_info.append({
                'price': target,
                'r_r': rr_ratio,
                'pct_from_entry': (entry_price - target) / entry_price * 100
            })
    
    # Assign confidence level
    confidence = _assign_confidence_level(
        is_consolidating=is_consolidating,
        consolidation_strength=consolidation_strength,
        price_to_level_ratio=(current_price / support),
        target_info=target_info
    )
    
    return {
        'entry': entry_price,
        'entry_distance': (current_price - entry_price) / current_price * 100,  # % below current price
        'stop': stop_price,
        'stop_distance': (stop_price - entry_price) / entry_price * 100,  # % above entry
        'risk_pct': risk_pct,
        'targets': target_info,
        'confidence': confidence
    }

def _assign_confidence_level(
    is_consolidating: bool,
    consolidation_strength: str,
    price_to_level_ratio: float,
    target_info: List[Dict[str, Any]]
) -> str:
    """
    Assign a confidence level to the breakout strategy.
    
    Args:
        is_consolidating: Whether the market is in consolidation
        consolidation_strength: Strength of consolidation
        price_to_level_ratio: Ratio of current price to support/resistance level
        target_info: Target price information
        
    Returns:
        String with confidence level ('low', 'medium', 'high')
    """
    # Base score starts at 0
    score = 0
    
    # Consolidation factors
    if is_consolidating:
        score += 2
        if consolidation_strength == 'high':
            score += 2
        elif consolidation_strength == 'medium':
            score += 1
    
    # Proximity to level
    if 0.95 <= price_to_level_ratio <= 1.05:
        score += 1  # Price is close to support/resistance
    
    # Risk/reward quality
    if target_info:
        best_rr = max([t.get('r_r', 0) for t in target_info])
        if best_rr >= 3.0:
            score += 2
        elif best_rr >= 2.0:
            score += 1
    
    # Assign confidence based on score
    if score >= 5:
        return 'high'
    elif score >= 3:
        return 'medium'
    else:
        return 'low' 