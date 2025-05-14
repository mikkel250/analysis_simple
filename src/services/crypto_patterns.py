"""
Crypto Pattern Analysis Module

This module analyzes crypto-specific patterns to identify high-probability breakout setups.
Key patterns include:
1. Accumulation patterns before explosive moves
2. Volume profile analysis
3. Whale wallet analysis
4. Historical volatility patterns
5. Multi-timeframe momentum alignment
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class PatternConfig:
    """Configuration for pattern detection."""
    min_consolidation_periods: int = 12  # Minimum periods for consolidation
    volume_surge_threshold: float = 1.5   # Volume surge multiplier (reduced from 2.0)
    momentum_threshold: float = 0.7       # Minimum momentum score
    volatility_compression: float = 0.5   # Maximum volatility during compression (increased from 0.3)
    price_compression: float = 0.03       # Maximum price range during compression (increased from 0.02)

class CryptoPatternAnalyzer:
    """Analyzes crypto-specific patterns for breakout opportunities."""
    
    def __init__(self, config: Optional[PatternConfig] = None):
        """
        Initialize the pattern analyzer.
        
        Args:
            config: Optional configuration for pattern detection
        """
        self.config = config or PatternConfig()
    
    def analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze price data for crypto-specific patterns.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Dictionary with pattern analysis results
        """
        if data is None or len(data) < self.config.min_consolidation_periods:
            return {
                'status': 'error',
                'message': 'Insufficient data for pattern analysis'
            }
        
        try:
            # Detect accumulation patterns
            accumulation = self._detect_accumulation(data)
            
            # Analyze volume profile
            volume_profile = self._analyze_volume_profile(data)
            
            # Check for volatility compression
            volatility = self._analyze_volatility(data)
            
            # Analyze momentum alignment
            momentum = self._analyze_momentum_alignment(data)
            
            # Calculate pattern probability score
            probability = self._calculate_probability(
                accumulation=accumulation,
                volume_profile=volume_profile,
                volatility=volatility,
                momentum=momentum
            )
            
            return {
                'status': 'success',
                'probability': probability,
                'patterns': {
                    'accumulation': accumulation,
                    'volume_profile': volume_profile,
                    'volatility': volatility,
                    'momentum': momentum
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _detect_accumulation(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect accumulation patterns before breakouts."""
        # Calculate price range over shorter window for tighter range detection
        window = min(self.config.min_consolidation_periods, 10)  # Even shorter window
        
        # Calculate normalized price range
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        price_range = (data['high'] - data['low']) / typical_price
        rolling_range = price_range.rolling(window=window).mean()
        
        # Calculate volume characteristics with shorter window
        volume = data['volume']
        avg_volume = volume.rolling(window=15).mean()  # Longer window for base volume
        volume_trend = (volume / avg_volume).rolling(window=5).mean()
        
        # Check for decreasing volatility with shorter window
        returns = typical_price.pct_change()
        volatility = returns.rolling(window=10).std()
        vol_trend = volatility.rolling(window=10).mean().diff()  # Smoother trend calculation
        
        # Detect accumulation characteristics
        is_tight_range = rolling_range.iloc[-1] < self.config.price_compression
        is_high_volume = volume_trend.iloc[-5:].mean() > 0.95  # Reduced threshold
        is_decreasing_volatility = vol_trend.iloc[-5:].mean() < -1e-6  # Small negative threshold
        
        # Calculate accumulation score
        score = 0.0
        if is_tight_range:
            score += 0.4
        if is_high_volume:
            score += 0.3
        if is_decreasing_volatility:
            score += 0.3
            
        return {
            'score': score,
            'is_accumulating': score > 0.6,
            'metrics': {
                'price_range': rolling_range.iloc[-1],
                'volume_trend': volume_trend.iloc[-5:].mean(),
                'volatility_trend': vol_trend.iloc[-5:].mean()
            }
        }
    
    def _analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume profile for breakout potential."""
        volume = data['volume']
        close = data['close']
        
        # Calculate volume-weighted metrics
        vwap = (close * volume).cumsum() / volume.cumsum()
        
        # Detect volume surges with more sensitive thresholds
        recent_volume = volume.iloc[-5:].mean()
        base_volume = volume.iloc[-20:-5].mean()  # Exclude recent volume
        recent_surge = recent_volume > (base_volume * 1.5)  # Use fixed threshold
        
        try:
            # Calculate price levels with high volume
            price_levels = pd.qcut(close, q=10)
            volume_at_price = volume.groupby(price_levels, observed=True).sum()
            
            # Find dominant volume levels
            high_vol_levels = volume_at_price[volume_at_price > volume_at_price.mean()]
            
            # Get the mid-point of each interval
            level_mids = [(level.left + level.right) / 2 for level in high_vol_levels.index]
            
            # Check for high volume near current price
            high_vol_near_price = any(abs(level - close.iloc[-1]) / close.iloc[-1] < 0.02 
                                    for level in level_mids)
        except Exception as e:
            logger.warning(f"Error in volume profile calculation: {e}")
            high_vol_near_price = False
            level_mids = []
        
        # Calculate final metrics
        above_vwap = close.iloc[-1] > vwap.iloc[-1]
        
        score = 0.0
        if recent_surge:
            score += 0.4
        if above_vwap:
            score += 0.3
        if high_vol_near_price:
            score += 0.3
            
        return {
            'score': score,
            'metrics': {
                'recent_surge': recent_surge,
                'above_vwap': above_vwap,
                'high_vol_levels': level_mids
            }
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility patterns for breakout potential."""
        close = data['close']
        
        # Calculate different volatility metrics with shorter windows
        returns = close.pct_change()
        volatility = returns.rolling(window=10).std()
        
        # Compare recent volatility to historical with more lenient threshold
        recent_vol = volatility.iloc[-5:].mean()
        historical_vol = volatility.iloc[-20:-5].mean()
        
        # Calculate compression ratio
        compression_ratio = recent_vol / historical_vol if historical_vol > 0 else 1.0
        is_compressed = compression_ratio < 0.8  # More lenient compression threshold
        
        # Calculate historical volatility percentile with minimum threshold
        min_vol = volatility.min()
        max_vol = volatility.max()
        if max_vol > min_vol:
            normalized_vol = (volatility.iloc[-1] - min_vol) / (max_vol - min_vol)
            vol_percentile = normalized_vol
        else:
            vol_percentile = 0.5  # Default to middle if no range
        
        # Detect volatility patterns with smoothing
        vol_trend = volatility.rolling(window=10).mean().diff()  # Smoother trend calculation
        recent_trend = vol_trend.iloc[-5:].mean()
        decreasing_vol = recent_trend < -1e-6  # Small negative threshold
        
        # Calculate volatility score with more granular conditions
        score = 0.0
        
        # Score compression
        if is_compressed:
            score += 0.4
        elif compression_ratio < 0.9:  # Partial score for near compression
            score += 0.2
        
        # Score percentile
        if vol_percentile < 0.3:
            score += 0.3
        elif vol_percentile < 0.4:  # Partial score for near threshold
            score += 0.15
        
        # Score trend
        if decreasing_vol:
            score += 0.3
        elif recent_trend < 0:  # Partial score for recent decrease
            score += 0.15
        
        # Add minimum score if volatility is relatively low
        if compression_ratio < 1.0:
            score = max(score, 0.3)
        
        return {
            'score': score,
            'metrics': {
                'current_volatility': volatility.iloc[-1],
                'is_compressed': is_compressed,
                'vol_percentile': vol_percentile,
                'vol_trend': 'decreasing' if decreasing_vol else 'increasing',
                'compression_ratio': compression_ratio
            }
        }
    
    def _analyze_momentum_alignment(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum alignment across indicators."""
        close = data['close']
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        # Calculate momentum score
        rsi_score = 0.0
        if rsi.iloc[-1] > 50 and rsi.iloc[-1] < 70:
            rsi_score = 0.4  # Strong but not overbought
            
        macd_score = 0.0
        if histogram.iloc[-1] > 0 and histogram.diff().iloc[-1] > 0:
            macd_score = 0.3  # Rising and positive
            
        trend_score = 0.0
        if close.iloc[-1] > close.rolling(window=20).mean().iloc[-1]:
            trend_score = 0.3  # Above moving average
            
        total_score = rsi_score + macd_score + trend_score
        
        return {
            'score': total_score,
            'metrics': {
                'rsi': rsi.iloc[-1],
                'macd_hist': histogram.iloc[-1],
                'trend': 'bullish' if trend_score > 0 else 'bearish'
            }
        }
    
    def _calculate_probability(
        self,
        accumulation: Dict[str, Any],
        volume_profile: Dict[str, Any],
        volatility: Dict[str, Any],
        momentum: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall probability of successful breakout."""
        # Weight the different components
        weights = {
            'accumulation': 0.3,
            'volume_profile': 0.25,
            'volatility': 0.25,
            'momentum': 0.2
        }
        
        # Calculate weighted score
        weighted_score = (
            accumulation['score'] * weights['accumulation'] +
            volume_profile['score'] * weights['volume_profile'] +
            volatility['score'] * weights['volatility'] +
            momentum['score'] * weights['momentum']
        )
        
        # Determine confidence level
        confidence = 'low'
        if weighted_score > 0.8:
            confidence = 'high'
        elif weighted_score > 0.6:
            confidence = 'medium'
            
        # Generate explanation
        explanation = []
        if accumulation['score'] > 0.6:
            explanation.append("Strong accumulation pattern detected")
        if volume_profile['score'] > 0.6:
            explanation.append("Supportive volume profile")
        if volatility['score'] > 0.6:
            explanation.append("Favorable volatility conditions")
        if momentum['score'] > 0.6:
            explanation.append("Strong momentum alignment")
            
        return {
            'score': weighted_score,
            'confidence': confidence,
            'explanation': explanation,
            'component_scores': {
                'accumulation': accumulation['score'],
                'volume_profile': volume_profile['score'],
                'volatility': volatility['score'],
                'momentum': momentum['score']
            }
        } 