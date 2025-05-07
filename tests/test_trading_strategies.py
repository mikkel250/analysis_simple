"""
Unit tests for the trading strategies module.

These tests verify the functionality of our advanced trading strategy functions including:
- Market condition detection
- Support/resistance level detection
- Price target calculation
- Risk assessment
- Trade recommendation generation
"""

import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from src.services.trading_strategies import (
    detect_market_condition,
    detect_support_resistance,
    calculate_price_targets,
    calculate_risk_metrics,
    generate_trade_recommendation,
    detect_regime,
    suggest_strategy_for_regime
)
from src.services.indicators import forecast_volatility  # To be implemented

# Fixtures for testing

@pytest.fixture
def sample_uptrend_data():
    """Create sample data for an uptrend market condition."""
    # Create a DataFrame with an uptrend pattern
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate a linear uptrend with some noise
    closes = np.linspace(100, 150, 100) + np.random.normal(0, 3, 100)
    
    # Create highs and lows around closes
    highs = closes + np.random.uniform(1, 5, 100)
    lows = closes - np.random.uniform(1, 5, 100)
    opens = closes - np.random.uniform(-3, 3, 100)
    
    # Create some volume data
    volumes = np.random.randint(1000, 10000, 100)
    
    # Create DataFrame
    data = {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some indicator data
    df['ADX_14'] = 30 + np.random.normal(0, 5, 100)  # Strong trend
    df['rsi_14'] = 60 + np.random.normal(0, 10, 100)  # Bullish
    
    # Add MACD
    df['MACD_12_26_9'] = 2 + np.random.normal(0, 0.5, 100)
    df['MACDs_12_26_9'] = 1 + np.random.normal(0, 0.5, 100)
    df['MACDh_12_26_9'] = 1 + np.random.normal(0, 0.3, 100)
    
    # Add Bollinger Bands
    df['BBU_20_2.0'] = closes + 10
    df['BBM_20_2.0'] = closes
    df['BBL_20_2.0'] = closes - 10
    
    # Add Stochastic
    df['STOCHk_14_3_3'] = 70 + np.random.normal(0, 10, 100)
    df['STOCHd_14_3_3'] = 65 + np.random.normal(0, 10, 100)
    
    # Add ATR
    df['ATR_14'] = 3 + np.random.normal(0, 0.5, 100)
    
    return df

@pytest.fixture
def sample_downtrend_data():
    """Create sample data for a downtrend market condition."""
    # Create a DataFrame with a downtrend pattern
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate a linear downtrend with some noise
    closes = np.linspace(150, 100, 100) + np.random.normal(0, 3, 100)
    
    # Create highs and lows around closes
    highs = closes + np.random.uniform(1, 5, 100)
    lows = closes - np.random.uniform(1, 5, 100)
    opens = closes - np.random.uniform(-3, 3, 100)
    
    # Create some volume data
    volumes = np.random.randint(1000, 10000, 100)
    
    # Create DataFrame
    data = {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some indicator data
    df['ADX_14'] = 30 + np.random.normal(0, 5, 100)  # Strong trend
    df['rsi_14'] = 40 + np.random.normal(0, 10, 100)  # Bearish
    
    # Add MACD
    df['MACD_12_26_9'] = -2 + np.random.normal(0, 0.5, 100)
    df['MACDs_12_26_9'] = -1 + np.random.normal(0, 0.5, 100)
    df['MACDh_12_26_9'] = -1 + np.random.normal(0, 0.3, 100)
    
    # Add Bollinger Bands
    df['BBU_20_2.0'] = closes + 10
    df['BBM_20_2.0'] = closes
    df['BBL_20_2.0'] = closes - 10
    
    # Add Stochastic
    df['STOCHk_14_3_3'] = 30 + np.random.normal(0, 10, 100)
    df['STOCHd_14_3_3'] = 35 + np.random.normal(0, 10, 100)
    
    # Add ATR
    df['ATR_14'] = 3 + np.random.normal(0, 0.5, 100)
    
    return df

@pytest.fixture
def sample_ranging_data():
    """Create sample data for a ranging market condition."""
    # Create a DataFrame with a ranging pattern
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Generate a ranging pattern with some noise
    base = np.ones(100) * 125
    closes = base + np.sin(np.linspace(0, 4*np.pi, 100)) * 10 + np.random.normal(0, 2, 100)
    
    # Create highs and lows around closes
    highs = closes + np.random.uniform(1, 5, 100)
    lows = closes - np.random.uniform(1, 5, 100)
    opens = closes - np.random.uniform(-3, 3, 100)
    
    # Create some volume data
    volumes = np.random.randint(1000, 10000, 100)
    
    # Create DataFrame
    data = {
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }
    
    df = pd.DataFrame(data, index=dates)
    
    # Add some indicator data
    df['ADX_14'] = 15 + np.random.normal(0, 3, 100)  # Weak trend
    df['rsi_14'] = 50 + np.random.normal(0, 15, 100)  # Neutral
    
    # Add MACD
    df['MACD_12_26_9'] = np.random.normal(0, 0.5, 100)
    df['MACDs_12_26_9'] = np.random.normal(0, 0.5, 100)
    df['MACDh_12_26_9'] = np.random.normal(0, 0.3, 100)
    
    # Add Bollinger Bands
    df['BBU_20_2.0'] = closes + 10
    df['BBM_20_2.0'] = closes
    df['BBL_20_2.0'] = closes - 10
    
    # Add Stochastic
    df['STOCHk_14_3_3'] = 50 + np.random.normal(0, 20, 100)
    df['STOCHd_14_3_3'] = 50 + np.random.normal(0, 20, 100)
    
    # Add ATR
    df['ATR_14'] = 2 + np.random.normal(0, 0.3, 100)
    
    return df

# Test cases

def test_detect_market_condition_uptrend(sample_uptrend_data):
    """Test market condition detection for uptrend."""
    result = detect_market_condition(sample_uptrend_data)
    
    assert result['condition'] == 'trending'
    assert 'uptrend' in result['sub_condition']
    assert result['confidence'] in ['medium', 'high']
    assert 'metrics' in result
    assert 'trend_strength' in result['metrics']
    assert 'normalized_slope' in result['metrics']
    assert result['metrics']['normalized_slope'] > 0

def test_detect_market_condition_downtrend(sample_downtrend_data):
    """Test market condition detection for downtrend."""
    result = detect_market_condition(sample_downtrend_data)
    
    assert result['condition'] == 'trending'
    assert 'downtrend' in result['sub_condition']
    assert result['confidence'] in ['medium', 'high']
    assert 'metrics' in result
    assert 'trend_strength' in result['metrics']
    assert 'normalized_slope' in result['metrics']
    assert result['metrics']['normalized_slope'] < 0

def test_detect_market_condition_ranging(sample_ranging_data):
    """Test market condition detection for ranging market."""
    result = detect_market_condition(sample_ranging_data)
    
    assert result['condition'] in ['ranging', 'volatile']
    if result['condition'] == 'ranging':
        assert result['sub_condition'] in ['sideways', 'tight_range', 'wide_range']
    else:
        assert result['sub_condition'] in ['choppy', 'volatile_uptrend', 'volatile_downtrend']
    assert 'metrics' in result
    assert 'trend_strength' in result['metrics']
    assert 'normalized_slope' in result['metrics']

def test_detect_support_resistance_uptrend(sample_uptrend_data):
    """Test support and resistance detection for uptrend."""
    result = detect_support_resistance(sample_uptrend_data)
    
    assert 'supports' in result
    assert 'resistances' in result
    assert 'key_level' in result
    assert 'confidence' in result
    
    # In an uptrend, we expect at least some support levels
    if len(result['supports']) > 0:
        for level in result['supports']:
            assert level < sample_uptrend_data['close'].iloc[-1]

def test_detect_support_resistance_downtrend(sample_downtrend_data):
    """Test support and resistance detection for downtrend."""
    result = detect_support_resistance(sample_downtrend_data)
    
    assert 'supports' in result
    assert 'resistances' in result
    assert 'confidence' in result
    
    # In a downtrend, we expect at least some resistance levels
    if len(result['resistances']) > 0:
        for level in result['resistances']:
            assert level > sample_downtrend_data['close'].iloc[-1]

def test_calculate_price_targets_uptrend(sample_uptrend_data):
    """Test price target calculation for uptrend."""
    sr_levels = detect_support_resistance(sample_uptrend_data)
    result = calculate_price_targets(sample_uptrend_data, sr_levels)
    
    assert 'next_target' in result
    assert 'stop_loss' in result
    assert 'targets' in result
    assert 'target_methods' in result
    assert 'trend_direction' in result
    
    assert result['trend_direction'] == 'up'
    
    # In an uptrend, targets should be higher than current price
    if result['next_target'] is not None:
        assert result['next_target'] > sample_uptrend_data['close'].iloc[-1]

def test_calculate_price_targets_downtrend(sample_downtrend_data):
    """Test price target calculation for downtrend."""
    sr_levels = detect_support_resistance(sample_downtrend_data)
    result = calculate_price_targets(sample_downtrend_data, sr_levels)
    
    assert 'next_target' in result
    assert 'stop_loss' in result
    assert 'targets' in result
    assert 'target_methods' in result
    assert 'trend_direction' in result
    
    assert result['trend_direction'] == 'down'
    
    # In a downtrend, targets should be lower than current price
    if result['next_target'] is not None:
        assert result['next_target'] < sample_downtrend_data['close'].iloc[-1]

def test_calculate_risk_metrics_valid_targets(sample_uptrend_data):
    """Test risk metrics calculation with valid price targets."""
    sr_levels = detect_support_resistance(sample_uptrend_data)
    price_targets = calculate_price_targets(sample_uptrend_data, sr_levels)
    
    # Only run this test if we have valid targets
    if price_targets['next_target'] is not None and price_targets['stop_loss'] is not None:
        result = calculate_risk_metrics(sample_uptrend_data, price_targets)
        
        assert 'risk_reward_ratio' in result
        assert 'position_size' in result
        assert 'max_loss_pct' in result
        
        # Risk reward should be positive for a valid trade
        assert result['risk_reward_ratio'] > 0
        
        # Position size should be expressed as a percentage
        assert 0 <= result['position_size'] <= 100
        
        # Max loss should be capped at 1%
        assert 0 <= result['max_loss_pct'] <= 1.0

def test_generate_trade_recommendation_uptrend(sample_uptrend_data):
    """Test trade recommendation generation for uptrend."""
    result = generate_trade_recommendation(sample_uptrend_data)
    
    assert 'market_condition' in result
    assert 'strategy' in result
    assert 'action' in result
    assert 'entry_points' in result
    assert 'exit_points' in result
    assert 'risk_assessment' in result
    assert 'supportive_indicators' in result
    assert 'contrary_indicators' in result
    assert 'confidence' in result
    
    # In an uptrend, we expect a buy action with confidence
    if result['market_condition']['condition'] == 'trending' and 'uptrend' in result['market_condition']['sub_condition']:
        assert result['action'] == 'buy'
        assert result['confidence'] in ['medium', 'high']

def test_generate_trade_recommendation_downtrend(sample_downtrend_data):
    """Test trade recommendation generation for downtrend."""
    result = generate_trade_recommendation(sample_downtrend_data)
    
    assert 'market_condition' in result
    assert 'strategy' in result
    assert 'action' in result
    assert 'entry_points' in result
    assert 'exit_points' in result
    assert 'risk_assessment' in result
    assert 'supportive_indicators' in result
    assert 'contrary_indicators' in result
    assert 'confidence' in result
    
    # In a downtrend, we expect a sell action with confidence
    if result['market_condition']['condition'] == 'trending' and 'downtrend' in result['market_condition']['sub_condition']:
        assert result['action'] == 'sell'
        assert result['confidence'] in ['medium', 'high']

def test_generate_trade_recommendation_ranging(sample_ranging_data):
    """Test trade recommendation generation for ranging market."""
    result = generate_trade_recommendation(sample_ranging_data)
    
    assert 'market_condition' in result
    assert 'strategy' in result
    assert 'action' in result
    assert 'entry_points' in result
    assert 'exit_points' in result
    assert 'risk_assessment' in result
    assert 'supportive_indicators' in result
    assert 'contrary_indicators' in result
    assert 'confidence' in result
    
    # In a ranging market, the action could be buy, sell, or hold depending on where the price is
    if result['market_condition']['condition'] == 'ranging':
        assert result['action'] in ['buy', 'sell', 'hold']
        # Ranging markets typically have lower confidence
        assert result['confidence'] in ['low', 'medium']

def test_empty_data_handling():
    """Test how functions handle empty data."""
    empty_df = pd.DataFrame()
    
    # All functions should gracefully handle empty data
    market_condition = detect_market_condition(empty_df)
    assert market_condition['condition'] == 'unknown'
    assert market_condition['sub_condition'] == 'insufficient_data'
    
    sr_levels = detect_support_resistance(empty_df)
    assert sr_levels['supports'] == []
    assert sr_levels['resistances'] == []
    
    price_targets = calculate_price_targets(empty_df, sr_levels)
    assert price_targets['next_target'] is None
    assert price_targets['stop_loss'] is None
    
    risk_metrics = calculate_risk_metrics(empty_df, price_targets)
    assert risk_metrics['risk_reward_ratio'] is None
    
    recommendation = generate_trade_recommendation(empty_df)
    assert recommendation['strategy'] == 'insufficient_data'
    assert recommendation['action'] == 'hold'
    assert recommendation['confidence'] == 'low'

def test_none_data_handling():
    """Test how functions handle None data."""
    # All functions should gracefully handle None data
    market_condition = detect_market_condition(None)
    assert market_condition['condition'] == 'unknown'
    assert market_condition['sub_condition'] == 'insufficient_data'
    
    sr_levels = detect_support_resistance(None)
    assert sr_levels['supports'] == []
    assert sr_levels['resistances'] == []
    
    price_targets = calculate_price_targets(None, sr_levels)
    assert price_targets['next_target'] is None
    assert price_targets['stop_loss'] is None
    
    risk_metrics = calculate_risk_metrics(None, price_targets)
    assert risk_metrics['risk_reward_ratio'] is None
    
    recommendation = generate_trade_recommendation(None)
    assert recommendation['strategy'] == 'insufficient_data'
    assert recommendation['action'] == 'hold'
    assert recommendation['confidence'] == 'low'

def test_stop_loss_buffer_and_position_size_uptrend(sample_uptrend_data):
    """Test stop loss buffer and position size logic for uptrend."""
    sr_levels = detect_support_resistance(sample_uptrend_data)
    price_targets = calculate_price_targets(sample_uptrend_data, sr_levels)
    # Stop loss should not be equal to entry/support
    if sr_levels['supports']:
        support = float(sr_levels['supports'][0])
        assert price_targets['stop_loss'] != support
        # Stop loss should be below support
        assert price_targets['stop_loss'] < support
    # Position size should be capped and confidence-scaled
    price_targets['confidence'] = 'high'
    risk_metrics = calculate_risk_metrics(sample_uptrend_data, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10
    price_targets['confidence'] = 'medium'
    risk_metrics = calculate_risk_metrics(sample_uptrend_data, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10
    price_targets['confidence'] = 'low'
    risk_metrics = calculate_risk_metrics(sample_uptrend_data, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10


def test_stop_loss_buffer_and_position_size_downtrend(sample_downtrend_data):
    """Test stop loss buffer and position size logic for downtrend."""
    sr_levels = detect_support_resistance(sample_downtrend_data)
    price_targets = calculate_price_targets(sample_downtrend_data, sr_levels)
    # Stop loss should not be equal to entry/resistance
    if sr_levels['resistances']:
        resistance = float(sr_levels['resistances'][0])
        assert price_targets['stop_loss'] != resistance
        # Stop loss should be above resistance
        assert price_targets['stop_loss'] > resistance
    # Position size should be capped and confidence-scaled
    price_targets['confidence'] = 'high'
    risk_metrics = calculate_risk_metrics(sample_downtrend_data, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10
    price_targets['confidence'] = 'medium'
    risk_metrics = calculate_risk_metrics(sample_downtrend_data, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10
    price_targets['confidence'] = 'low'
    risk_metrics = calculate_risk_metrics(sample_downtrend_data, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10


def test_stop_loss_and_position_size_no_support_resistance():
    """Test logic when no support/resistance is available (edge case)."""
    # Create a DataFrame with 20 rows and no clear support/resistance
    n = 20
    df = pd.DataFrame({
        'open': np.linspace(100, 119, n),
        'high': np.linspace(101, 120, n),
        'low': np.linspace(99, 118, n),
        'close': np.linspace(100, 119, n),
        'volume': [1000] * n,
        'ATR_14': [1] * n
    })
    sr_levels = {'supports': [], 'resistances': [], 'key_level': None, 'confidence': 'low'}
    price_targets = calculate_price_targets(df, sr_levels)
    # Stop loss should still be set (ATR fallback)
    assert price_targets['stop_loss'] is not None
    # Position size should be capped
    price_targets['confidence'] = 'medium'
    risk_metrics = calculate_risk_metrics(df, price_targets)
    assert 0 <= risk_metrics['position_size'] <= 10

@pytest.mark.parametrize("horizon", ["24h", "4h", "1h"])
def test_volatility_forecast_valid(sample_uptrend_data, horizon):
    """Test volatility forecast output for valid data and all horizons."""
    result = forecast_volatility(sample_uptrend_data, horizon=horizon)
    assert isinstance(result, dict)
    assert result["horizon"] == horizon
    assert isinstance(result["forecast"], float)
    assert 0 <= result["forecast"] < 100  # Volatility as percent, reasonable range
    assert result["confidence"] in ["high", "medium", "low"]


def test_volatility_forecast_insufficient_data():
    """Test volatility forecast with insufficient data."""
    df = pd.DataFrame({"close": [100]})
    result = forecast_volatility(df, horizon="24h")
    assert result["forecast"] is None or np.isnan(result["forecast"])
    assert result["confidence"] == "low"


def test_volatility_forecast_constant_price():
    """Test volatility forecast with constant price data."""
    df = pd.DataFrame({"close": [100]*50, "high": [100]*50, "low": [100]*50})
    result = forecast_volatility(df, horizon="24h")
    assert result["forecast"] == 0
    assert result["confidence"] in ["low", "medium"]

def test_regime_detection_trending_high_volatility():
    """Test regime detection for trending, high volatility market."""
    n = 100
    # Strong uptrend, high volatility
    closes = np.linspace(100, 200, n) + np.random.normal(0, 10, n)
    df = pd.DataFrame({
        'close': closes,
        'high': closes + np.random.uniform(1, 5, n),
        'low': closes - np.random.uniform(1, 5, n),
        'ADX_14': [35]*n,  # Strong trend
        'BBU_20_2.0': closes + 10,
        'BBL_20_2.0': closes - 10
    })
    result = detect_regime(df)
    assert isinstance(result, dict)
    assert result['trend_regime'] == 'trending'
    assert result['volatility_regime'] == 'high_volatility'
    assert result['confidence'] in ['high', 'medium']
    assert 'metrics' in result


def test_regime_detection_ranging_low_volatility():
    """Test regime detection for range-bound, low volatility market."""
    n = 100
    closes = np.ones(n) * 100 + np.sin(np.linspace(0, 4*np.pi, n)) * 2  # Low volatility
    df = pd.DataFrame({
        'close': closes,
        'high': closes + 1,
        'low': closes - 1,
        'ADX_14': [10]*n,  # Weak trend
        'BBU_20_2.0': closes + 2,
        'BBL_20_2.0': closes - 2
    })
    result = detect_regime(df)
    assert isinstance(result, dict)
    assert result['trend_regime'] == 'range_bound'
    assert result['volatility_regime'] == 'low_volatility'
    assert result['confidence'] in ['high', 'medium']
    assert 'metrics' in result


def test_regime_detection_ambiguous():
    """Test regime detection for ambiguous/unclear market."""
    n = 30
    closes = np.random.normal(100, 1, n)
    df = pd.DataFrame({
        'close': closes,
        'high': closes + 1,
        'low': closes - 1,
        'ADX_14': [15]*n,
        'BBU_20_2.0': closes + 2,
        'BBL_20_2.0': closes - 2
    })
    result = detect_regime(df)
    assert isinstance(result, dict)
    assert result['trend_regime'] in ['ambiguous', 'range_bound', 'trending']
    assert result['volatility_regime'] in ['ambiguous', 'low_volatility', 'high_volatility']
    assert result['confidence'] in ['low', 'medium']
    assert 'metrics' in result


def test_regime_detection_insufficient_data():
    """Test regime detection with insufficient data."""
    df = pd.DataFrame({'close': [100]})
    result = detect_regime(df)
    assert result['trend_regime'] == 'ambiguous'
    assert result['volatility_regime'] == 'ambiguous'
    assert result['confidence'] == 'low'
    assert 'metrics' in result

def test_strategy_suggestion_trending_high_volatility():
    """Test strategy suggestion for trending, high volatility regime."""
    regime = {
        'trend_regime': 'trending',
        'volatility_regime': 'high_volatility',
        'confidence': 'high',
        'metrics': {}
    }
    result = suggest_strategy_for_regime(regime)
    assert isinstance(result, dict)
    assert result['strategy'] == 'trend_following'
    assert 'educational_rationale' in result
    assert 'actionable_advice' in result
    assert 'leverage' in result['actionable_advice']


def test_strategy_suggestion_range_bound_low_volatility():
    """Test strategy suggestion for range-bound, low volatility regime."""
    regime = {
        'trend_regime': 'range_bound',
        'volatility_regime': 'low_volatility',
        'confidence': 'high',
        'metrics': {}
    }
    result = suggest_strategy_for_regime(regime)
    assert isinstance(result, dict)
    assert result['strategy'] == 'range_trading'
    assert 'educational_rationale' in result
    assert 'actionable_advice' in result
    assert 'leverage' in result['actionable_advice']


def test_strategy_suggestion_ambiguous():
    """Test strategy suggestion for ambiguous regime."""
    regime = {
        'trend_regime': 'ambiguous',
        'volatility_regime': 'ambiguous',
        'confidence': 'low',
        'metrics': {}
    }
    result = suggest_strategy_for_regime(regime)
    assert isinstance(result, dict)
    assert result['strategy'] == 'reduce_exposure'
    assert 'educational_rationale' in result
    assert 'actionable_advice' in result


def test_strategy_suggestion_insufficient_data():
    """Test strategy suggestion with insufficient data."""
    regime = None
    result = suggest_strategy_for_regime(regime)
    assert isinstance(result, dict)
    assert result['strategy'] == 'insufficient_data'
    assert 'educational_rationale' in result
    assert 'actionable_advice' in result 