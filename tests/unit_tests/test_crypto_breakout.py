import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.services.breakout_strategy import CryptoBreakoutStrategy

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=30, freq='H')
    base_price = 50000  # Base price for BTC
    
    # Create a ranging market followed by a breakout
    prices = []
    for i in range(30):
        if i < 20:  # Ranging market
            price = base_price + np.sin(i/3) * 100  # Small oscillations
        else:  # Breakout
            price = base_price + (i-19) * 500  # Strong upward move
        prices.append(price)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p + 50 for p in prices],
        'low': [p - 50 for p in prices],
        'close': prices,
        'volume': [1000000] * 30,
        'ATR_14': [100.0] * 30,  # Fixed ATR for predictable tests
        'rsi_14': [50.0] * 30, # Added for _get_momentum_from_df
        'MACDh_12_26_9': [0.5] * 30 # Added for _get_momentum_from_df
    })
    
    return data.set_index('date')

def test_crypto_breakout_initialization():
    """Test CryptoBreakoutStrategy initialization."""
    strategy = CryptoBreakoutStrategy(
        volatility_factor=1.5,
        min_risk_reward=2.0,
        max_position_size=0.1,
        base_leverage=2.0
    )
    
    assert strategy.volatility_factor == 1.5
    assert strategy.min_risk_reward == 2.0
    assert strategy.max_position_size == 0.1
    assert strategy.base_leverage == 2.0

def test_breakout_level_calculation(sample_data):
    """Test breakout level calculation with multiple targets."""
    strategy = CryptoBreakoutStrategy()
    result = strategy.calculate_breakout_levels(sample_data)
    
    assert result['status'] == 'success'
    assert 'long_breakout' in result
    assert 'short_breakout' in result
    
    # Check long breakout structure
    long_breakout = result['long_breakout']
    assert 'entry' in long_breakout
    assert 'targets' in long_breakout
    assert 'initial_stop' in long_breakout
    assert 'breakeven_stop' in long_breakout
    assert 'trailing_stops' in long_breakout
    assert 'position_size' in long_breakout
    assert 'leverage' in long_breakout
    
    # Verify multiple targets
    assert len(long_breakout['targets']) == 3  # Should have 3 targets
    for target in long_breakout['targets']:
        assert 'price' in target
        assert 'pct_from_entry' in target
        assert 'atr_multiple' in target
        assert 'portion' in target
    
    # Verify trailing stops
    trailing_stops = long_breakout['trailing_stops']
    assert len(trailing_stops) > 0
    for stop in trailing_stops:
        assert 'price_trigger' in stop
        assert 'stop_level' in stop
        assert 'atr_multiple' in stop

def test_position_size_calculation(sample_data):
    """Test dynamic position size and leverage calculation."""
    strategy = CryptoBreakoutStrategy(
        max_position_size=0.1,  # 10% max position
        base_leverage=2.0
    )
    
    # Get current price and ATR
    current_price = sample_data['close'].iloc[-1]
    atr = sample_data['ATR_14'].iloc[-1]
    
    # Calculate momentum
    momentum = strategy._get_momentum_from_df(sample_data)
    assert momentum is not None
    
    # Test with a distant entry (should increase size)
    far_entry = current_price * 1.1  # 10% away
    far_position = strategy._calculate_position_size(
        current_price=current_price,
        entry=far_entry,
        atr=atr,
        direction='long',
        momentum=momentum
    )
    
    # Test with a close entry (should decrease size)
    near_entry = current_price * 1.01  # 1% away
    near_position = strategy._calculate_position_size(
        current_price=current_price,
        entry=near_entry,
        atr=atr,
        direction='long',
        momentum=momentum
    )
    
    # Further entry should have larger position size
    assert far_position['size'] > near_position['size']
    
    # Verify leverage caps
    assert far_position['leverage'] <= strategy.max_leverage
    assert near_position['leverage'] <= strategy.max_leverage

def test_profit_target_calculation():
    """Test multiple profit target calculation."""
    strategy = CryptoBreakoutStrategy()
    
    # Test long targets
    entry_price = 50000
    atr = 1000
    long_targets = strategy._calculate_profit_targets(entry_price, atr, 'long')
    
    assert len(long_targets) == 3  # Should have 3 targets
    assert long_targets[0]['price'] > entry_price  # First target should be above entry
    assert long_targets[1]['price'] > long_targets[0]['price']  # Second target higher than first
    assert long_targets[2]['price'] > long_targets[1]['price']  # Third target highest
    
    # Test short targets
    short_targets = strategy._calculate_profit_targets(entry_price, atr, 'short')
    
    assert len(short_targets) == 3
    assert short_targets[0]['price'] < entry_price  # First target should be below entry
    assert short_targets[1]['price'] < short_targets[0]['price']  # Second target lower than first
    assert short_targets[2]['price'] < short_targets[1]['price']  # Third target lowest

def test_atr_calculation(sample_data):
    """Test ATR calculation functionality."""
    strategy = CryptoBreakoutStrategy()
    
    # Test with ATR column present
    atr_from_column = strategy._get_atr_from_df(sample_data)
    assert atr_from_column == 100  # Should match our fixed ATR value
    
    # Test with ATR column removed
    data_no_atr = sample_data.drop('ATR_14', axis=1)
    atr_calculated_none = strategy._get_atr_from_df(data_no_atr)
    assert atr_calculated_none is None  # Should be None as ATR_14 is missing

def test_error_handling():
    """Test error handling for invalid inputs."""
    strategy = CryptoBreakoutStrategy()
    
    # Test with empty DataFrame
    empty_result = strategy.calculate_breakout_levels(pd.DataFrame())
    assert empty_result['status'] == 'error'
    assert 'Insufficient data' in empty_result['message']
    
    # Test with None
    none_result = strategy.calculate_breakout_levels(None)
    assert none_result['status'] == 'error'
    assert 'Insufficient data' in none_result['message']
    
    # Test with too little data
    small_data = pd.DataFrame({
        'close': [100, 101, 102],
        'high': [103, 104, 105],
        'low': [98, 99, 100],
        'volume': [1000, 1000, 1000]
    })
    small_result = strategy.calculate_breakout_levels(small_data)
    assert small_result['status'] == 'error'
    assert 'Insufficient data' in small_result['message']

def test_high_leverage_calculation(sample_data):
    """Test high leverage calculation with momentum."""
    strategy = CryptoBreakoutStrategy(
        base_leverage=10.0,
        max_leverage=30.0
    )
    
    # Test with strong momentum setup
    strong_momentum = {
        'rsi': 75.0,           # Strong bullish RSI
        'macd_hist': 50.0,     # Strong MACD momentum
        'volume_trend': 25.0   # Good volume increase
    }
    
    strong_position = strategy._calculate_position_size(
        current_price=50000,
        entry=55000,           # 10% above current price
        atr=1000,
        direction='long',
        momentum=strong_momentum
    )
    
    # Should use high leverage for strong setup
    assert strong_position['leverage'] > 20.0  # Should be using high leverage
    assert strong_position['leverage'] <= 30.0  # But not exceed max
    
    # Test with weak momentum setup
    weak_momentum = {
        'rsi': 55.0,           # Neutral RSI
        'macd_hist': 5.0,      # Weak MACD momentum
        'volume_trend': 5.0    # Small volume increase
    }
    
    weak_position = strategy._calculate_position_size(
        current_price=50000,
        entry=55000,
        atr=1000,
        direction='long',
        momentum=weak_momentum
    )
    
    # Should use lower leverage for weak setup
    assert weak_position['leverage'] < strong_position['leverage']
    assert weak_position['leverage'] >= 10.0  # Should still use base leverage

def test_dynamic_stop_losses(sample_data):
    """Test dynamic stop loss calculation."""
    strategy = CryptoBreakoutStrategy()
    
    # Calculate momentum
    momentum = strategy._get_momentum_from_df(sample_data)
    
    # Test long position stops
    long_stops = strategy._calculate_dynamic_stops(
        entry=50000,
        atr=1000,
        direction='long',
        momentum=momentum,
        leverage=20.0  # High leverage
    )
    
    # Verify stop structure
    assert 'initial' in long_stops
    assert 'breakeven' in long_stops
    assert 'trailing' in long_stops
    
    # Initial stop should be tight for high leverage
    initial_stop = long_stops['initial']
    stop_distance = (50000 - initial_stop) / 50000 * 100  # Distance as percentage
    assert stop_distance < 1.0  # Less than 1% for high leverage
    
    # Breakeven stop should be above entry
    assert long_stops['breakeven'] > 50000
    
    # Verify trailing stops
    trailing_stops = long_stops['trailing']
    assert len(trailing_stops) > 0
    
    # Stops should get wider as price moves up
    for i in range(len(trailing_stops) - 1):
        current_stop = trailing_stops[i]['stop_level']
        next_stop = trailing_stops[i + 1]['stop_level']
        assert next_stop > current_stop
    
    # Test short position stops
    short_stops = strategy._calculate_dynamic_stops(
        entry=50000,
        atr=1000,
        direction='short',
        momentum=momentum,
        leverage=20.0
    )
    
    # Verify short stop structure
    assert 'initial' in short_stops
    assert 'breakeven' in short_stops
    assert 'trailing' in short_stops
    
    # Initial stop should be tight for high leverage
    initial_stop = short_stops['initial']
    stop_distance = (initial_stop - 50000) / 50000 * 100
    assert stop_distance < 1.0
    
    # Breakeven stop should be below entry
    assert short_stops['breakeven'] < 50000
    
    # Verify trailing stops
    trailing_stops = short_stops['trailing']
    assert len(trailing_stops) > 0
    
    # Stops should get wider as price moves down
    for i in range(len(trailing_stops) - 1):
        current_stop = trailing_stops[i]['stop_level']
        next_stop = trailing_stops[i + 1]['stop_level']
        assert next_stop < current_stop

def test_momentum_scoring():
    """Test momentum score calculation."""
    strategy = CryptoBreakoutStrategy()
    
    # Test strong bullish momentum
    strong_bull = {
        'rsi': 75.0,
        'macd_hist': 50.0,
        'volume_trend': 25.0
    }
    bull_score = strategy._calculate_momentum_score(strong_bull, 'long')
    assert bull_score > 0.8  # Should be high score
    
    # Test strong bearish momentum
    strong_bear = {
        'rsi': 25.0,
        'macd_hist': -50.0,
        'volume_trend': 25.0
    }
    bear_score = strategy._calculate_momentum_score(strong_bear, 'short')
    assert bear_score > 0.8  # Should be high score
    
    # Test neutral momentum
    neutral = {
        'rsi': 50.0,
        'macd_hist': 0.0,
        'volume_trend': 0.0
    }
    neutral_score = strategy._calculate_momentum_score(neutral, 'long')
    assert neutral_score < 0.3  # Should be low score 