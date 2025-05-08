import pytest
import json
import time
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.services.open_interest import (
    analyze_open_interest,
    _calculate_advanced_metrics,
    _detect_divergence,
    _classify_market_regime
)

# Sample test data with different market scenarios
@pytest.fixture
def bullish_trend_data():
    """Sample data for a bullish trend scenario - increasing OI and price"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Generate data points with increasing OI and price over time
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 1.05,
            "price": base_price * 1.02,
            "volume": base_volume * 1.1
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 1.12,
            "price": base_price * 1.04,
            "volume": base_volume * 1.2
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 1.18,
            "price": base_price * 1.07,
            "volume": base_volume * 1.3
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 1.25,
            "price": base_price * 1.10,
            "volume": base_volume * 1.4
        }
    ]

@pytest.fixture
def bearish_trend_data():
    """Sample data for a bearish trend scenario - increasing OI, decreasing price"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Generate data points with increasing OI but decreasing price over time
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 1.03,
            "price": base_price * 0.98,
            "volume": base_volume * 1.1
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 1.07,
            "price": base_price * 0.95,
            "volume": base_volume * 1.2
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 1.12,
            "price": base_price * 0.92,
            "volume": base_volume * 1.3
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 1.18,
            "price": base_price * 0.88,
            "volume": base_volume * 1.4
        }
    ]

@pytest.fixture
def potential_reversal_data():
    """Sample data for a potential reversal scenario - decreasing OI, increasing price"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Generate data points with decreasing OI but increasing price over time
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 0.98,
            "price": base_price * 1.01,
            "volume": base_volume * 0.95
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 0.95,
            "price": base_price * 1.02,
            "volume": base_volume * 0.92
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 0.91,
            "price": base_price * 1.04,
            "volume": base_volume * 0.90
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 0.85,
            "price": base_price * 1.06,
            "volume": base_volume * 0.88
        }
    ]

@pytest.fixture
def liquidation_data():
    """Sample data for a liquidation scenario - decreasing OI and price"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Generate data points with decreasing OI and price over time
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 0.95,
            "price": base_price * 0.97,
            "volume": base_volume * 1.2
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 0.86,
            "price": base_price * 0.94,
            "volume": base_volume * 1.5
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 0.75,
            "price": base_price * 0.90,
            "volume": base_volume * 1.8
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 0.60,
            "price": base_price * 0.85,
            "volume": base_volume * 2.0
        }
    ]

@pytest.fixture
def market_spike_data():
    """Sample data for a market spike scenario - large OI increase with price increase"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Generate data points with a large spike in OI and price
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 1.05,
            "price": base_price * 1.02,
            "volume": base_volume * 1.1
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 1.12,
            "price": base_price * 1.05,
            "volume": base_volume * 1.3
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 1.25,
            "price": base_price * 1.10,
            "volume": base_volume * 1.6
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 1.50,  # 50% increase overall
            "price": base_price * 1.20,      # 20% price increase
            "volume": base_volume * 2.5
        }
    ]

@pytest.fixture
def divergence_bearish_data():
    """Sample data for bearish divergence - price up, OI down"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Price increases while OI decreases - potential bearish reversal
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 0.98,
            "price": base_price * 1.01,
            "volume": base_volume * 0.9
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 0.96,
            "price": base_price * 1.02,
            "volume": base_volume * 0.85
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 0.94,
            "price": base_price * 1.04,
            "volume": base_volume * 0.8
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 0.92,
            "price": base_price * 1.06,
            "volume": base_volume * 0.75
        }
    ]

@pytest.fixture
def neutral_market_data():
    """Sample data for a neutral market - minimal changes in OI and price"""
    base_time = int(datetime.now().timestamp() * 1000)
    base_oi = 100000
    base_price = 50000
    base_volume = 5000
    
    # Minimal changes in OI and price - sideways market
    return [
        {
            "timestamp": base_time - (24 * 3600 * 1000),  # 24 hours ago
            "open_interest": base_oi,
            "price": base_price,
            "volume": base_volume
        },
        {
            "timestamp": base_time - (18 * 3600 * 1000),  # 18 hours ago
            "open_interest": base_oi * 1.01,
            "price": base_price * 0.99,
            "volume": base_volume * 0.95
        },
        {
            "timestamp": base_time - (12 * 3600 * 1000),  # 12 hours ago
            "open_interest": base_oi * 0.99,
            "price": base_price * 1.01,
            "volume": base_volume * 0.98
        },
        {
            "timestamp": base_time - (6 * 3600 * 1000),   # 6 hours ago
            "open_interest": base_oi * 1.02,
            "price": base_price * 0.995,
            "volume": base_volume * 0.97
        },
        {
            "timestamp": base_time,                      # Now
            "open_interest": base_oi * 1.01,
            "price": base_price * 1.005,
            "volume": base_volume * 0.99
        }
    ]

def test_analyze_open_interest_bullish_trend(bullish_trend_data):
    """Test analyze_open_interest with bullish trend data"""
    # Print data for debugging
    print("\nBULLISH TREND DATA:")
    for point in bullish_trend_data:
        print(f"Timestamp: {point['timestamp']}, OI: {point['open_interest']}, Price: {point['price']}")
    
    # Calculate changes for debugging
    first_point = bullish_trend_data[0]
    last_point = bullish_trend_data[-1]
    oi_change = last_point['open_interest'] - first_point['open_interest']
    price_change = last_point['price'] - first_point['price']
    oi_change_pct = (oi_change / first_point['open_interest']) * 100
    price_change_pct = (price_change / first_point['price']) * 100
    print(f"OI Change: {oi_change}, Price Change: {price_change}")
    print(f"OI Change %: {oi_change_pct:.2f}%, Price Change %: {price_change_pct:.2f}%")
    
    result = analyze_open_interest(bullish_trend_data)
    print(f"Result regime: {result['regime']}")
    print(f"Regime conditions: {result.get('details', '')}")
    
    # Verify the function correctly identifies a bullish regime
    assert result["regime"] == "bullish_trend"
    assert "confidence" in result
    assert "trading_signals" in result
    assert "metrics" in result
    
    # Verify trading signals for bullish market
    signals = result["trading_signals"]
    assert signals["signal"] in ["bullish", "strong_bullish"]
    assert signals["action"] == "buy"
    assert signals["entry"] is not None
    assert signals["stop_loss"] is not None
    assert signals["take_profit"] is not None

def test_analyze_open_interest_bearish_trend(bearish_trend_data):
    """Test analyze_open_interest with bearish trend data"""
    result = analyze_open_interest(bearish_trend_data)
    
    # Verify the function correctly identifies a bearish regime
    assert result["regime"] == "bearish_trend"
    assert "confidence" in result
    assert "trading_signals" in result
    assert "metrics" in result
    
    # Verify trading signals for bearish market
    signals = result["trading_signals"]
    assert signals["signal"] == "bearish"
    assert signals["action"] == "sell"
    assert signals["entry"] is not None
    assert signals["stop_loss"] is not None
    assert signals["take_profit"] is not None

def test_analyze_open_interest_potential_reversal(potential_reversal_data):
    """Test analyze_open_interest with potential reversal data"""
    result = analyze_open_interest(potential_reversal_data)
    
    # Verify the function correctly identifies a potential reversal
    assert result["regime"] == "potential_reversal"
    assert "confidence" in result
    assert "trading_signals" in result
    assert "metrics" in result
    
    # Verify trading signals for reversal market
    signals = result["trading_signals"]
    assert signals["signal"] == "short_term_bullish"
    assert signals["action"] == "short_term_buy"
    assert signals["entry"] is not None
    assert signals["stop_loss"] is not None
    assert signals["take_profit"] is not None

def test_analyze_open_interest_liquidation(liquidation_data):
    """Test analyze_open_interest with liquidation data"""
    # Print data for debugging
    print("\nLIQUIDATION DATA:")
    for point in liquidation_data:
        print(f"Timestamp: {point['timestamp']}, OI: {point['open_interest']}, Price: {point['price']}")
    
    # Calculate changes for debugging
    first_point = liquidation_data[0]
    last_point = liquidation_data[-1]
    oi_change = last_point['open_interest'] - first_point['open_interest']
    price_change = last_point['price'] - first_point['price']
    oi_change_pct = (oi_change / first_point['open_interest']) * 100
    price_change_pct = (price_change / first_point['price']) * 100
    print(f"OI Change: {oi_change}, Price Change: {price_change}")
    print(f"OI Change %: {oi_change_pct:.2f}%, Price Change %: {price_change_pct:.2f}%")
    
    result = analyze_open_interest(liquidation_data)
    print(f"Result regime: {result['regime']}")
    print(f"Regime conditions: {result.get('details', '')}")
    
    # Check for expected liquidation regime identification
    assert result["regime"] in ["liquidation_or_bottoming", "potential_bottoming"]
    assert "confidence" in result
    assert "trading_signals" in result
    assert "metrics" in result
    
    # For large decreases, might signal a potential bottom
    signals = result["trading_signals"]
    if result["regime"] == "potential_bottoming":
        assert signals["action"] in ["prepare_to_buy", "wait"]
    else:
        assert signals["action"] == "wait"

def test_analyze_open_interest_market_spike(market_spike_data):
    """Test analyze_open_interest with market spike data"""
    result = analyze_open_interest(market_spike_data)
    
    # Verify spike detection
    assert result["regime"] == "spike_or_breakout"
    assert result["confidence"] == "high"
    assert "trading_signals" in result
    assert "metrics" in result
    
    # Verify trading signals for spike market
    signals = result["trading_signals"]
    assert signals["signal"] in ["bullish_breakout", "cautious_bearish"]
    
    # Ensure large OI change is detected in metrics
    assert result["change_24h"] > 20  # Should be around 50%

def test_analyze_open_interest_divergence(divergence_bearish_data):
    """Test analyze_open_interest with divergence data"""
    result = analyze_open_interest(divergence_bearish_data)
    
    # Check divergence detection
    assert "divergence" in result
    assert result["divergence"]["detected"] == True
    assert result["divergence"]["type"] == "bearish"
    
    # Verify the resulting signals account for the divergence
    signals = result["trading_signals"]
    # The signal should be cautious due to the divergence
    assert "cautious" in signals["signal"] or signals["action"] in ["reduce_longs", "wait"]

def test_analyze_open_interest_neutral(neutral_market_data):
    """Test analyze_open_interest with neutral market data"""
    # Print data for debugging
    print("\nNEUTRAL MARKET DATA:")
    for point in neutral_market_data:
        print(f"Timestamp: {point['timestamp']}, OI: {point['open_interest']}, Price: {point['price']}")
    
    # Calculate changes for debugging
    first_point = neutral_market_data[0]
    last_point = neutral_market_data[-1]
    oi_change = last_point['open_interest'] - first_point['open_interest']
    price_change = last_point['price'] - first_point['price']
    oi_change_pct = (oi_change / first_point['open_interest']) * 100
    price_change_pct = (price_change / first_point['price']) * 100
    print(f"OI Change: {oi_change}, Price Change: {price_change}")
    print(f"OI Change %: {oi_change_pct:.2f}%, Price Change %: {price_change_pct:.2f}%")
    
    result = analyze_open_interest(neutral_market_data)
    print(f"Result regime: {result['regime']}")
    print(f"Regime conditions: {result.get('details', '')}")
    
    # Verify neutral regime identification
    assert result["regime"] == "neutral"
    assert "confidence" in result
    assert "trading_signals" in result
    assert "metrics" in result
    
    # Verify neutral signals
    signals = result["trading_signals"]
    assert signals["signal"] == "neutral"
    assert signals["action"] == "wait"
    assert signals["entry"] is None
    assert signals["stop_loss"] is None
    assert signals["take_profit"] is None

def test_insufficient_data():
    """Test behavior with insufficient data"""
    result = analyze_open_interest([])  # Empty list
    
    assert result["regime"] == "insufficient_data"
    assert result["confidence"] == "low"
    assert "metrics" in result
    assert result["metrics"] == {}
    
    # Single data point
    single_data = [{"timestamp": 1635350400000, "open_interest": 100000, "price": 60000, "volume": 1000}]
    result = analyze_open_interest(single_data)
    
    assert result["regime"] == "insufficient_data"
    assert result["confidence"] == "low"

def test_advanced_metrics_calculation(bullish_trend_data):
    """Test the calculation of advanced metrics"""
    oi_values = [entry["open_interest"] for entry in bullish_trend_data]
    prices = [entry["price"] for entry in bullish_trend_data]
    volumes = [entry["volume"] for entry in bullish_trend_data]
    
    metrics = _calculate_advanced_metrics(bullish_trend_data, oi_values, prices, volumes)
    
    # Verify metrics structure
    assert "oi_volume_ratio" in metrics
    assert "oi_volume_ratio_status" in metrics
    assert "oi_roc_short_term" in metrics
    assert "oi_roc_medium_term" in metrics
    assert "oi_acceleration" in metrics
    assert "oi_momentum" in metrics
    
    # Since bullish_trend_data has increasing OI
    assert metrics["oi_roc_short_term"] > 0
    assert metrics["oi_roc_medium_term"] > 0

def test_divergence_detection(divergence_bearish_data):
    """Test the divergence detection algorithm"""
    oi_values = [entry["open_interest"] for entry in divergence_bearish_data]
    prices = [entry["price"] for entry in divergence_bearish_data]
    
    divergence = _detect_divergence(divergence_bearish_data, oi_values, prices)
    
    # Verify divergence structure
    assert "detected" in divergence
    assert "type" in divergence
    assert "strength" in divergence
    assert "correlation" in divergence
    
    # For bearish divergence data, we should detect it
    assert divergence["detected"] == True
    assert divergence["type"] == "bearish"
    assert divergence["strength"] > 0
    assert divergence["correlation"] < 0  # Negative correlation for divergence

def test_market_regime_classification():
    """Test the market regime classification logic"""
    # Test bullish market conditions
    regime, confidence, summary, signals = _classify_market_regime(
        oi_change=10000,
        price_change=5000,
        oi_change_pct=10.0,
        price_change_pct=10.0,
        metrics={"oi_momentum": "increasing"},
        divergence={"detected": False},
        current_price=55000
    )
    
    assert regime == "bullish_trend"
    assert confidence == "high"
    assert "bullish" in summary.lower()
    assert signals["signal"] == "strong_bullish"
    assert signals["action"] == "buy"
    
    # Test bearish market conditions
    regime, confidence, summary, signals = _classify_market_regime(
        oi_change=10000,
        price_change=-5000,
        oi_change_pct=10.0,
        price_change_pct=-10.0,
        metrics={"oi_momentum": "stable"},
        divergence={"detected": False},
        current_price=45000
    )
    
    assert regime == "bearish_trend"
    assert "bearish" in summary.lower()
    assert signals["signal"] == "bearish"
    assert signals["action"] == "sell"
    
    # Test neutral market conditions
    print("\nTESTING NEUTRAL MARKET REGIME CLASSIFICATION:")
    regime, confidence, summary, signals = _classify_market_regime(
        oi_change=500,
        price_change=250,
        oi_change_pct=0.5,
        price_change_pct=0.5,
        metrics={"oi_momentum": "stable"},
        divergence={"detected": False},
        current_price=50250
    )
    
    print(f"Regime: {regime}, Confidence: {confidence}")
    print(f"Summary: {summary}")
    
    assert regime == "neutral"
    assert "equilibrium" in summary.lower()
    assert signals["signal"] == "neutral"
    assert signals["action"] == "wait" 