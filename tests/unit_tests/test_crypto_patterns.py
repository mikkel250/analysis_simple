import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.services.crypto_patterns import CryptoPatternAnalyzer, PatternConfig

@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create more data points for better pattern formation
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    
    # Create a consolidation pattern followed by a breakout
    prices = []
    volumes = []
    base_price = 50000
    base_volume = 1000000
    
    # Initialize with some randomness
    price = base_price
    
    for i in range(100):
        if i < 70:  # Extended consolidation phase
            if i < 35:  # Initial higher volatility
                volatility = 0.003  # Higher initial volatility
                volume_mult = 1.0 + np.random.uniform(-0.1, 0.1)
            else:  # Decreasing volatility and increasing volume
                volatility = 0.001  # Very low volatility during accumulation
                # Steadily increasing volume
                volume_mult = 1.0 + (i-35)/35 * 0.5  # Up to 50% volume increase
                
            # Create mean-reverting price movement
            deviation = np.random.normal(0, volatility)
            price = base_price * (1 + deviation)
            # Keep price within a tight range
            price = max(min(price, base_price * 1.005), base_price * 0.995)
            
        else:  # Breakout phase
            if i == 70:  # Initial breakout
                volatility = 0.002
                volume_mult = 3.0  # Strong volume surge
            else:  # Continued momentum
                volatility = 0.002
                volume_mult = 2.5  # Sustained high volume
            
            # Create trending price movement
            trend = 0.002 * (i - 69)  # Steady trend
            price = price * (1 + trend)  # More consistent trend
            
        # Set volume with less randomness during key phases
        if i >= 70:  # Breakout phase
            volume = base_volume * volume_mult * (1 + np.random.uniform(-0.1, 0.1))
        else:  # Accumulation phase
            volume = base_volume * volume_mult * (1 + np.random.uniform(-0.05, 0.05))
        
        prices.append(price)
        volumes.append(volume)
    
    # Create OHLCV data with tight spreads
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'close': prices,
        'volume': volumes
    })
    
    # Add realistic high/low prices with controlled spread
    for i in range(len(prices)):
        if i < 35:  # Initial phase
            spread = 0.003
        elif i < 70:  # Accumulation phase
            spread = 0.001  # Tighter spreads during accumulation
        else:  # Breakout phase
            spread = 0.002
            
        data.loc[data.index[i], 'high'] = prices[i] * (1 + spread/2)
        data.loc[data.index[i], 'low'] = prices[i] * (1 - spread/2)
    
    return data

def test_pattern_config():
    """Test pattern configuration."""
    config = PatternConfig(
        min_consolidation_periods=15,
        volume_surge_threshold=2.5,
        momentum_threshold=0.8,
        volatility_compression=0.25,
        price_compression=0.015
    )
    
    assert config.min_consolidation_periods == 15
    assert config.volume_surge_threshold == 2.5
    assert config.momentum_threshold == 0.8
    assert config.volatility_compression == 0.25
    assert config.price_compression == 0.015

def test_accumulation_detection(sample_data):
    """Test accumulation pattern detection."""
    analyzer = CryptoPatternAnalyzer()
    accumulation = analyzer._detect_accumulation(sample_data)
    
    assert 'score' in accumulation
    assert 'is_accumulating' in accumulation
    assert 'metrics' in accumulation
    
    # Should detect accumulation in the first part of the data
    metrics = accumulation['metrics']
    assert metrics['price_range'] < 0.03  # Increased threshold for tight range
    assert metrics['volume_trend'] > 0.95  # Adjusted volume trend threshold
    assert metrics['volatility_trend'] < 0  # Decreasing volatility
    assert accumulation['is_accumulating'] == True  # Should detect accumulation

def test_volume_profile_analysis(sample_data):
    """Test volume profile analysis."""
    analyzer = CryptoPatternAnalyzer()
    volume_profile = analyzer._analyze_volume_profile(sample_data)
    
    assert 'score' in volume_profile
    assert 'metrics' in volume_profile
    
    metrics = volume_profile['metrics']
    assert 'recent_surge' in metrics
    assert 'above_vwap' in metrics
    assert 'high_vol_levels' in metrics
    
    # Should have significant volume profile score
    assert volume_profile['score'] >= 0.3  # At least one component should be positive

def test_volatility_analysis(sample_data):
    """Test volatility pattern analysis."""
    analyzer = CryptoPatternAnalyzer()
    volatility = analyzer._analyze_volatility(sample_data)
    
    assert 'score' in volatility
    assert 'metrics' in volatility
    
    metrics = volatility['metrics']
    assert 'current_volatility' in metrics
    assert 'is_compressed' in metrics
    assert 'vol_percentile' in metrics
    assert 'vol_trend' in metrics
    
    # Should have significant volatility score
    assert volatility['score'] >= 0.3  # At least one component should be positive

def test_momentum_alignment(sample_data):
    """Test momentum alignment analysis."""
    analyzer = CryptoPatternAnalyzer()
    momentum = analyzer._analyze_momentum_alignment(sample_data)
    
    assert 'score' in momentum
    assert 'metrics' in momentum
    
    metrics = momentum['metrics']
    assert 'rsi' in metrics
    assert 'macd_hist' in metrics
    assert 'trend' in metrics
    
    # RSI should be in reasonable range
    assert 0 <= metrics['rsi'] <= 100

def test_probability_calculation(sample_data):
    """Test overall probability calculation."""
    analyzer = CryptoPatternAnalyzer()
    result = analyzer.analyze_patterns(sample_data)
    
    assert result['status'] == 'success'
    assert 'probability' in result
    
    probability = result['probability']
    assert 'score' in probability
    assert 'confidence' in probability
    assert 'explanation' in probability
    assert 'component_scores' in probability
    
    # Score should be between 0 and 1
    assert 0 <= probability['score'] <= 1
    
    # Confidence should be one of low/medium/high
    assert probability['confidence'] in ['low', 'medium', 'high']
    
    # Component scores should all be present
    component_scores = probability['component_scores']
    assert 'accumulation' in component_scores
    assert 'volume_profile' in component_scores
    assert 'volatility' in component_scores
    assert 'momentum' in component_scores

def test_error_handling():
    """Test error handling for invalid inputs."""
    analyzer = CryptoPatternAnalyzer()
    
    # Test with empty DataFrame
    result = analyzer.analyze_patterns(pd.DataFrame())
    assert result['status'] == 'error'
    assert 'Insufficient data' in result['message']
    
    # Test with None
    result = analyzer.analyze_patterns(None)
    assert result['status'] == 'error'
    assert 'Insufficient data' in result['message']
    
    # Test with too little data
    small_data = pd.DataFrame({
        'close': [100, 101, 102],
        'high': [103, 104, 105],
        'low': [98, 99, 100],
        'volume': [1000, 1000, 1000]
    })
    result = analyzer.analyze_patterns(small_data)
    assert result['status'] == 'error'
    assert 'Insufficient data' in result['message'] 