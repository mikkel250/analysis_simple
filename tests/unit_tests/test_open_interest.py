import pytest
import json
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.services.open_interest import (
    fetch_open_interest,
    analyze_open_interest,
    _get_cached_data,
    _cache_data,
    OPEN_INTEREST_CACHE_FILE
)

# Sample test data
SAMPLE_CURRENT_OI = {
    "symbol": "BTCUSDT",
    "openInterest": "100000.0",
    "timestamp": 1635350400000
}

SAMPLE_HISTORICAL_OI = [
    {
        "symbol": "BTCUSDT",
        "sumOpenInterest": "105000.0",
        "timestamp": 1635264000000  # 24 hours ago
    },
    {
        "symbol": "BTCUSDT",
        "sumOpenInterest": "102000.0",
        "timestamp": 1635307200000  # 12 hours ago
    }
]

@pytest.fixture
def mock_binance_client():
    """Create a mock Binance client for testing"""
    mock_client = MagicMock()
    mock_client.futures_open_interest.return_value = SAMPLE_CURRENT_OI
    mock_client.futures_open_interest_hist.return_value = SAMPLE_HISTORICAL_OI
    return mock_client

@pytest.fixture
def setup_cache_dir(tmp_path):
    """Setup a temporary cache directory for testing"""
    # Save the original cache path
    original_cache_file = OPEN_INTEREST_CACHE_FILE
    
    # Override with a temporary path for testing
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_file = cache_dir / "open_interest_cache.json"
    
    # Patch the cache file location
    with patch('src.services.open_interest.OPEN_INTEREST_CACHE_FILE', cache_file):
        yield cache_file
    
    # No need to restore as the test fixtures are isolated

@patch('src.services.open_interest.Client')
def test_fetch_open_interest(mock_client_class, mock_binance_client):
    """Test fetching open interest data from Binance API"""
    # Set up mock
    mock_client_class.return_value = mock_binance_client
    
    # Ensure _get_cached_data returns None to force API call
    with patch('src.services.open_interest._get_cached_data', return_value=None), \
         patch('src.services.open_interest._fetch_okx_open_interest') as mock_fetch_okx:
        # Configure the mock to return a valid result
        mock_fetch_okx.return_value = {
            "open_interest_value": 100000.0,
            "open_interest_change_24h": -1.96,
            "trend": "neutral",
            "interpretation": "Open interest has changed by -1.96% in the last 24 hours. This relatively stable open interest suggests a balance between new positions and positions being closed."
        }
        
        # Call the function - now OKX is the default
        result = fetch_open_interest("BTC-USDT")
    
    # Verify OKX function was called
    mock_fetch_okx.assert_called_once()
    
    # Check the result format
    assert "open_interest_value" in result
    assert "open_interest_change_24h" in result
    assert "trend" in result
    assert "interpretation" in result
    
    # Verify values based on our mocked return data
    assert result["open_interest_value"] == 100000.0
    assert result["open_interest_change_24h"] < 0  # Should show a decrease based on our mocked data

def test_analyze_open_interest():
    """Test analyzing raw open interest data"""
    # Sample input data for legacy function
    input_data = [
        {"timestamp": 1635350400000, "open_interest": 100000, "price": 60000, "volume": 1000},
        {"timestamp": 1635264000000, "open_interest": 95000, "price": 58000, "volume": 900}
    ]
    
    # Call the function
    result = analyze_open_interest(input_data)
    
    # Check expected format and values
    assert isinstance(result, dict)
    assert "regime" in result
    assert "summary" in result
    assert "value" in result
    assert "change_24h" in result
    assert "details" in result

def test_cached_data(setup_cache_dir):
    """Test caching functionality"""
    symbol = "BTCUSDT"
    test_data = {
        "open_interest_value": 100000.0,
        "open_interest_change_24h": -4.76,
        "trend": "neutral",
        "interpretation": "Test interpretation"
    }
    
    # Initially, no cache should exist
    assert _get_cached_data(symbol) is None
    
    # Cache the data
    _cache_data(symbol, test_data)
    
    # Verify cache file was created
    assert setup_cache_dir.exists()
    
    # Read the cache and verify content
    cached_data = _get_cached_data(symbol)
    assert cached_data is not None
    assert cached_data["open_interest_value"] == test_data["open_interest_value"]
    assert cached_data["trend"] == test_data["trend"]

@patch('src.services.open_interest.Client')
@patch('src.services.open_interest._get_cached_data')
def test_fetch_uses_cache_when_available(mock_get_cached, mock_client_class, mock_binance_client):
    """Test that fetch function uses cached data when available"""
    # Set up cached data
    cached_data = {
        "open_interest_value": 200000.0,  # Different from what the API would return
        "open_interest_change_24h": 10.5,
        "trend": "bullish",
        "interpretation": "Cached interpretation"
    }
    mock_get_cached.return_value = cached_data
    
    # Set up API mock
    mock_client_class.return_value = mock_binance_client
    
    # Call the function
    result = fetch_open_interest("BTC-USDT")
    
    # Verify cache was checked
    mock_get_cached.assert_called_once_with("BTCUSDT")
    
    # Verify API was NOT called because cache was used
    mock_client_class.assert_not_called()
    
    # Verify result is the cached data
    assert result == cached_data

@patch('src.services.open_interest.Client')
def test_fetch_open_interest_api_error(mock_client_class):
    """Test error handling for API errors"""
    # Set up mock to raise an exception
    from binance.exceptions import BinanceAPIException
    
    # Create the mock client
    mock_client = MagicMock()
    
    # Configure mock to raise exception on the first call
    mock_exception = BinanceAPIException(
        response=MagicMock(status_code=400, text="API Error"),
        status_code=400,
        text="API Error"
    )
    mock_client.futures_open_interest.side_effect = mock_exception
    
    # Return our configured mock client when Client is instantiated
    mock_client_class.return_value = mock_client
    
    # Also ensure _get_cached_data returns None to force API call
    with patch('src.services.open_interest._get_cached_data', return_value=None):
        # Call the function
        result = fetch_open_interest("BTC-USDT")
    
    # Verify error handling
    assert "error" in result
    assert result["trend"] == "neutral"
    assert result["open_interest_value"] == 0
    assert "Could not fetch" in result["interpretation"]

def test_breakout_strategy_content():
    """Test that the breakout strategy content is correctly included in the educational section"""
    result = analyze_open_interest()
    
    # Verify basic structure
    assert isinstance(result, dict)
    assert "educational" in result
    assert "breakout_strategy" in result["educational"]
    
    # Get the breakout strategy section
    breakout_strategy = result["educational"]["breakout_strategy"]
    
    # Verify all required fields exist
    assert "description" in breakout_strategy
    assert "when_to_use" in breakout_strategy
    assert "support_resistance_identification" in breakout_strategy
    assert "long_entry_guidelines" in breakout_strategy
    assert "short_entry_guidelines" in breakout_strategy
    assert "volume_confirmation" in breakout_strategy
    assert "risk_management" in breakout_strategy
    
    # Verify the content of key fields
    assert isinstance(breakout_strategy["support_resistance_identification"], list)
    assert len(breakout_strategy["support_resistance_identification"]) > 0
    
    # Verify long entry guidelines structure
    assert "entry_placement" in breakout_strategy["long_entry_guidelines"]
    assert "stop_loss" in breakout_strategy["long_entry_guidelines"]
    assert "target" in breakout_strategy["long_entry_guidelines"]
    
    # Verify short entry guidelines structure
    assert "entry_placement" in breakout_strategy["short_entry_guidelines"]
    assert "stop_loss" in breakout_strategy["short_entry_guidelines"]
    assert "target" in breakout_strategy["short_entry_guidelines"] 