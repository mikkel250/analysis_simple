"""
Test script for cache_service module.

This script tests the cache_service module by storing and retrieving DataFrames
with different timeframes and TTL values.
"""

import sys
import time
import pandas as pd
from datetime import datetime, timedelta
import os
import json

# Add the src directory to the Python path
sys.path.append('.')

# Import the data_fetcher and cache_service modules
from services.data_fetcher import get_historical_data
from services.cache_service import (
    store_dataframe,
    get_cached_dataframe,
    invalidate_cache,
    clean_expired_cache,
    clear_all_cache,
    get_cache_stats,
    get_cache_entries,
    DEFAULT_TTL,
    CACHE_DIR
)

def create_sample_dataframe():
    """Create a sample DataFrame for testing."""
    # Create a date range
    dates = pd.date_range(start=datetime.now() - timedelta(days=10), periods=100, freq='H')
    
    # Create sample data
    data = {
        'open': [100 + i * 0.1 for i in range(100)],
        'high': [105 + i * 0.1 for i in range(100)],
        'low': [95 + i * 0.1 for i in range(100)],
        'close': [102 + i * 0.1 for i in range(100)],
        'volume': [1000 + i * 10 for i in range(100)]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'timestamp'
    
    return df

def test_basic_cache_operations():
    """Test basic cache operations with a sample DataFrame."""
    print("\n=== Testing Basic Cache Operations ===")
    
    # Create a sample DataFrame
    df = create_sample_dataframe()
    print(f"Created sample DataFrame with shape: {df.shape}")
    
    # Store DataFrame in cache
    key = "test_sample_data"
    result = store_dataframe(key, df, metadata={"source": "test"})
    print(f"Stored DataFrame in cache: {result}")
    
    # Retrieve DataFrame from cache
    cached_df = get_cached_dataframe(key)
    print(f"Retrieved DataFrame from cache with shape: {cached_df.shape}")
    
    # Verify data integrity
    if cached_df is not None:
        print(f"Original DataFrame equals cached DataFrame: {df.equals(cached_df)}")
    else:
        print("Failed to retrieve DataFrame from cache")
    
    # Invalidate cache
    invalidated = invalidate_cache(key)
    print(f"Invalidated cache entry: {invalidated}")
    
    # Verify cache entry is gone
    cached_df = get_cached_dataframe(key)
    print(f"DataFrame after invalidation: {cached_df is None}")
    
    return True

def test_timeframe_ttl_strategies():
    """Test different TTL strategies based on timeframes."""
    print("\n=== Testing Timeframe TTL Strategies ===")
    
    # Create a sample DataFrame
    df = create_sample_dataframe()
    
    # Test different timeframes
    timeframes = ["1m", "5m", "1h", "1d", "1w"]
    
    for timeframe in timeframes:
        # Store DataFrame with specific timeframe
        key = f"test_data_{timeframe}"
        metadata = {"source": "test", "symbol": "BTC/USDT", "exchange": "binance"}
        result = store_dataframe(key, df, metadata=metadata, timeframe=timeframe)
        
        # Get the TTL value for this timeframe
        ttl = DEFAULT_TTL.get(timeframe, DEFAULT_TTL.get("1h"))
        
        print(f"Stored DataFrame for timeframe {timeframe} with TTL {ttl} seconds: {result}")
    
    # Get cache entries to verify TTL values
    entries = get_cache_entries()
    print(f"Cache entries: {len(entries)}")
    
    # Print basic info about each entry
    for entry in entries:
        key = entry["key"]
        timeframe = entry["timeframe"]
        created_at = datetime.fromisoformat(entry["created_at"])
        expires_at = datetime.fromisoformat(entry["expires_at"])
        ttl = entry["ttl"]
        
        # Calculate actual TTL
        actual_ttl = (expires_at - created_at).total_seconds()
        
        print(f"Entry: {key}, Timeframe: {timeframe}, TTL: {ttl} seconds, Actual TTL: {actual_ttl} seconds")
    
    return True

def test_expiration_handling():
    """Test cache expiration handling."""
    print("\n=== Testing Expiration Handling ===")
    
    # Create a sample DataFrame
    df = create_sample_dataframe()
    
    # Store DataFrame with a short TTL
    key = "test_short_ttl"
    short_ttl = 2  # 2 seconds
    result = store_dataframe(key, df, ttl=short_ttl)
    print(f"Stored DataFrame with short TTL ({short_ttl} seconds): {result}")
    
    # Verify cache hit immediately
    cached_df = get_cached_dataframe(key)
    print(f"Immediate cache hit: {cached_df is not None}")
    
    # Wait for expiration
    print(f"Waiting {short_ttl + 1} seconds for cache entry to expire...")
    time.sleep(short_ttl + 1)
    
    # Verify cache miss after expiration
    cached_df = get_cached_dataframe(key)
    print(f"Cache hit after expiration: {cached_df is not None}")
    
    # Try to get expired data with allow_expired=True
    cached_df = get_cached_dataframe(key, allow_expired=True)
    print(f"Cache hit with allow_expired=True: {cached_df is not None}")
    
    # Clean expired cache
    cleaned = clean_expired_cache()
    print(f"Cleaned {cleaned} expired cache entries")
    
    return True

def test_real_data_caching():
    """Test caching with real data from data_fetcher."""
    print("\n=== Testing Real Data Caching ===")
    
    # Get historical data
    print("Fetching BTC/USDT historical data...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100)
    print(f"Fetched {len(df)} rows of data.")
    
    # Store in cache
    key = "btc_usdt_1d"
    metadata = {
        "symbol": "BTC/USDT",
        "exchange": "binance",
        "source": "coingecko"
    }
    result = store_dataframe(key, df, metadata=metadata, timeframe="1d")
    print(f"Stored real data in cache: {result}")
    
    # Get cache stats
    stats = get_cache_stats()
    print("\nCache Statistics:")
    for stat_key, value in stats.items():
        print(f"  {stat_key}: {value}")
    
    return True

def main():
    """Run all tests."""
    print("Starting cache_service tests...")
    
    # Clear any existing cache for clean testing
    clear_all_cache()
    
    # Run tests
    tests = [
        test_basic_cache_operations,
        test_timeframe_ttl_strategies,
        test_expiration_handling,
        test_real_data_caching
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"Error in test {test.__name__}: {str(e)}")
            results.append((test.__name__, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, result in results:
        print(f"{test_name}: {'✅' if result else '❌'}")
    
if __name__ == "__main__":
    main() 