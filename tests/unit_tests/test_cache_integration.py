"""
Test script for cache integration with data_fetcher.

This script tests the integration of cache_service with data_fetcher
by fetching data with and without caching to verify improved performance.
"""

import sys
import time
from datetime import datetime

# Add the src directory to the Python path
sys.path.append('.')

# Import the data_fetcher and cache_service modules
from services.data_fetcher import get_historical_data, invalidate_price_cache
from services.cache_service import clear_all_cache, get_cache_stats, get_cache_entries

def test_cache_performance():
    """Test cache performance improvement."""
    print("\n=== Testing Cache Performance ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Fetch data without cache (first request)
    print("Fetching BTC/USDT data without cache...")
    start_time = time.time()
    df1 = get_historical_data(symbol='BTC', timeframe='1d', limit=100, use_cache=True)
    first_fetch_time = time.time() - start_time
    print(f"First fetch (cache miss) took {first_fetch_time:.2f} seconds")
    print(f"Retrieved {len(df1)} rows of data")
    
    # Fetch the same data with cache (should be a cache hit)
    print("\nFetching the same data with cache...")
    start_time = time.time()
    df2 = get_historical_data(symbol='BTC', timeframe='1d', limit=100, use_cache=True)
    second_fetch_time = time.time() - start_time
    print(f"Second fetch (cache hit) took {second_fetch_time:.2f} seconds")
    print(f"Retrieved {len(df2)} rows of data")
    
    # Calculate the speedup
    if first_fetch_time > 0:
        speedup = first_fetch_time / max(second_fetch_time, 0.001)  # Avoid division by zero
        print(f"Cache speedup: {speedup:.1f}x faster")
    
    # Get cache statistics
    stats = get_cache_stats()
    print("\nCache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True

def test_different_timeframes():
    """Test caching with different timeframes."""
    print("\n=== Testing Different Timeframes ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Test different timeframes
    timeframes = ['1h', '4h', '1d']
    
    for timeframe in timeframes:
        # Fetch data
        print(f"\nFetching BTC/USDT data with timeframe {timeframe}...")
        start_time = time.time()
        df = get_historical_data(symbol='BTC', timeframe=timeframe, limit=50, use_cache=True)
        fetch_time = time.time() - start_time
        print(f"Fetched {len(df)} rows of data in {fetch_time:.2f} seconds")
        
        # Fetch the same data again (should be faster)
        print(f"Fetching the same data again...")
        start_time = time.time()
        df = get_historical_data(symbol='BTC', timeframe=timeframe, limit=50, use_cache=True)
        second_fetch_time = time.time() - start_time
        print(f"Second fetch took {second_fetch_time:.2f} seconds (speedup: {fetch_time / max(second_fetch_time, 0.001):.1f}x)")
    
    # Get cache entries
    entries = get_cache_entries()
    print(f"\nCache entries: {len(entries)}")
    
    # Print basic info about each entry
    for entry in entries[:5]:  # Show only first 5 entries to keep output clean
        print(f"Entry: {entry['key']}, Timeframe: {entry['timeframe']}, Expires: {entry['expires_at']}")
    
    return True

def test_cache_invalidation():
    """Test cache invalidation."""
    print("\n=== Testing Cache Invalidation ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Fetch data to fill cache
    print("Fetching BTC/USDT data to fill cache...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100, use_cache=True)
    print(f"Fetched {len(df)} rows of data")
    
    # Get cache entries before invalidation
    entries_before = get_cache_entries()
    print(f"Cache entries before invalidation: {len(entries_before)}")
    
    # Invalidate cache
    print("\nInvalidating cache for BTC 1d timeframe...")
    invalidated = invalidate_price_cache(symbol='BTC', timeframe='1d')
    print(f"Cache invalidated: {invalidated}")
    
    # Get cache entries after invalidation
    entries_after = get_cache_entries()
    print(f"Cache entries after invalidation: {len(entries_after)}")
    
    # Fetch data again (should be a cache miss)
    print("\nFetching BTC/USDT data after invalidation...")
    start_time = time.time()
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100, use_cache=True)
    fetch_time = time.time() - start_time
    print(f"Fetched {len(df)} rows of data in {fetch_time:.2f} seconds")
    
    return True

def main():
    """Run all tests."""
    print("Starting cache integration tests...")
    
    # Run tests
    tests = [
        test_cache_performance,
        test_different_timeframes,
        test_cache_invalidation
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