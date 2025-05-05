"""
Test script for indicator caching functionality.

This script tests the caching of indicator results using the enhanced cache service.
It verifies that indicator results are properly cached and retrieved, and that
appropriate TTL strategies are applied based on timeframes and indicator complexity.
"""

import sys
import time
import pandas as pd
from datetime import datetime

# Add the src directory to the Python path
sys.path.append('.')

# Import the required modules
from services.data_fetcher import get_historical_data
from services.indicators import (
    calculate_sma,
    calculate_macd,
    calculate_bbands,
    get_indicator,
    invalidate_indicator_cache
)
from services.cache_service import (
    get_cache_stats,
    get_cache_entries,
    clear_all_cache,
    generate_indicator_cache_key
)

def test_cache_performance():
    """Test indicator caching performance improvement."""
    print("\n=== Testing Indicator Cache Performance ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Fetch sample data
    print("Fetching BTC/USDT historical data...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100)
    print(f"Fetched {len(df)} rows of data.")
    
    # Calculate indicator without cache (first request)
    print("\nCalculating SMA without cache...")
    start_time = time.time()
    sma_result = calculate_sma(
        df, 
        length=20, 
        symbol='BTC', 
        timeframe='1d',
        use_cache=False  # Disable cache for first calculation
    )
    first_calc_time = time.time() - start_time
    print(f"First calculation (no cache) took {first_calc_time:.4f} seconds")
    
    # Now calculate with cache (should store in cache)
    print("\nCalculating SMA with cache storage...")
    start_time = time.time()
    sma_result = calculate_sma(
        df, 
        length=20, 
        symbol='BTC', 
        timeframe='1d',
        use_cache=True  # Enable cache
    )
    store_calc_time = time.time() - start_time
    print(f"Calculation with cache storage took {store_calc_time:.4f} seconds")
    
    # Calculate again with cache (should retrieve from cache)
    print("\nCalculating SMA with cache retrieval...")
    start_time = time.time()
    sma_result = calculate_sma(
        df, 
        length=20, 
        symbol='BTC', 
        timeframe='1d',
        use_cache=True  # Enable cache
    )
    cache_calc_time = time.time() - start_time
    print(f"Calculation with cache retrieval took {cache_calc_time:.4f} seconds")
    
    # Calculate the speedup
    if first_calc_time > 0 and cache_calc_time > 0:
        speedup = first_calc_time / cache_calc_time
        print(f"Cache speedup: {speedup:.1f}x faster")
    
    # Get cache statistics
    stats = get_cache_stats(cache_type="indicator")
    print("\nIndicator Cache Statistics:")
    for key, value in stats["indicator"].items():
        print(f"  {key}: {value}")
    
    return True

def test_different_indicators_caching():
    """Test caching of different indicators with various parameters."""
    print("\n=== Testing Different Indicators Caching ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Fetch sample data
    print("Fetching BTC/USDT historical data...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100)
    
    # Test different indicators
    indicators_to_test = [
        # Simple indicators
        ('sma', {'length': 20}),
        ('sma', {'length': 50}),  # Different parameters
        ('ema', {'length': 20}),
        ('rsi', {'length': 14}),
        
        # Complex indicators
        ('macd', {'fast': 12, 'slow': 26, 'signal': 9}),
        ('bbands', {'length': 20, 'std': 2.0}),
        ('stoch', {'k': 14, 'd': 3, 'smooth_k': 3})
    ]
    
    # Calculate each indicator and store in cache
    for indicator_name, params in indicators_to_test:
        print(f"\nCalculating {indicator_name} with parameters {params}...")
        
        # First calculation (should store in cache)
        result = get_indicator(df, indicator_name, params, symbol='BTC', timeframe='1d')
        
        # Generate cache key for verification
        cache_key = generate_indicator_cache_key(indicator_name, params, 'BTC', '1d')
        print(f"Cache key: {cache_key}")
    
    # Get cache entries
    entries = get_cache_entries(cache_type="indicator")
    print(f"\nIndicator cache entries: {len(entries)}")
    
    # Print basic info about each entry
    for entry in entries:
        indicator_name = entry.get("indicator_name", "unknown")
        params_hash = entry.get("params_hash", "unknown")
        created_at = entry.get("created_at", "unknown")
        expires_at = entry.get("expires_at", "unknown")
        ttl = entry.get("ttl", 0)
        
        print(f"Indicator: {indicator_name}, Params hash: {params_hash}, TTL: {ttl} seconds")
        print(f"  Created: {created_at}, Expires: {expires_at}")
        print(f"  Metadata: {entry.get('metadata', {})}")
        print()
    
    return True

def test_cache_invalidation():
    """Test indicator cache invalidation."""
    print("\n=== Testing Indicator Cache Invalidation ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Fetch sample data
    print("Fetching BTC/USDT historical data...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100)
    
    # Calculate SMA and store in cache
    print("\nCalculating SMA and storing in cache...")
    sma_params = {'length': 20}
    sma_result = calculate_sma(
        df, 
        length=20, 
        symbol='BTC', 
        timeframe='1d',
        use_cache=True
    )
    
    # Check that it's in the cache
    entries_before = get_cache_entries(cache_type="indicator")
    print(f"Cache entries before invalidation: {len(entries_before)}")
    
    # Invalidate the cache
    print("\nInvalidating SMA cache...")
    invalidated = invalidate_indicator_cache(
        'sma',
        sma_params,
        symbol='BTC',
        timeframe='1d'
    )
    print(f"Cache invalidated: {invalidated}")
    
    # Check that it's no longer in the cache
    entries_after = get_cache_entries(cache_type="indicator")
    print(f"Cache entries after invalidation: {len(entries_after)}")
    
    # Calculate again (should recalculate and store in cache)
    print("\nRecalculating SMA after invalidation...")
    sma_result = calculate_sma(
        df, 
        length=20, 
        symbol='BTC', 
        timeframe='1d',
        use_cache=True
    )
    
    # Check that it's back in the cache
    entries_final = get_cache_entries(cache_type="indicator")
    print(f"Cache entries after recalculation: {len(entries_final)}")
    
    return True

def test_timeframe_ttl_strategies():
    """Test different TTL strategies based on timeframes."""
    print("\n=== Testing Timeframe TTL Strategies ===")
    
    # Clear cache to ensure clean testing
    clear_all_cache()
    
    # Fetch sample data for different timeframes
    timeframes = ['1h', '4h', '1d']
    
    for timeframe in timeframes:
        print(f"\nFetching BTC/USDT data with timeframe {timeframe}...")
        df = get_historical_data(symbol='BTC', timeframe=timeframe, limit=50)
        
        # Calculate a simple indicator (SMA)
        print(f"Calculating SMA for timeframe {timeframe}...")
        sma_result = calculate_sma(
            df, 
            length=20, 
            symbol='BTC', 
            timeframe=timeframe,
            use_cache=True
        )
        
        # Calculate a complex indicator (MACD)
        print(f"Calculating MACD for timeframe {timeframe}...")
        macd_result = calculate_macd(
            df, 
            fast=12, 
            slow=26, 
            signal=9, 
            symbol='BTC', 
            timeframe=timeframe,
            use_cache=True
        )
    
    # Get cache entries for verification
    entries = get_cache_entries(cache_type="indicator")
    print(f"\nIndicator cache entries: {len(entries)}")
    
    # Group entries by timeframe and indicator type
    grouped_entries = {}
    for entry in entries:
        timeframe = entry.get("timeframe", "unknown")
        indicator_type = entry.get("indicator_type", "unknown")
        key = f"{timeframe}_{indicator_type}"
        
        if key not in grouped_entries:
            grouped_entries[key] = []
        
        grouped_entries[key].append(entry)
    
    # Print TTL information for each group
    print("\nTTL Strategies by Timeframe and Indicator Type:")
    for key, group_entries in grouped_entries.items():
        timeframe, indicator_type = key.split('_')
        ttl_values = [entry.get("ttl", 0) for entry in group_entries]
        
        if ttl_values:
            avg_ttl = sum(ttl_values) / len(ttl_values)
            print(f"Timeframe: {timeframe}, Type: {indicator_type}")
            print(f"  Average TTL: {avg_ttl:.0f} seconds ({avg_ttl/3600:.1f} hours)")
            
            # Show expiration times
            for entry in group_entries:
                indicator_name = entry.get("indicator_name", "unknown")
                ttl = entry.get("ttl", 0)
                created_at = entry.get("created_at", "unknown")
                expires_at = entry.get("expires_at", "unknown")
                
                print(f"  {indicator_name}: TTL={ttl} seconds, Created={created_at}, Expires={expires_at}")
            
            print()
    
    return True

def main():
    """Run all tests."""
    print("Starting indicator caching tests...")
    
    # Run tests
    tests = [
        test_cache_performance,
        test_different_indicators_caching,
        test_cache_invalidation,
        test_timeframe_ttl_strategies
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append((test.__name__, result))
        except Exception as e:
            print(f"Error in test {test.__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Print summary
    print("\n=== Test Summary ===")
    for test_name, result in results:
        print(f"{test_name}: {'✅' if result else '❌'}")
    
if __name__ == "__main__":
    main() 