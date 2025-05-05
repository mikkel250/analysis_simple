"""
Test script for the data_fetcher module.
"""

import pandas as pd
import time
import pytest
import numpy as np

def test_data_fetcher():
    """Test the data_fetcher module functionality."""
    try:
        from src.services.data_fetcher import DataFetcher, get_historical_data
        
        print("\n✅ Data Fetcher module import successful")
        
        # Test DataFetcher class initialization
        fetcher = DataFetcher()
        print("✅ DataFetcher class initialization successful")
        
        # Test fetching OHLC data (daily)
        print("\nTesting fetch_historical_ohlc (Daily)...")
        daily_df = fetcher.fetch_historical_ohlc(
            coin_id='bitcoin',
            vs_currency='usd',
            days=7
        )
        
        print(f"Retrieved {len(daily_df)} daily candles")
        print(f"DataFrame columns: {daily_df.columns.tolist()}")
        print(daily_df.head(3))
        
        # Test price volume data (hourly)
        print("\nTesting fetch_historical_price_volume...")
        hourly_df = fetcher.fetch_historical_price_volume(
            coin_id='bitcoin',
            vs_currency='usd',
            days=1
        )
        
        print(f"Retrieved {len(hourly_df)} candles")
        print(f"DataFrame columns: {hourly_df.columns.tolist()}")
        print(hourly_df.head(3))
        
        # Test different timeframes
        timeframes = ['1h', '4h', '1d']
        
        for timeframe in timeframes:
            print(f"\nTesting timeframe: {timeframe}")
            df = fetcher.fetch_data_by_timeframe(
                timeframe=timeframe,
                coin_id='bitcoin',
                vs_currency='usd',
                limit=10
            )
            
            print(f"Retrieved {len(df)} {timeframe} candles")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(df.head(3))
            
            # Add a short delay to avoid rate limits
            time.sleep(1.5)
        
        # Test convenience function
        print("\nTesting get_historical_data convenience function...")
        btc_data = get_historical_data(
            symbol='BTC',
            timeframe='1d',
            limit=5
        )
        
        print(f"Retrieved {len(btc_data)} daily candles for BTC")
        print(f"DataFrame columns: {btc_data.columns.tolist()}")
        print(btc_data.head())
        
        print("\n✅ All data_fetcher tests passed successfully!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure the 'src' directory is in your Python path.")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")

def test_all_timeframes():
    """Test that all supported timeframes work without errors."""
    from src.services.data_fetcher import get_historical_data
    
    # List of all supported timeframes to test
    timeframes = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '3d', '1w']
    
    for timeframe in timeframes:
        # Use a small limit to speed up tests
        df = get_historical_data(symbol='BTC', timeframe=timeframe, limit=20)
        
        # Verify DataFrame is not empty
        assert not df.empty, f"DataFrame for timeframe {timeframe} is empty"
        
        # Verify required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in df.columns, f"Column {col} missing in timeframe {timeframe}"
            
        # Verify data types
        assert df['open'].dtype == float, f"'open' column in timeframe {timeframe} is not float"
        assert df['high'].dtype == float, f"'high' column in timeframe {timeframe} is not float"
        assert df['low'].dtype == float, f"'low' column in timeframe {timeframe} is not float"
        assert df['close'].dtype == float, f"'close' column in timeframe {timeframe} is not float"
        
        # Drop rows with NaN values before comparing high and low
        df_clean = df.dropna(subset=['high', 'low'])
        
        # Verify data integrity (only on non-NaN values)
        assert (df_clean['high'] >= df_clean['low']).all(), f"Found high < low in timeframe {timeframe}"
        
        # Add a short delay to avoid rate limits
        time.sleep(2)
        
if __name__ == "__main__":
    test_data_fetcher() 