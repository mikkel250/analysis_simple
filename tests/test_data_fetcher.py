"""
Test script for the data_fetcher module.
"""

import pandas as pd
import time

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
        
if __name__ == "__main__":
    test_data_fetcher() 