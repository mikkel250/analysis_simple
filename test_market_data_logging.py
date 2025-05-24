import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# Ensure src is in the path for import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import get_logger first to ensure logging is configured
from src.config.logging_config import get_logger
logger = get_logger(__name__) # Logger for this test script itself

# Now import the module to test
try:
    from analysis import market_data
except ImportError as e:
    print(f"Error importing market_data: {e}")
    sys.exit(1)

def run_market_data_tests():
    print("--- Starting Market Data Logging Test ---")
    
    # Test get_stock_data (which should be logged)
    print("\nTesting get_stock_data...")
    try:
        # Use a common, reliable stock and a short period to avoid hitting API limits / long waits
        # Using a non-crypto stock to simplify yfinance symbol handling for this test
        btc_df = market_data.get_stock_data(symbol="AAPL", period="5d", interval="1d")
        if not btc_df.empty:
            print(f"Successfully fetched AAPL data. Shape: {btc_df.shape}")
            # print(btc_df.head())
        else:
            print("Fetched AAPL data but it was empty.")
    except Exception as e:
        print(f"Error during get_stock_data test for AAPL: {e}")
        logger.error("Error in get_stock_data test (AAPL)", exc_info=True)

    # Create a dummy DataFrame to test other functions
    # This avoids repeated API calls and isolates logging for processing functions
    print("\nCreating dummy DataFrame for further tests...")
    date_today = datetime.now().date()
    dummy_dates = pd.to_datetime([date_today - timedelta(days=x) for x in range(5)][::-1])
    dummy_data = {
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [104, 105, 103, 106, 107],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    dummy_df = pd.DataFrame(dummy_data, index=dummy_dates)
    print(f"Dummy DataFrame created. Shape: {dummy_df.shape}")

    # Test calculate_returns
    print("\nTesting calculate_returns...")
    try:
        returns_df = market_data.calculate_returns(dummy_df.copy())
        print(f"Calculated returns. Shape: {returns_df.shape}")
        # print(returns_df.head())
    except Exception as e:
        print(f"Error during calculate_returns test: {e}")
        logger.error("Error in calculate_returns test", exc_info=True)

    # Test calculate_rolling_statistics
    print("\nTesting calculate_rolling_statistics...")
    try:
        rolling_df = market_data.calculate_rolling_statistics(dummy_df.copy())
        print(f"Calculated rolling stats. Shape: {rolling_df.shape}")
        # print(rolling_df.head())
    except Exception as e:
        print(f"Error during calculate_rolling_statistics test: {e}")
        logger.error("Error in calculate_rolling_statistics test", exc_info=True)

    # Test add_technical_indicators (this might be complex if it relies on live API for configs)
    # For now, we'll just see if it runs and logs entry/exit
    # Note: `get_indicator` within `add_technical_indicators` might itself have issues if not mocked
    # or if underlying cache/API calls fail. This test focuses on `market_data.py` logging.
    print("\nTesting add_technical_indicators...")
    try:
        indicators_df = market_data.add_technical_indicators(dummy_df.copy(), symbol="DUMMY", timeframe="1d")
        print(f"Added technical indicators (attempted). Shape: {indicators_df.shape}")
        # print(indicators_df.head())
    except Exception as e:
        print(f"Error during add_technical_indicators test: {e}")
        logger.error("Error in add_technical_indicators test", exc_info=True)

    print("\n--- Market Data Logging Test Complete ---")
    print(f"Please check the console output and the 'app.log' file in the workspace root for logs from 'src.analysis.market_data'.")

if __name__ == "__main__":
    run_market_data_tests() 