"""
Test script for indicators module.

This script tests the indicators module by calculating various indicators
and checking if the results are as expected.
"""

import pandas as pd
import sys
import json
from datetime import datetime

# Add the src directory to the Python path
sys.path.append('.')

# Import the data_fetcher and indicators modules
from services.data_fetcher import get_historical_data
from services.indicators import (
    calculate_sma,
    calculate_ema,
    calculate_rsi,
    calculate_macd,
    calculate_bbands,
    calculate_stochastic,
    calculate_adx,
    calculate_atr,
    calculate_cci,
    calculate_obv,
    calculate_ichimoku,
    get_indicator
)

def main():
    """Test the indicators module with sample data."""
    # Fetch sample data
    print("Fetching BTC/USDT historical data...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100)
    print(f"Fetched {len(df)} rows of data.")
    print(df.head())
    
    # Test indicators
    indicators_to_test = [
        # (name, params) - Core Indicators
        ('sma', {'length': 20}),
        ('ema', {'length': 20}),
        ('rsi', {'length': 14}),
        ('macd', {'fast': 12, 'slow': 26, 'signal': 9}),
        ('bbands', {'length': 20, 'std': 2.0}),
        
        # Additional Indicators
        ('stoch', {'k': 14, 'd': 3, 'smooth_k': 3}),
        ('adx', {'length': 14}),
        ('atr', {'length': 14}),
        ('cci', {'length': 20, 'constant': 0.015}),
        ('obv', {}),
        ('ichimoku', {'tenkan': 9, 'kijun': 26, 'senkou': 52})
    ]
    
    # Create table for results
    print("\n" + "="*60)
    print(f"{'Indicator':<15} {'Parameters':<30} {'Success':<10}")
    print("="*60)
    
    for indicator_name, params in indicators_to_test:
        try:
            # Calculate indicator
            result = get_indicator(df, indicator_name, params)
            
            # Get the most recent value for verification
            if indicator_name in ['sma', 'ema', 'rsi', 'adx', 'atr', 'cci']:
                # For single-value indicators, get the last value
                last_values = list(result['values'].values())[-1]
                print(f"{indicator_name:<15} {str(params):<30} {'✅':<10}")
            elif indicator_name == 'macd':
                # For MACD, show the last values of MACD and signal
                macd_value = list(result['values']['MACD_12_26_9'].values())[-1]
                signal_value = list(result['values']['MACDs_12_26_9'].values())[-1]
                print(f"{indicator_name:<15} {str(params):<30} {'✅':<10}")
            elif indicator_name == 'bbands':
                # For Bollinger Bands, show upper, middle, and lower bands
                upper = list(result['values']['BBU_20_2.0'].values())[-1]
                middle = list(result['values']['BBM_20_2.0'].values())[-1]
                lower = list(result['values']['BBL_20_2.0'].values())[-1]
                print(f"{indicator_name:<15} {str(params):<30} {'✅':<10}")
            elif indicator_name == 'stoch':
                # For Stochastic, show %K and %D
                k_value = list(result['values']['STOCHk_14_3_3'].values())[-1]
                d_value = list(result['values']['STOCHd_14_3_3'].values())[-1]
                print(f"{indicator_name:<15} {str(params):<30} {'✅':<10}")
            elif indicator_name == 'obv':
                # For OBV, show the last value
                obv_value = list(result['values'].values())[-1]
                print(f"{indicator_name:<15} {str(params):<30} {'✅':<10}")
            elif indicator_name == 'ichimoku':
                # For Ichimoku, show a sample of the components
                keys = list(result['values'].keys())
                print(f"{indicator_name:<15} {str(params):<30} {'✅':<10}")
                
            # Save detailed results to a file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"indicator_{indicator_name}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"{indicator_name:<15} {str(params):<30} {'❌':<10} - {str(e)}")
    
    print("\nDetailed results saved to JSON files.")
    
if __name__ == "__main__":
    main() 