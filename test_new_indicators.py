#!/usr/bin/env python3
"""
Test script for new advanced technical indicators
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.services.indicators.get_indicator import get_indicator
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    price_base = 50000
    price_changes = np.random.normal(0, 0.02, 100)
    prices = [price_base]
    
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))
    
    # Create OHLCV data
    df = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    return df

def test_indicator(df, indicator_name, params=None):
    """Test a single indicator"""
    try:
        result = get_indicator(df, indicator_name, params=params, use_cache=False)
        values = result.get('values', {})
        
        if isinstance(values, dict):
            # Count non-null values
            if isinstance(list(values.values())[0], dict):
                # Multi-series indicator (like MACD)
                total_values = sum(len([v for v in series.values() if v is not None]) 
                                 for series in values.values())
            else:
                # Single series indicator
                total_values = len([v for v in values.values() if v is not None])
        else:
            total_values = 0
        
        print(f"✓ {indicator_name}: Success - {total_values} values")
        return True
    except Exception as e:
        print(f"✗ {indicator_name}: Error - {str(e)}")
        return False

def main():
    """Main test function"""
    print("Testing new advanced technical indicators...")
    print("=" * 50)
    
    # Create sample data
    df = create_sample_data()
    print(f"Created sample data with {len(df)} rows")
    print()
    
    # Test new indicators
    indicators_to_test = [
        ('awesome_oscillator', None),
        ('ultimate_oscillator', None),
        ('cci_enhanced', None),
        ('williams_r', None),
        ('vortex', None),
        ('alma', None),
        ('kama', None),
        ('trix', None),
        ('ppo', None),
        ('roc', None),
        ('aroon', None),
        ('fisher_transform', None),
    ]
    
    success_count = 0
    total_count = len(indicators_to_test)
    
    for indicator_name, params in indicators_to_test:
        if test_indicator(df, indicator_name, params):
            success_count += 1
    
    print()
    print("=" * 50)
    print(f"Test Results: {success_count}/{total_count} indicators working correctly")
    
    if success_count == total_count:
        print("🎉 All indicators are working correctly!")
        return 0
    else:
        print("⚠️  Some indicators have issues")
        return 1

if __name__ == "__main__":
    exit(main()) 