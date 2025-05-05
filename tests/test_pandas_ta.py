"""
Test script to verify pandas-ta installation.
"""

try:
    # Import required libraries
    import pandas as pd
    import numpy as np
    import pandas_ta as ta
    
    # Create a sample DataFrame with OHLCV data
    data = {
        'open': [10, 11, 12, 13, 14],
        'high': [12, 13, 14, 15, 16],
        'low': [9, 10, 11, 12, 13],
        'close': [11, 12, 13, 14, 15],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    
    df = pd.DataFrame(data)
    
    # Test a simple SMA calculation
    result = ta.sma(df['close'], length=3)
    
    # Add result to the DataFrame
    df['sma_3'] = result
    
    # Print the result
    print("\n✅ pandas-ta installation test successful!")
    print("\nSample DataFrame with SMA(3):")
    print(df)
    
    # Print version information
    print(f"\nPandas-TA version: {ta.__version__}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    
except ImportError as e:
    print(f"❌ Installation test failed: {e}")
    print("Please install the required libraries with: pip install -r requirements.txt")
except Exception as e:
    print(f"❌ Test failed with error: {e}") 