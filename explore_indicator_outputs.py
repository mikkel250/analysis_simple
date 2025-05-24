import pandas as pd
from src.services.data_fetcher import get_historical_data
from src.services.indicators import calculate_rsi, calculate_macd

if __name__ == "__main__":
    # Fetch sample data
    print("Fetching BTC/USDT historical data...")
    df = get_historical_data(symbol='BTC', timeframe='1d', limit=100)
    print(f"Fetched {len(df)} rows of data.")
    print("DataFrame dtypes:")
    print(df.dtypes)
    print("DataFrame index type:", type(df.index))
    print("DataFrame index (first 10):", df.index[:10])
    print("DataFrame index (last 10):", df.index[-10:])
    print("DataFrame head (first 20 rows):")
    print(df.head(20))
    print("DataFrame tail (last 20 rows):")
    print(df.tail(20))
    print("NaN count per column:")
    print(df.isna().sum())

    # Calculate RSI
    print("\nCalculating RSI...")
    rsi_result = calculate_rsi(df, length=14, column='close', symbol='BTC', timeframe='1d', use_cache=False)
    print("RSI output (first 20 rows):")
    print(pd.Series(rsi_result['values']).head(20))
    print("RSI output (last 10 rows):")
    print(pd.Series(rsi_result['values']).tail(10))
    print(f"RSI NaN count: {pd.Series(rsi_result['values']).isna().sum()}")

    # Calculate MACD
    print("\nCalculating MACD...")
    macd_result = calculate_macd(df, fast=12, slow=26, signal=9, column='close', symbol='BTC', timeframe='1d', use_cache=False)
    for key, values in macd_result['values'].items():
        print(f"MACD component: {key} (first 20 rows):")
        print(pd.Series(values).head(20))
        print(f"MACD component: {key} (last 10 rows):")
        print(pd.Series(values).tail(10))
        print(f"MACD component: {key} NaN count: {pd.Series(values).isna().sum()}")

    print("\n--- Minimal pandas-ta test with synthetic data ---")
    import numpy as np
    import pandas_ta as ta
    # Create synthetic data
    idx = pd.date_range(start="2023-01-01", periods=100, freq="D")
    df_test = pd.DataFrame({
        'open': np.linspace(100, 200, 100),
        'high': np.linspace(101, 201, 100),
        'low': np.linspace(99, 199, 100),
        'close': np.linspace(100, 200, 100) + np.random.normal(0, 2, 100),
        'volume': np.random.uniform(1000, 2000, 100)
    }, index=idx)
    print("Synthetic DataFrame head:")
    print(df_test.head())
    print("Synthetic DataFrame tail:")
    print(df_test.tail())
    # RSI
    rsi = ta.rsi(df_test['close'], length=14)
    print("\n[Minimal Test] RSI first 20:")
    print(rsi.head(20))
    print("[Minimal Test] RSI last 20:")
    print(rsi.tail(20))
    # MACD
    macd = ta.macd(df_test['close'], fast=12, slow=26, signal=9)
    print("\n[Minimal Test] MACD first 20:")
    print(macd.head(20))
    print("[Minimal Test] MACD last 20:")
    print(macd.tail(20)) 