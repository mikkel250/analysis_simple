"""
Market data functionality for financial analysis.

This module provides functions for fetching, processing, and visualizing
financial market data.
"""

import datetime as dt
from functools import lru_cache
from typing import Dict, List, Union, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas_ta as ta
from src.services.indicators import get_indicator

# Configure logging
logger = logging.getLogger(__name__)

# Default visualization settings
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Data fetching functions
@lru_cache(maxsize=128)
def get_stock_data(
    symbol: str, 
    period: str = 'max', 
    interval: str = '1d',
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None
) -> pd.DataFrame:
    """
    Fetch stock data using yfinance.
    
    Args:
        symbol: Stock ticker symbol
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        start: Start date for data fetching
        end: End date for data fetching
    
    Returns:
        DataFrame containing the stock data
    """
    # Format cryptocurrency symbols correctly for Yahoo Finance
    original_symbol = symbol
    crypto_symbols = ["BTC", "ETH", "XRP", "LTC", "BCH", "ADA", "DOT", "LINK", "BNB", "DOGE"]
    
    if symbol in crypto_symbols and not "-USD" in symbol and not "/" in symbol:
        symbol = f"{symbol}-USD"
    
    ticker = yf.Ticker(symbol)
    
    if start is not None and end is not None:
        data = ticker.history(start=start, end=end, interval=interval)
    else:
        data = ticker.history(period=period, interval=interval)
    
    # Drop rows with NaN values and reset index
    data = data.dropna()
    
    # If the data is empty, raise an exception
    if data.empty:
        raise ValueError(f"No data found for {symbol} with the given parameters")
    
    # Verify reasonable price range for Bitcoin (should be > $1,000)
    if original_symbol == "BTC" and 'close' in data.columns:
        last_price = data['Close'].iloc[-1] if 'Close' in data.columns else data['close'].iloc[-1]
        if last_price < 1000:
            # Price is suspiciously low for Bitcoin, try again with explicit BTC-USD
            if symbol != "BTC-USD":
                return get_stock_data("BTC-USD", period, interval, start, end)
            else:
                # If we're already using BTC-USD and still getting weird prices, log a warning
                logger.warning(f"Bitcoin price suspiciously low: ${last_price:.2f}. Data may be incorrect.")
    
    # Convert column names to lowercase
    data.columns = [col.lower() for col in data.columns]
    
    # Add symbol column
    data['symbol'] = original_symbol
    
    return data

def fetch_market_data(
    symbol: str, 
    period: str = 'max', 
    interval: str = '1d',
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None
) -> pd.DataFrame:
    """
    Fetch market data using yfinance (alias for get_stock_data).
    
    Args:
        symbol: Trading symbol (e.g., 'BTC-USDT', 'ETH-USDT')
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        start: Start date for data fetching
        end: End date for data fetching
    
    Returns:
        DataFrame containing the market data
    """
    logger.info(f"Fetching market data for {symbol} with interval={interval}, period={period}")
    return get_stock_data(symbol=symbol, period=period, interval=interval, start=start, end=end)

def get_multiple_stocks(
    symbols: List[str], 
    period: str = 'max', 
    interval: str = '1d',
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None
) -> pd.DataFrame:
    """
    Fetch data for multiple stock symbols.
    
    Args:
        symbols: List of stock ticker symbols
        period: Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        start: Start date for data fetching
        end: End date for data fetching
    
    Returns:
        DataFrame containing the combined stock data
    """
    dfs = []
    for symbol in symbols:
        try:
            df = get_stock_data(symbol=symbol, period=period, interval=interval, 
                               start=start, end=end)
            dfs.append(df)
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
    
    if not dfs:
        raise ValueError("No data found for any of the provided symbols")
    
    return pd.concat(dfs)

def get_market_index(index_symbol: str = '^GSPC', period: str = '1y') -> pd.DataFrame:
    """
    Fetch market index data (default is S&P 500).
    
    Args:
        index_symbol: Market index symbol (default is S&P 500)
        period: Time period to fetch
    
    Returns:
        DataFrame containing the market index data
    """
    return get_stock_data(symbol=index_symbol, period=period)

# Data preprocessing functions
def calculate_returns(
    data: pd.DataFrame, 
    price_col: str = 'close', 
    return_type: str = 'log'
) -> pd.DataFrame:
    """
    Calculate returns from price data.
    
    Args:
        data: DataFrame containing price data
        price_col: Column name containing price data
        return_type: Type of return to calculate ('log' or 'pct')
    
    Returns:
        DataFrame with an additional column for returns
    """
    df = data.copy()
    
    if return_type == 'log':
        df['return'] = np.log(df[price_col] / df[price_col].shift(1))
    else:  # 'pct'
        df['return'] = df[price_col].pct_change()
    
    return df

def calculate_rolling_statistics(
    data: pd.DataFrame, 
    column: str = 'close', 
    windows: List[int] = [20, 50, 200]
) -> pd.DataFrame:
    """
    Calculate rolling mean and standard deviation.
    
    Args:
        data: DataFrame containing price data
        column: Column name to calculate statistics for
        windows: List of window sizes for rolling calculations
    
    Returns:
        DataFrame with additional columns for rolling statistics
    """
    df = data.copy()
    
    for window in windows:
        df[f'sma_{window}'] = df[column].rolling(window=window).mean()
        df[f'std_{window}'] = df[column].rolling(window=window).std()
    
    return df

def add_technical_indicators(
    df: pd.DataFrame, 
    window_size: int = 20,
    price_col: str = 'close',
    symbol: str = 'unknown',
    timeframe: str = 'unknown'
) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame using the central indicators service.
    
    Args:
        df: DataFrame containing OHLCV data
        window_size: Default window size for indicators if not overridden by specific params
        price_col: Column name containing price data (typically 'close')
        symbol: Symbol for caching purposes
        timeframe: Timeframe for caching purposes
    
    Returns:
        DataFrame with technical indicators
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    required_ohlc = ['open', 'high', 'low', 'close']
    if not all(col in df.columns for col in required_ohlc):
        # Allow 'price_col' to be one of the ohlc columns if others are missing for specific indicators.
        # get_indicator will handle specific column requirements.
        pass # Further validation is implicitly handled by get_indicator or its sub-functions

    df_out = df.copy()

    indicator_configs = [
        {'name': 'sma', 'params': {'length': window_size, 'column': price_col}, 'col_name': f'sma_{window_size}'},
        {'name': 'ema', 'params': {'length': window_size, 'column': price_col}, 'col_name': f'ema_{window_size}'},
        {'name': 'rsi', 'params': {'length': 14, 'column': price_col}, 'col_name': 'rsi_14'},
        {'name': 'macd', 'params': {'fast': 12, 'slow': 26, 'signal': 9, 'column': price_col}, 
         'multi_cols': {'MACD_12_26_9': 'MACD_12_26_9', 'MACDh_12_26_9': 'MACDh_12_26_9', 'MACDs_12_26_9': 'MACDs_12_26_9'}},
        {'name': 'bbands', 'params': {'length': window_size, 'std': 2.0, 'column': price_col},
         'multi_cols': {f'BBL_{window_size}_2.0': f'BBL_{window_size}_2.0', f'BBM_{window_size}_2.0': f'BBM_{window_size}_2.0', f'BBU_{window_size}_2.0': f'BBU_{window_size}_2.0'}}, # Corrected order/names based on pandas_ta output for bbands
        {'name': 'adx', 'params': {'length': 14}, # ADX uses high, low, close internally
         'multi_cols': {'ADX_14': 'ADX_14', 'DMP_14': 'DMP_14', 'DMN_14': 'DMN_14'}},
        {'name': 'stoch', 'params': {'k': 14, 'd': 3, 'smooth_k': 3}, # Stochastic uses high, low, close
         'multi_cols': {'STOCHk_14_3_3': 'STOCHk_14_3_3', 'STOCHd_14_3_3': 'STOCHd_14_3_3'}},
        {'name': 'cci', 'params': {'length': 20, 'constant': 0.015}, 'col_name': 'CCI_20_0.015'}, # CCI uses high, low, close. Adjusted name for constant.
        {'name': 'atr', 'params': {'length': 14}, 'col_name': 'ATR_14'}, # ATR uses high, low, close
        {'name': 'obv', 'params': {}, 'col_name': 'OBV'}, # OBV uses close and volume
        {'name': 'ichimoku', 'params': {'tenkan': 9, 'kijun': 26, 'senkou': 52}, # Ichimoku uses high, low, close
         'multi_cols': { # Default pandas_ta names
             'ITS_9': 'ITS_9', # Tenkan-sen
             'IKS_26': 'IKS_26', # Kijun-sen
             'ISA_9': 'ISA_9',   # Senkou Span A (Note: pandas_ta uses 'ISA_9', 'ISB_26', 'ITS_9', 'IKS_26', 'ICS_26')
             'ISB_26': 'ISB_26', # Senkou Span B
             'ICS_26': 'ICS_26'  # Chikou Span
         }
        }
    ]

    for config in indicator_configs:
        try:
            indicator_data = get_indicator(
                df.copy(), # Pass a copy to avoid modification by get_indicator if it does so
                indicator=config['name'],
                params=config['params'],
                symbol=symbol,
                timeframe=timeframe,
                use_cache=True # Enable caching
            )
            
            if indicator_data and 'values' in indicator_data:
                values = indicator_data['values']
                if isinstance(values, dict) and not config.get('multi_cols'): # Single series expected
                    # This handles cases where 'values' is a dict of {timestamp: value}
                    # Convert to Series, align index with df_out, and assign
                    s = pd.Series(values, name=config['col_name']).astype(float)
                    # Align series index with dataframe index before assigning
                    s_aligned = s.reindex(df_out.index)
                    df_out[config['col_name']] = s_aligned

                elif isinstance(values, dict) and config.get('multi_cols'): # Multiple series expected (e.g., MACD, BBands)
                    # 'values' is a dict of {col_name_from_ta: {timestamp: value}}
                    for ta_col_name, df_col_name in config['multi_cols'].items():
                        if ta_col_name in values:
                            s = pd.Series(values[ta_col_name], name=df_col_name).astype(float)
                            s_aligned = s.reindex(df_out.index)
                            df_out[df_col_name] = s_aligned
                        else:
                            # Initialize column with NaNs if not returned by indicator service
                            df_out[df_col_name] = np.nan
                            logging.warning(f"Column {ta_col_name} not found in {config['name']} output. Initializing {df_col_name} with NaNs.")
                else:
                    logging.warning(f"Unexpected format for {config['name']} values or mismatched config. Skipping.")
            else:
                logging.warning(f"Could not retrieve or invalid data for indicator: {config['name']}. It will be missing from the DataFrame.")
                # Ensure columns are created with NaNs if indicator fails
                if config.get('col_name'):
                     df_out[config['col_name']] = np.nan
                if config.get('multi_cols'):
                    for df_col_name in config['multi_cols'].values():
                        df_out[df_col_name] = np.nan

        except Exception as e:
            logging.error(f"Error calculating indicator {config['name']} via get_indicator: {e}")
            # Ensure columns are created with NaNs if indicator fails
            if config.get('col_name'):
                    df_out[config['col_name']] = np.nan
            if config.get('multi_cols'):
                for df_col_name in config['multi_cols'].values():
                    df_out[df_col_name] = np.nan
    
    # Calculate Ichimoku Cloud Bullish flag after all indicators are processed
    if 'ISA_9' in df_out.columns and 'ISB_26' in df_out.columns:
        # Ensure columns are numeric before comparison
        isa_numeric = pd.to_numeric(df_out['ISA_9'], errors='coerce')
        isb_numeric = pd.to_numeric(df_out['ISB_26'], errors='coerce')
        # Check for NaNs that pd.to_numeric might introduce if errors='coerce'
        if isa_numeric.notna().all() and isb_numeric.notna().all():
            df_out['ichimoku_cloud_bullish'] = isa_numeric > isb_numeric
        else:
            df_out['ichimoku_cloud_bullish'] = False # Default if conversion resulted in NaNs
            logging.warning("NaN values encountered in ISA_9 or ISB_26 after numeric conversion; 'ichimoku_cloud_bullish' set to False.")
    else:
        df_out['ichimoku_cloud_bullish'] = False # Default if source spans not present
        if not ('ISA_9' in df_out.columns and 'ISB_26' in df_out.columns):
             logging.warning("'ISA_9' or 'ISB_26' not found for 'ichimoku_cloud_bullish' calculation. Defaulting to False.")

    return df_out

# Helper functions for common analysis tasks
def get_performance_summary(data: pd.DataFrame, price_col: str = 'close') -> Dict[str, float]:
    """
    Calculate performance summary statistics for a stock.
    
    Args:
        data: DataFrame containing price data
        price_col: Column name containing price data
    
    Returns:
        Dictionary of performance metrics
    """
    # Use percent returns for correct annualized return/volatility in percent
    df = calculate_returns(data, price_col=price_col, return_type='pct')
    
    # Calculate performance metrics
    start_price = df[price_col].iloc[0]
    end_price = df[price_col].iloc[-1]
    total_return = (end_price / start_price - 1) * 100
    
    daily_returns = df['return'].dropna()
    annual_return = daily_returns.mean() * 252 * 100
    annual_volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
    
    max_drawdown = 0
    peak = df[price_col].iloc[0]
    
    for price in df[price_col]:
        if price > peak:
            peak = price
        drawdown = (peak - price) / peak
        max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown *= 100  # Convert to percentage
    
    return {
        'start_date': df.index[0].strftime('%Y-%m-%d'),
        'end_date': df.index[-1].strftime('%Y-%m-%d'),
        'start_price': start_price,
        'end_price': end_price,
        'total_return_pct': total_return,
        'annualized_return_pct': annual_return,
        'annualized_volatility_pct': annual_volatility,
        'volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown
    }

def compare_stocks(
    symbols: List[str], 
    period: str = '1y', 
    normalize: bool = True
) -> Tuple[pd.DataFrame, go.Figure]:
    """
    Compare performance of multiple stocks.
    
    Args:
        symbols: List of stock ticker symbols
        period: Time period to fetch
        normalize: Whether to normalize prices to starting at 100
    
    Returns:
        Tuple of (DataFrame with combined data, Plotly figure)
    """
    # Get data for all symbols
    df = get_multiple_stocks(symbols=symbols, period=period)
    
    # Create a pivot table to make easier to compare closing prices
    pivot_df = df.pivot_table(values='close', index=df.index, columns='symbol')
    
    # Normalize if requested
    if normalize:
        for col in pivot_df.columns:
            pivot_df[col] = pivot_df[col] / pivot_df[col].iloc[0] * 100
    
    # Create plot
    fig = px.line(
        pivot_df, 
        x=pivot_df.index, 
        y=pivot_df.columns,
        title=f'Stock Price Comparison ({period})' + (' - Normalized' if normalize else ''),
        labels={'value': 'Price' + (' (Normalized)' if normalize else ''), 'variable': 'Symbol'}
    )
    
    fig.update_layout(template='plotly_white')
    
    return pivot_df, fig
