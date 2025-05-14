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
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas_ta as ta

# Configure logging
logger = logging.getLogger(__name__)

# Default visualization settings
DEFAULT_FIGSIZE = (12, 8)
DEFAULT_STYLE = 'ggplot'
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
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df: DataFrame containing OHLCV data
        window_size: Window size for indicators
        price_col: Column name containing price data
    
    Returns:
        DataFrame with technical indicators
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        raise ValueError("DataFrame must contain OHLCV columns (open, high, low, close)")
    
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate SMA
    df[f'sma_{window_size}'] = df[price_col].rolling(window=window_size).mean()
    
    # Calculate EMA
    df[f'ema_{window_size}'] = df[price_col].ewm(span=window_size, adjust=False).mean()
    
    # Calculate RSI
    # Initialize RSI column with neutral value
    df['rsi_14'] = 50.0  # Neutral RSI value as fallback (use float)
    try:
        rsi_values = ta.rsi(df[price_col], length=14)
        # Only update non-NaN values
        if rsi_values is not None:
            # Convert column to float64 if it's not already
            if df['rsi_14'].dtype != 'float64':
                df['rsi_14'] = df['rsi_14'].astype('float64')
            # Extract values as arrays before using them to avoid Series hashability issues
            mask = ~pd.isna(rsi_values).values  # Convert to numpy array
            valid_values = rsi_values.values[mask]  # Extract values as array
            valid_indices = rsi_values.index[mask]
            df.loc[valid_indices, 'rsi_14'] = valid_values
    except Exception as e:
        logging.warning(f"Failed to calculate RSI, using fallback value (50): {e}")
    
    # Calculate MACD
    # Initialize MACD columns with neutral values
    df['MACD_12_26_9'] = 0  # Neutral MACD line
    df['MACDs_12_26_9'] = 0  # Neutral MACD signal
    df['MACDh_12_26_9'] = 0  # Neutral MACD histogram
    try:
        df_macd = ta.macd(df[price_col], fast=12, slow=26, signal=9)
        if df_macd is not None and not df_macd.empty:
            # Rename columns to match original format
            df_macd.columns = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']
            # Ensure columns have float64 dtype before assignment
            for col in df_macd.columns:
                if col in df.columns:
                    # Convert column to float64 if it's not already
                    if df[col].dtype != 'float64':
                        df[col] = df[col].astype('float64')
                    # Only update non-NaN values for each column
                    df.loc[~pd.isna(df_macd[col]), col] = df_macd.loc[~pd.isna(df_macd[col]), col]
    except Exception as e:
        logging.warning(f"Failed to calculate MACD, using fallback values (0): {e}")
    
    # Calculate Bollinger Bands
    # Initialize Bollinger Bands with default values based on price
    avg_price = df[price_col].mean()
    bb_std = df[price_col].std() if not df[price_col].empty else avg_price * 0.02  # Default to 2% of avg price if empty
    
    df[f'BBM_{window_size}_2.0'] = df[price_col]  # Middle band defaults to price
    df[f'BBU_{window_size}_2.0'] = df[price_col] + 2 * bb_std  # Upper band defaults to price + 2*std
    df[f'BBL_{window_size}_2.0'] = df[price_col] - 2 * bb_std  # Lower band defaults to price - 2*std
    
    try:
        df_bb = ta.bbands(df[price_col], length=window_size, std=2.0)
        if df_bb is not None and not df_bb.empty:
            # Only update non-NaN values for each column
            for col in df_bb.columns:
                if col in df.columns:
                    df.loc[~pd.isna(df_bb[col]), col] = df_bb.loc[~pd.isna(df_bb[col]), col]
    except Exception as e:
        logging.warning(f"Failed to calculate Bollinger Bands, using price-based fallbacks: {e}")
    
    # Calculate ADX
    # Initialize ADX with weak trend value
    df['ADX_14'] = 15.0  # Default to weak trend (as float)
    df['DMP_14'] = 20.0  # Default positive directional movement (as float)
    df['DMN_14'] = 20.0  # Default negative directional movement (as float)
    
    try:
        df_adx = ta.adx(df['high'], df['low'], df[price_col], length=14)
        if df_adx is not None and not df_adx.empty:
            # Only update non-NaN values for each column
            for col in df_adx.columns:
                if col in df.columns:
                    # Convert column to float64 if it's not already
                    if df[col].dtype != 'float64':
                        df[col] = df[col].astype('float64')
                    # Only update non-NaN values
                    df.loc[~pd.isna(df_adx[col]), col] = df_adx.loc[~pd.isna(df_adx[col]), col]
    except Exception as e:
        logging.warning(f"Failed to calculate ADX, using fallback value (15): {e}")
    
    # Calculate Stochastic Oscillator
    # Initialize Stochastic with neutral values
    df['STOCHk_14_3_3'] = 50  # Default %K to neutral
    df['STOCHd_14_3_3'] = 50  # Default %D to neutral
    
    try:
        df_stoch = ta.stoch(df['high'], df['low'], df[price_col], k=14, d=3, smooth_k=3)
        if df_stoch is not None and not df_stoch.empty:
            # Only update non-NaN values for each column
            for col in df_stoch.columns:
                if col in df.columns:
                    # Convert column to float64 if it's not already
                    if df[col].dtype != 'float64':
                        df[col] = df[col].astype('float64')
                    # Only update non-NaN values with explicit array extraction
                    # Extract values as arrays before using them to avoid Series hashability issues
                    mask = ~pd.isna(df_stoch[col]).values  # Convert mask to numpy array
                    valid_values = df_stoch[col].values[mask]  # Extract values as array
                    valid_indices = df_stoch.index[mask]  # Get the valid indices
                    df.loc[valid_indices, col] = valid_values
    except Exception as e:
        logging.warning(f"Failed to calculate Stochastic Oscillator, using fallback values (50): {e}")
    
    # Calculate CCI (Commodity Channel Index)
    # Initialize CCI with neutral value
    df['CCI_20'] = 0.0  # Default to neutral (as float)
    
    try:
        cci_values = ta.cci(df['high'], df['low'], df[price_col], length=20)
        if cci_values is not None:
            # Convert column to float64 if it's not already
            if df['CCI_20'].dtype != 'float64':
                df['CCI_20'] = df['CCI_20'].astype('float64')
            # Only update non-NaN values
            df.loc[~pd.isna(cci_values), 'CCI_20'] = cci_values[~pd.isna(cci_values)]
    except Exception as e:
        logging.warning(f"Failed to calculate CCI, using fallback value (0): {e}")
    
    # Calculate ATR (Average True Range)
    # Initialize ATR with default value (1% of average price)
    df['ATR_14'] = avg_price * 0.01  # Default to 1% of average price
    
    try:
        atr_values = ta.atr(df['high'], df['low'], df[price_col], length=14)
        if atr_values is not None:
            # Only update non-NaN values
            df.loc[~pd.isna(atr_values), 'ATR_14'] = atr_values[~pd.isna(atr_values)]
    except Exception as e:
        logging.warning(f"Failed to calculate ATR, using fallback value (1% of price): {e}")
    
    # Calculate OBV (On-Balance Volume)
    # Initialize OBV with 0
    df['OBV'] = 0
    
    try:
        if 'volume' in df.columns:
            obv_values = ta.obv(df[price_col], df['volume'])
            if obv_values is not None:
                # Only update non-NaN values
                df.loc[~pd.isna(obv_values), 'OBV'] = obv_values[~pd.isna(obv_values)]
    except Exception as e:
        logging.warning(f"Failed to calculate OBV, using fallback value (0): {e}")
    
    # Calculate Ichimoku Cloud
    try:
        if len(df) >= 52:  # Ichimoku requires more data points
            # Initialize Ichimoku Cloud components with default values
            tenkan_key = 'ITS_9'  # Tenkan-sen (Conversion Line)
            kijun_key = 'IKS_26'  # Kijun-sen (Base Line)
            senkou_a_key = 'ISA_9'  # Senkou Span A (Leading Span A)
            senkou_b_key = 'ISB_26'  # Senkou Span B (Leading Span B)
            chikou_key = 'ICS_26'  # Chikou Span (Lagging Span)
            
            # Default values based on price
            df[tenkan_key] = df[price_col]
            df[kijun_key] = df[price_col]
            df[senkou_a_key] = df[price_col]
            df[senkou_b_key] = df[price_col] * 0.98  # Default slightly below price
            df[chikou_key] = df[price_col].shift(-26)  # Default to price shifted backward
            df['ichimoku_cloud_bullish'] = False  # Default to False

            # pandas_ta.ichimoku returns a tuple of DataFrames, need to handle differently
            try:
                ichimoku = ta.ichimoku(df['high'], df['low'], df[price_col])
                
                # Check if result is a tuple (which is what pandas_ta.ichimoku returns)
                if isinstance(ichimoku, tuple):
                    # Process all DataFrames in the tuple
                    for i, df_ichimoku in enumerate(ichimoku):
                        if isinstance(df_ichimoku, pd.DataFrame):
                            # Align indexes before updating values
                            common_index = df.index.intersection(df_ichimoku.index)
                            
                            if len(common_index) > 0:
                                # Only update non-NaN values for each column
                                for col in df_ichimoku.columns:
                                    if col in df.columns:
                                        # Convert column to float64 if it's not already
                                        if df[col].dtype != 'float64':
                                            df[col] = df[col].astype('float64')
                                        
                                        # Get values only for the common index
                                        df_ichimoku_aligned = df_ichimoku.loc[common_index]
                                        # Update only non-NaN values within the common index using arrays
                                        mask = ~df_ichimoku_aligned[col].isna().values  # Convert to numpy array
                                        if mask.any():
                                            valid_indices = common_index[mask]
                                            valid_values = df_ichimoku_aligned.loc[mask, col].values
                                            df.loc[valid_indices, col] = valid_values
            except Exception as e:
                logging.warning(f"Failed to process Ichimoku Cloud components: {e}")
                # Fall back to default values already set
                
            # Add custom columns for Ichimoku components for easier reference
            if 'ISA_9' in df.columns and 'ISB_26' in df.columns:
                # Create boolean array directly for cloud_bullish
                cloud_bullish = np.zeros(len(df), dtype=bool)
                
                # Process each row to compare values, handling different structures
                for i in range(len(df)):
                    try:
                        # Get the values for comparison - convert to float to ensure comparison works
                        isa_val = float(df['ISA_9'].iloc[i])
                        isb_val = float(df['ISB_26'].iloc[i])
                        
                        # Compare only if both values are valid numbers
                        if not (pd.isna(isa_val) or pd.isna(isb_val)):
                            cloud_bullish[i] = isa_val > isb_val
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Row-wise comparison failed at index {i}: {e}")
                        # Keep the default value (False) for this row
                
                # Assign the calculated bullish values to the DataFrame
                df['ichimoku_cloud_bullish'] = cloud_bullish
    except Exception as e:
        logging.warning(f"Failed to calculate Ichimoku Cloud: {e}")
    
    return df

# Visualization functions
def plot_price_history(
    data: pd.DataFrame, 
    title: str = 'Stock Price History',
    price_col: str = 'close',
    figsize: Tuple[int, int] = DEFAULT_FIGSIZE,
    style: str = DEFAULT_STYLE
) -> go.Figure:
    """
    Plot stock price history using plotly.
    
    Args:
        data: DataFrame containing price data
        title: Plot title
        price_col: Column name containing price data
        figsize: Figure size as (width, height)
        style: Matplotlib style (ignored in plotly implementation)
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Check if we have multiple symbols
    if 'symbol' in data.columns and len(data['symbol'].unique()) > 1:
        for symbol, group in data.groupby('symbol'):
            fig.add_trace(
                go.Scatter(
                    x=group.index, 
                    y=group[price_col], 
                    name=symbol,
                    mode='lines'
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data[price_col],
                name=price_col,
                mode='lines'
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white'
    )
    
    return fig

def plot_candlestick(
    data: pd.DataFrame, 
    title: str = 'Candlestick Chart',
    include_volume: bool = True
) -> go.Figure:
    """
    Create a candlestick chart using plotly.
    
    Args:
        data: DataFrame containing OHLC data
        title: Chart title
        include_volume: Whether to include volume subplot
    
    Returns:
        Plotly figure
    """
    if include_volume:
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('', 'Volume'),
            row_heights=[0.8, 0.2]
        )
    else:
        fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add volume trace if requested
    if include_volume and 'volume' in data.columns:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

def plot_technical_analysis(
    data: pd.DataFrame, 
    title: str = 'Technical Analysis',
    price_col: str = 'close',
    include_volume: bool = True
) -> go.Figure:
    """
    Create a technical analysis chart with indicators.
    
    Args:
        data: DataFrame containing price data and indicators
        title: Chart title
        price_col: Column name containing price data
        include_volume: Whether to include volume subplot
    
    Returns:
        Plotly figure
    """
    # First ensure we have technical indicators
    if not any(col in data.columns for col in ['rsi_14', 'MACD_12_26_9', 'BBL_20_2.0']):
        data = add_technical_indicators(data, price_col=price_col)
    
    # Determine how many rows we need for the subplots based on available indicators
    rows = 2  # Default: price and RSI
    
    # Check if we have each indicator type
    has_volume = 'volume' in data.columns and include_volume
    has_rsi = 'rsi_14' in data.columns
    has_macd = 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns
    has_stochastic = 'STOCHk_14_3_3' in data.columns and 'STOCHd_14_3_3' in data.columns
    has_ichimoku = 'ISA_9' in data.columns and 'ISB_26' in data.columns
    
    # Count additional rows needed
    if has_volume:
        rows += 1
    if has_macd:
        rows += 1
    if has_stochastic:
        rows += 1
    
    # Create figure with subplots - dynamic rows based on indicators available
    subplot_titles = ['Price']
    row_heights = [0.5]  # Price chart gets more space
    
    if has_volume:
        subplot_titles.append('Volume')
        row_heights.append(0.1)
    
    if has_rsi:
        subplot_titles.append('RSI')
        row_heights.append(0.1)
    
    if has_macd:
        subplot_titles.append('MACD')
        row_heights.append(0.1)
    
    if has_stochastic:
        subplot_titles.append('Stochastic')
        row_heights.append(0.1)
    
    # Normalize heights to sum to 1
    row_heights = [h/sum(row_heights) for h in row_heights]
    
    # Create the subplots
    fig = make_subplots(
        rows=len(subplot_titles), 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=subplot_titles,
        row_heights=row_heights
    )
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='OHLC'
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    for col, color, name in zip(['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'], 
                              ['rgba(0,255,0,0.5)', 'rgba(0,0,255,0.5)', 'rgba(255,0,0,0.5)'],
                              ['Lower Band', 'Middle Band', 'Upper Band']):
        if col in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[col],
                    name=name,
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )
    
    # Add Ichimoku Cloud if available
    if has_ichimoku:
        # Add Tenkan-sen (Conversion Line)
        if 'ITS_9' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['ITS_9'],
                    name='Tenkan-sen',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
        
        # Add Kijun-sen (Base Line)
        if 'IKS_26' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['IKS_26'],
                    name='Kijun-sen',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # Add Senkou Span A (Leading Span A)
        if 'ISA_9' in data.columns:
            senkou_a = data['ISA_9']
            
            # Add Senkou Span B (Leading Span B)
            if 'ISB_26' in data.columns:
                senkou_b = data['ISB_26']
                
                # Create filled cloud area
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=senkou_a,
                        name='Senkou Span A',
                        line=dict(color='rgba(119, 210, 131, 0.5)', width=1)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=senkou_b,
                        name='Senkou Span B',
                        line=dict(color='rgba(210, 131, 119, 0.5)', width=1),
                        fill='tonexty',
                        fillcolor='rgba(180, 180, 180, 0.2)'
                    ),
                    row=1, col=1
                )
    
    # Current row tracker
    current_row = 2
    
    # Add volume trace if requested
    if has_volume:
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.5)'
            ),
            row=current_row, col=1
        )
        current_row += 1
    
    # Add RSI
    if has_rsi:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['rsi_14'],
                name='RSI (14)',
                line=dict(color='purple', width=1)
            ),
            row=current_row, col=1
        )
        
        # Add RSI lines at 70 and 30
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # Add MACD
    if has_macd:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACD_12_26_9'],
                name='MACD Line',
                line=dict(color='blue', width=1)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['MACDs_12_26_9'],
                name='MACD Signal',
                line=dict(color='red', width=1)
            ),
            row=current_row, col=1
        )
        
        # Add MACD Histogram
        if 'MACDh_12_26_9' in data.columns:
            colors = ['rgba(0,255,0,0.5)' if val >= 0 else 'rgba(255,0,0,0.5)' 
                      for val in data['MACDh_12_26_9']]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['MACDh_12_26_9'],
                    name='MACD Histogram',
                    marker_color=colors
                ),
                row=current_row, col=1
            )
        current_row += 1
    
    # Add Stochastic
    if has_stochastic:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['STOCHk_14_3_3'],
                name='%K',
                line=dict(color='blue', width=1)
            ),
            row=current_row, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['STOCHd_14_3_3'],
                name='%D',
                line=dict(color='red', width=1)
            ),
            row=current_row, col=1
        )
        
        # Add Stochastic lines at 80 and 20
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        template='plotly_white'
    )
    
    return fig

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
