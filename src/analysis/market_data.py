"""
Market data functionality for financial analysis.

This module provides functions for fetching, processing, and visualizing
financial market data.
"""

from typing import Dict, List, Optional, Any
import logging

import numpy as np
import pandas as pd
# import plotly.express as px # Currently unused
# import plotly.graph_objects as go # Currently unused
import pandas_ta as ta

# Configure logging
logger = logging.getLogger(__name__)

# Default visualization settings (May be unused if plotting functions are removed)
DEFAULT_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class MarketData:
    """
    Represents and processes market data for a specific symbol and timeframe.
    """
    def __init__(self, data_df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Initialize MarketData.

        Args:
            data_df: DataFrame containing OHLCV data.
            symbol: The trading symbol (e.g., 'BTC/USDT').
            timeframe: The timeframe for the data (e.g., '1h', '1d').
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.error("MarketData initialized with empty or invalid DataFrame.")
            # Potentially raise ValueError or handle appropriately
            self.data = pd.DataFrame() # Initialize with empty DataFrame
        else:
            self.data = data_df.copy() # Ensure we work with a copy
        
        self.symbol = symbol
        self.timeframe = timeframe
        logger.info(f"MarketData initialized for {symbol} ({timeframe}) with {len(self.data)} rows.")

    def add_technical_indicators(
        self,
        indicators_config: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        """
        Add technical indicators to the internal DataFrame using pandas-ta.

        Args:
            indicators_config: Optional dictionary to specify indicators and their
                               parameters. Example:
                               {
                                   'sma': {'lengths': [20, 50]},
                                   'rsi': {'length': 14},
                                   'macd': {'fast': 12, 'slow': 26, 'signal': 9}
                               }
        """
        if self.data is None or self.data.empty:
            logger.warning("DataFrame is empty or None, cannot add indicators.")
            return

        required_ohlc = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_ohlc if col not in self.data.columns]
        if missing_cols:
            logger.warning(
                f"DataFrame for {self.symbol} ({self.timeframe}) is missing one or more required OHLC columns: "
                f"{missing_cols}. Some indicators might fail."
            )

        default_indicators = {
            'sma': {'lengths': [20, 50, 100, 200]},
            'ema': {'lengths': [12, 26, 50]},
            'rsi': {'length': 14},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bbands': {'length': 20, 'std': 2},
            'atr': {'length': 14},
            'obv': {},
            'vwap': {},
            'ichimoku': {'tenkan': 9, 'kijun': 26, 'senkou': 52, 'include_chikou': True},
            'psar': {},
            'willr': {'length': 14},
            'cmf': {'length': 20},
            'heikinashi': {},
            'stoch': {'k': 14, 'd': 3},
            'kc': {'length': 20, 'scalar': 2, 'mamode': 'ema'},
            'cci': {'length': 20},
            'adx': {'length': 14},
        }

        config = indicators_config if indicators_config is not None else default_indicators
        logger.info(f"Adding technical indicators to {self.symbol} ({self.timeframe}) with config: {config}")

        strategy_ta_list = []
        if isinstance(config, dict):
            for indicator_key, params in config.items():
                study = {"kind": indicator_key}
                study.update(params)
                if indicator_key in ['sma', 'ema'] and 'lengths' in params:
                    for length_val in params['lengths']:
                        strategy_ta_list.append(
                            {"kind": indicator_key, "length": length_val}
                        )
                elif indicator_key == 'ichimoku':
                    strategy_ta_list.append({
                        "kind": "ichimoku",
                        "tenkan": params.get('tenkan', 9),
                        "kijun": params.get('kijun', 26),
                        "senkou": params.get('senkou', 52),
                        "include_chikou": params.get('include_chikou', True) # Ensure chikou can be configured
                    })
                elif indicator_key == 'heikinashi':
                    # Calculate Heikin Ashi candles using pandas_ta.ha()
                    ha_df = ta.ha(self.data['open'], self.data['high'], self.data['low'], self.data['close'])
                    ha_df.rename(columns={
                        'HA_open': 'heikinashi_open',
                        'HA_high': 'heikinashi_high',
                        'HA_low': 'heikinashi_low',
                        'HA_close': 'heikinashi_close',
                    }, inplace=True)
                    self.data = pd.concat([self.data, ha_df], axis=1)
                elif indicator_key == 'vwap':
                    if 'volume' in self.data.columns:
                        strategy_ta_list.append(study)
                    else:
                        logger.warning(
                            f"VWAP requires 'volume' column for {self.symbol} ({self.timeframe}), "
                            f"skipping VWAP strategy entry."
                        )
                elif indicator_key == 'obv':
                    if 'volume' in self.data.columns:
                        strategy_ta_list.append(study)
                    else:
                        logger.warning(
                            f"OBV requires 'volume' column for {self.symbol} ({self.timeframe}), "
                            f"skipping OBV strategy entry."
                        )
                elif indicator_key == 'cmf':
                    if 'volume' in self.data.columns:
                        strategy_ta_list.append(study)
                    else:
                        logger.warning(
                            f"CMF requires 'volume' column for {self.symbol} ({self.timeframe}), "
                            f"skipping CMF strategy entry."
                        )
                elif indicator_key == 'kc':
                    # Keltner Channels require OHLC columns, which are already checked above
                    strategy_ta_list.append(study)
                else:
                    strategy_ta_list.append(study)
        else:
            logger.error(f"Indicators config is not a dictionary: {config} for {self.symbol} ({self.timeframe})")

        if not strategy_ta_list:
            logger.warning(f"No valid TA studies configured for {self.symbol} ({self.timeframe}). Skipping indicator calculation.")
            return

        # Ensure all column names are lowercase for pandas_ta compatibility
        self.data.columns = [c.lower() if isinstance(c, str) else c for c in self.data.columns]

        try:
            strategy = ta.Strategy(
                name="MarketAnalysisStrategy",
                description="Calculates a standard set of technical indicators.",
                ta=strategy_ta_list
            )
            logger.debug(f"Applying pandas-ta strategy to {self.symbol} ({self.timeframe}): {strategy_ta_list}")
            self.data.ta.strategy(strategy)
            logger.info(f"Columns after indicator calculation: {self.data.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error applying pandas-ta strategy to {self.symbol} ({self.timeframe}): {e}", exc_info=True)

        logger.info(
            f"DataFrame for {self.symbol} ({self.timeframe}) with indicators. Columns: {self.data.columns.tolist()}"
        )


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
    df_calc = data.copy() # Work with a copy

    if price_col not in df_calc.columns:
        logger.error(f"Price column '{price_col}' not found in DataFrame.")
        return df_calc # Or raise error

    if df_calc[price_col].isnull().all():
        logger.warning(f"Price column '{price_col}' contains all NaNs. Cannot calculate returns.")
        df_calc['return'] = np.nan
        return df_calc

    if return_type == 'log':
        # Ensure no zero or negative prices for log returns
        if (df_calc[price_col] <= 0).any():
            logger.warning("Non-positive prices found. Log returns may be undefined or inaccurate. Proceeding with caution.")
        # Replace non-positive with NaN before log to avoid errors, or handle as per strategy
        # For simplicity, let log handle np.log(0) -> -inf, np.log(<0) -> nan
        df_calc['return'] = np.log(df_calc[price_col] / df_calc[price_col].shift(1))
    else:  # 'pct'
        df_calc['return'] = df_calc[price_col].pct_change()

    return df_calc


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
    df_calc = data.copy()

    if column not in df_calc.columns:
        logger.error(f"Column '{column}' for rolling stats not found.")
        return df_calc

    for window in windows:
        if window <= 0:
            logger.warning(f"Invalid window size {window} <= 0. Skipping.")
            continue
        if len(df_calc[column]) < window:
            logger.warning(f"Data length ({len(df_calc[column])}) is less than window size ({window}). Rolling stats will be mostly NaN.")
        
        df_calc[f'sma_{window}'] = df_calc[column].rolling(window=window, min_periods=1).mean() # min_periods=1 to get value even if less than window data
        df_calc[f'std_{window}'] = df_calc[column].rolling(window=window, min_periods=1).std()

    return df_calc


# Helper functions for common analysis tasks
# (Consider moving or removing if unused by core CLI)
# These functions might have been for yfinance data originally.

def get_performance_summary(
    data: pd.DataFrame, price_col: str = 'close'
) -> Dict[str, Any]: # Changed return type to Any for flexibility with str dates
    """
    Calculate performance summary statistics for a stock.

    Args:
        data: DataFrame containing price data
        price_col: Column name containing price data

    Returns:
        Dictionary of performance metrics
    """
    if data is None or data.empty or price_col not in data.columns or data[price_col].isnull().all():
        logger.warning("Performance summary cannot be calculated due to missing or all-NaN price data.")
        return {"error": "Invalid data for performance summary"}

    df_returns_calc = calculate_returns(
        data, price_col=price_col, return_type='pct'
    )

    if 'return' not in df_returns_calc.columns or df_returns_calc['return'].isnull().all():
        logger.warning("Returns could not be calculated or are all NaN. Performance summary will be limited.")
        start_price_val = data[price_col].dropna().iloc[0] if not data[price_col].dropna().empty else np.nan
        end_price_val = data[price_col].dropna().iloc[-1] if not data[price_col].dropna().empty else np.nan
        total_return_val = ((end_price_val / start_price_val) - 1) * 100 if pd.notna(start_price_val) and pd.notna(end_price_val) and start_price_val != 0 else 0.0
        
        return {
            'start_date': data.index[0].strftime('%Y-%m-%d') if not data.empty else 'N/A',
            'end_date': data.index[-1].strftime('%Y-%m-%d') if not data.empty else 'N/A',
            'total_return_pct': total_return_val,
            'annual_return_pct': 0.0,
            'annual_volatility_pct': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown_pct': 0.0,
            'start_price': float(start_price_val) if pd.notna(start_price_val) else 0.0,
            'end_price': float(end_price_val) if pd.notna(end_price_val) else 0.0,
            'status': 'Limited due to issues with return calculation'
        }


    start_price = df_returns_calc[price_col].iloc[0]
    end_price = df_returns_calc[price_col].iloc[-1]
    total_return = (end_price / start_price - 1) * 100 if start_price != 0 else 0

    daily_returns = df_returns_calc['return'].dropna()
    if daily_returns.empty:
        logger.warning("No valid daily returns to calculate annualized metrics.")
        annual_return = 0.0
        annual_volatility = 0.0
        sharpe_ratio = 0.0
    else:
        annual_return = daily_returns.mean() * 252 * 100
        annual_volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

    max_drawdown = 0.0 # Initialize as float
    peak = df_returns_calc[price_col].iloc[0] if not df_returns_calc[price_col].empty else np.nan

    if pd.notna(peak):
        for price in df_returns_calc[price_col]:
            if pd.notna(price): # Ensure price is not NaN
                if price > peak:
                    peak = price
                # Ensure peak is not zero to avoid division by zero
                if peak != 0:
                    drawdown = (peak - price) / peak
                    max_drawdown = max(max_drawdown, drawdown)
    
    max_drawdown *= 100

    return {
        'start_date': df_returns_calc.index[0].strftime('%Y-%m-%d') if not df_returns_calc.empty else 'N/A',
        'end_date': df_returns_calc.index[-1].strftime('%Y-%m-%d') if not df_returns_calc.empty else 'N/A',
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'annual_volatility_pct': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown_pct': max_drawdown,
        'start_price': float(start_price) if pd.notna(start_price) else 0.0,
        'end_price': float(end_price) if pd.notna(end_price) else 0.0,
    }


def compare_stocks(
    stock_data_list: List[pd.DataFrame],
    stock_names: List[str],
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Compare performance metrics for a list of stocks.

    Args:
        stock_data_list: List of DataFrames, each containing price data for a stock
        stock_names: List of names corresponding to the stocks
        price_col: Column name containing price data

    Returns:
        DataFrame summarizing performance metrics for all stocks
    """
    performance_summaries = []

    for i, stock_data in enumerate(stock_data_list):
        if stock_data is None or stock_data.empty:
            logger.warning(f"Empty data for stock {stock_names[i]}, skipping comparison.")
            continue
        summary = get_performance_summary(stock_data, price_col=price_col)
        if "error" not in summary:
            summary['stock'] = stock_names[i]
            performance_summaries.append(summary)
        else:
            logger.warning(f"Could not get performance summary for {stock_names[i]}: {summary.get('error')}")


    if not performance_summaries:
        logger.warning("No performance summaries could be generated for stock comparison.")
        return pd.DataFrame() # Return empty DataFrame if no valid summaries

    comparison_df = pd.DataFrame(performance_summaries)
    # Reorder columns for better readability
    if not comparison_df.empty:
        cols_order = ['stock', 'total_return_pct', 'annual_return_pct',
                      'annual_volatility_pct', 'sharpe_ratio', 'max_drawdown_pct',
                      'start_date', 'end_date']
        # Filter out any columns not present to avoid KeyError
        final_cols = [col for col in cols_order if col in comparison_df.columns]
        comparison_df = comparison_df[final_cols]
    
    return comparison_df
