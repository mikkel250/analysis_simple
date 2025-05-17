"""
Financial market analysis package.

This package provides tools for fetching, analyzing, and visualizing
financial market data with different trading timeframes.
"""

from .market_data import (
    get_stock_data,
    fetch_market_data,
    get_multiple_stocks,
    get_market_index,
    calculate_returns,
    calculate_rolling_statistics,
    add_technical_indicators,
    get_performance_summary,
    compare_stocks
)

from .market_analyzer import MarketAnalyzer
from .trading_styles import (
    apply_trading_style,
    get_current_trading_style,
    SUPPORTED_STYLES,
    TRADING_STYLE
)

# Import plotting functions from the new location to re-export if desired
from src.plotting.charts import (
    plot_price_history,
    plot_candlestick,
    plot_technical_analysis
)

__all__ = [
    'get_stock_data',
    'fetch_market_data',
    'get_multiple_stocks',
    'get_market_index',
    'calculate_returns',
    'calculate_rolling_statistics',
    'add_technical_indicators',
    'plot_price_history',
    'plot_candlestick',
    'plot_technical_analysis',
    'get_performance_summary',
    'compare_stocks',
    'MarketAnalyzer',
    'apply_trading_style',
    'get_current_trading_style',
    'SUPPORTED_STYLES',
    'TRADING_STYLE'
] 