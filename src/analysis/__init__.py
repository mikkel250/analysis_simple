"""
Financial market analysis package.

This package provides tools for fetching, analyzing, and visualizing
financial market data with different trading timeframes.
"""

from src.analysis.market_analyzer import MarketAnalyzer
from src.analysis.trading_styles import (
    SHORT_SETTINGS,
    MEDIUM_SETTINGS,
    LONG_SETTINGS,
    short_style,
    medium_style,
    long_style,
    fetch_data_for_current_style,
    apply_analysis_for_current_style,
    plot_for_current_style
)
from src.analysis.market_data import (
    fetch_market_data,
    add_technical_indicators,
    plot_technical_analysis,
    plot_candlestick,
    get_performance_summary
)

__all__ = [
    'MarketAnalyzer',
    'SHORT_SETTINGS',
    'MEDIUM_SETTINGS',
    'LONG_SETTINGS',
    'short_style',
    'medium_style',
    'long_style',
    'fetch_data_for_current_style',
    'apply_analysis_for_current_style',
    'plot_for_current_style',
    'fetch_market_data',
    'add_technical_indicators',
    'plot_technical_analysis',
    'plot_candlestick',
    'get_performance_summary'
] 