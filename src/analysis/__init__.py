"""
Financial market analysis package.

This package provides tools for fetching, analyzing, and visualizing
financial market data with different trading timeframes.
"""

from .market_analyzer import MarketAnalyzer

# Import plotting functions from the new location to re-export if desired
from src.plotting.charts import (
    plot_price_history,
    plot_candlestick,
    plot_technical_analysis
)

__all__ = [
    'MarketAnalyzer',
    'plot_price_history',
    'plot_candlestick',
    'plot_technical_analysis',
] 