"""
Unit tests for the MarketAnalyzer class.

These tests cover all functionality of the MarketAnalyzer class including:
- Initialization with different timeframes
- Trading style selection
- Data fetching
- Analysis running
- Visualization generation
- Summary creation
"""

import os
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import patch, MagicMock, call

# Mock pandas_ta to avoid import issues
import sys
sys.modules['pandas_ta'] = MagicMock()

# Import the modules that will be used by the MarketAnalyzer
from src.jupyter.kernel import trading_styles

# Mock market_data module to avoid pandas_ta import issues
sys.modules['src.jupyter.kernel.market_data'] = MagicMock()
market_data = sys.modules['src.jupyter.kernel.market_data']

# Mock the market_analyzer module and MarketAnalyzer class
market_analyzer_module = MagicMock()
MarketAnalyzer = MagicMock()
market_analyzer_module.MarketAnalyzer = MarketAnalyzer
sys.modules['src.jupyter.market_analyzer'] = market_analyzer_module

# Fixtures

@pytest.fixture
def mock_stock_data():
    """Create mock stock data for testing."""
    # Create a simple DataFrame with OHLCV data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    data = {
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(100, 5, 100),
        'volume': np.random.randint(1000, 10000, 100),
        'symbol': ['AAPL'] * 100
    }
    
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def mock_performance_data():
    """Create mock performance data for testing."""
    return {
        'total_return_pct': 12.34,
        'annualized_return_pct': 8.76,
        'volatility': 15.67,
        'sharpe_ratio': 1.23,
        'max_drawdown_pct': 8.90,
        'win_rate_pct': 60.0
    }


# Test cases for initialization and trading style selection

def test_init_default_timeframe():
    """Test initialization with default timeframe (medium)."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    analyzer = MarketAnalyzer(symbol="AAPL")
    
    assert analyzer.symbol == "AAPL"
    assert analyzer.timeframe == "medium"
    assert analyzer.trading_style == trading_styles.MEDIUM_SETTINGS
    assert analyzer.data is None


def test_init_short_timeframe():
    """Test initialization with short timeframe."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="short")
    
    assert analyzer.symbol == "AAPL"
    assert analyzer.timeframe == "short"
    assert analyzer.trading_style == trading_styles.SHORT_SETTINGS
    assert analyzer.data is None


def test_init_medium_timeframe():
    """Test initialization with medium timeframe."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    
    assert analyzer.symbol == "AAPL"
    assert analyzer.timeframe == "medium"
    assert analyzer.trading_style == trading_styles.MEDIUM_SETTINGS
    assert analyzer.data is None


def test_init_long_timeframe():
    """Test initialization with long timeframe."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="long")
    
    assert analyzer.symbol == "AAPL"
    assert analyzer.timeframe == "long"
    assert analyzer.trading_style == trading_styles.LONG_SETTINGS
    assert analyzer.data is None


def test_init_invalid_timeframe():
    """Test initialization with invalid timeframe."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    with pytest.raises(ValueError) as excinfo:
        analyzer = MarketAnalyzer(symbol="AAPL", timeframe="invalid")
    
    assert "Invalid timeframe: invalid" in str(excinfo.value)


def test_get_trading_style():
    """Test _get_trading_style method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    analyzer = MarketAnalyzer(symbol="AAPL")
    
    # Test with valid timeframes
    assert analyzer._get_trading_style("short") == trading_styles.SHORT_SETTINGS
    assert analyzer._get_trading_style("medium") == trading_styles.MEDIUM_SETTINGS
    assert analyzer._get_trading_style("long") == trading_styles.LONG_SETTINGS
    
    # Test with invalid timeframe
    with pytest.raises(ValueError) as excinfo:
        analyzer._get_trading_style("invalid")
    
    assert "Invalid timeframe: invalid" in str(excinfo.value)


# Test cases for data fetching

@patch('src.jupyter.kernel.market_data.get_stock_data')
def test_fetch_data(mock_get_stock_data, mock_stock_data):
    """Test fetch_data method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Configure the mock to return our test data
    mock_get_stock_data.return_value = mock_stock_data
    
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    data = analyzer.fetch_data()
    
    # Verify get_stock_data was called with the correct parameters
    mock_get_stock_data.assert_called_once_with(
        symbol="AAPL", 
        interval=trading_styles.MEDIUM_SETTINGS['intervals'][1],
        period=trading_styles.MEDIUM_SETTINGS['periods'][1]
    )
    
    # Verify the data was stored and returned
    assert analyzer.data is not None
    assert data is mock_stock_data
    assert analyzer.data is mock_stock_data


@patch('src.jupyter.kernel.market_data.get_stock_data')
def test_fetch_data_error(mock_get_stock_data):
    """Test fetch_data method with an error."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Configure the mock to raise an exception
    mock_get_stock_data.side_effect = ValueError("No data found for AAPL")
    
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    
    with pytest.raises(ValueError) as excinfo:
        analyzer.fetch_data()
    
    assert "No data found for AAPL" in str(excinfo.value)


# Test cases for analysis running

@patch('src.jupyter.kernel.market_data.add_technical_indicators')
@patch('src.jupyter.kernel.market_data.get_performance_summary')
def test_run_analysis(mock_get_performance_summary, mock_add_technical_indicators, mock_stock_data, mock_performance_data):
    """Test run_analysis method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Configure the mocks
    mock_add_technical_indicators.return_value = mock_stock_data
    mock_get_performance_summary.return_value = mock_performance_data
    
    # Create the analyzer with pre-loaded data
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    analyzer.data = mock_stock_data
    
    # Run the analysis
    result = analyzer.run_analysis()
    
    # Verify the technical indicators were added
    mock_add_technical_indicators.assert_called_once_with(mock_stock_data)
    
    # Verify the performance summary was calculated
    mock_get_performance_summary.assert_called_once_with(mock_stock_data)
    
    # Verify the analysis result has the expected structure
    assert 'data' in result
    assert 'performance' in result
    assert 'trading_style' in result
    assert result['data'] is mock_stock_data
    assert result['performance'] is mock_performance_data
    assert result['trading_style'] == trading_styles.MEDIUM_SETTINGS
    
    # Verify the performance was stored
    assert analyzer.performance == mock_performance_data


@patch('src.jupyter.market_analyzer.MarketAnalyzer.fetch_data')
@patch('src.jupyter.kernel.market_data.add_technical_indicators')
@patch('src.jupyter.kernel.market_data.get_performance_summary')
def test_run_analysis_without_data(mock_get_performance_summary, mock_add_technical_indicators, 
                                   mock_fetch_data, mock_stock_data, mock_performance_data):
    """Test run_analysis method when data hasn't been fetched yet."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Configure the mocks
    mock_fetch_data.return_value = mock_stock_data
    mock_add_technical_indicators.return_value = mock_stock_data
    mock_get_performance_summary.return_value = mock_performance_data
    
    # Create the analyzer without data
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    assert analyzer.data is None
    
    # Run the analysis
    result = analyzer.run_analysis()
    
    # Verify data was fetched first
    mock_fetch_data.assert_called_once()
    
    # Verify the rest of the analysis proceeded as expected
    mock_add_technical_indicators.assert_called_once_with(mock_stock_data)
    mock_get_performance_summary.assert_called_once_with(mock_stock_data)
    assert result['data'] is mock_stock_data


# Test cases for visualization generation

@patch('src.analysis.market_analyzer.logger')
@patch('src.analysis.market_data.get_stock_data')
@patch('src.plotting.charts.plot_price_history')
@patch('src.plotting.charts.plot_technical_analysis')
@patch('src.plotting.charts.plot_candlestick')
def test_generate_visualizations(mock_plot_candlestick, mock_plot_technical, 
                                mock_plot_price, mock_get_stock_data, mock_logger, mock_stock_data):
    """Test generate_visualizations method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Create mock figures
    mock_fig1 = MagicMock(spec=go.Figure)
    mock_fig2 = MagicMock(spec=go.Figure)
    mock_fig3 = MagicMock(spec=go.Figure)
    
    # Configure the mocks
    mock_plot_price.return_value = mock_fig1
    mock_plot_technical.return_value = mock_fig2
    mock_plot_candlestick.return_value = mock_fig3
    
    # Create the analyzer with pre-loaded data
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    analyzer.data = mock_stock_data
    
    # Generate visualizations
    figures = analyzer.generate_visualizations()
    
    # Verify the plotting functions were called
    mock_plot_price.assert_called_once_with(mock_stock_data)
    mock_plot_technical.assert_called_once_with(mock_stock_data)
    mock_plot_candlestick.assert_called_once_with(mock_stock_data)
    
    # Verify the returned figures
    assert 'price' in figures
    assert 'technical' in figures
    assert 'candlestick' in figures
    assert figures['price'] is mock_fig1
    assert figures['technical'] is mock_fig2
    assert figures['candlestick'] is mock_fig3


@patch('src.analysis.market_analyzer.logger')
@patch('src.analysis.market_data.get_stock_data')
@patch('src.plotting.charts.plot_price_history')
@patch('src.plotting.charts.plot_technical_analysis')
@patch('src.plotting.charts.plot_candlestick')
def test_generate_visualizations_without_data(mock_plot_candlestick, mock_plot_technical, 
                                             mock_plot_price, mock_get_stock_data, mock_logger, mock_stock_data):
    """Test generate_visualizations method when analysis hasn't been run yet."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Create the analyzer
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    analyzer.data = None
    
    # Configure the mocks
    mock_plot_price.return_value = MagicMock(spec=go.Figure)
    mock_plot_technical.return_value = MagicMock(spec=go.Figure)
    mock_plot_candlestick.return_value = MagicMock(spec=go.Figure)
    
    # Generate visualizations
    analyzer.generate_visualizations()
    
    # Verify analysis was run first
    mock_run_analysis.assert_called_once()
    
    # Verify the plotting functions were called
    mock_plot_price.assert_called_once_with(mock_stock_data)
    mock_plot_technical.assert_called_once_with(mock_stock_data)
    mock_plot_candlestick.assert_called_once_with(mock_stock_data)


# Test cases for summary creation

def test_get_summary(mock_stock_data, mock_performance_data):
    """Test get_summary method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Create the analyzer with pre-loaded data and performance
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    analyzer.data = mock_stock_data
    analyzer.performance = mock_performance_data
    
    # Create a mock for the helper methods
    analyzer._determine_trend = MagicMock(return_value="Uptrend")
    analyzer._summarize_indicators = MagicMock(return_value={
        "rsi": "Neutral",
        "macd": "Bullish",
        "bollinger": "Neutral"
    })
    
    # Get the summary
    summary = analyzer.get_summary()
    
    # Verify the summary contains the expected fields
    assert summary['symbol'] == "AAPL"
    assert summary['timeframe'] == "medium"
    assert summary['current_price'] == mock_stock_data['close'].iloc[-1]
    assert summary['period_return'] == mock_performance_data['total_return_pct']
    assert summary['volatility'] == mock_performance_data['volatility']
    assert summary['trend'] == "Uptrend"
    assert 'indicators' in summary
    assert summary['indicators']['rsi'] == "Neutral"
    assert summary['indicators']['macd'] == "Bullish"
    assert summary['indicators']['bollinger'] == "Neutral"


@patch('src.jupyter.market_analyzer.MarketAnalyzer.run_analysis')
def test_get_summary_without_data(mock_run_analysis, mock_stock_data, mock_performance_data):
    """Test get_summary method when analysis hasn't been run yet."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Create the analyzer without data
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    analyzer.data = None
    
    # Configure the mocks
    mock_run_analysis.return_value = {
        'data': mock_stock_data,
        'performance': mock_performance_data
    }
    analyzer._determine_trend = MagicMock(return_value="Uptrend")
    analyzer._summarize_indicators = MagicMock(return_value={
        "rsi": "Neutral",
        "macd": "Bullish",
        "bollinger": "Neutral"
    })
    
    # Get the summary
    summary = analyzer.get_summary()
    
    # Verify analysis was run first
    mock_run_analysis.assert_called_once()
    
    # Verify the summary contains the expected fields
    assert summary['symbol'] == "AAPL"
    assert summary['timeframe'] == "medium"
    assert summary['current_price'] == mock_stock_data['close'].iloc[-1]


# Test helper methods

def test_determine_trend(mock_stock_data):
    """Test _determine_trend method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Create the analyzer with pre-loaded data
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    
    # Test with uptrend data
    uptrend_data = mock_stock_data.copy()
    uptrend_data['close'] = np.linspace(90, 110, 100)  # steadily increasing
    analyzer.data = uptrend_data
    assert analyzer._determine_trend() == "Uptrend"
    
    # Test with downtrend data
    downtrend_data = mock_stock_data.copy()
    downtrend_data['close'] = np.linspace(110, 90, 100)  # steadily decreasing
    analyzer.data = downtrend_data
    assert analyzer._determine_trend() == "Downtrend"
    
    # Test with sideways data
    sideways_data = mock_stock_data.copy()
    sideways_data['close'] = np.ones(100) * 100  # flat
    analyzer.data = sideways_data
    assert analyzer._determine_trend() == "Sideways"


def test_summarize_indicators(mock_stock_data):
    """Test _summarize_indicators method."""
    from src.jupyter.market_analyzer import MarketAnalyzer
    
    # Create the analyzer with pre-loaded data
    analyzer = MarketAnalyzer(symbol="AAPL", timeframe="medium")
    
    # Add some indicator values to the data
    data = mock_stock_data.copy()
    data['rsi_14'] = 70  # overbought
    data['MACD_12_26_9'] = 2.0  # positive
    data['MACDs_12_26_9'] = 1.0  # signal line
    data['BBL_20_2.0'] = 95  # lower band
    data['BBM_20_2.0'] = 100  # middle band
    data['BBU_20_2.0'] = 105  # upper band
    analyzer.data = data
    
    # Get the indicator summary
    indicators = analyzer._summarize_indicators()
    
    # Verify the summary contains the expected indicators
    assert 'rsi' in indicators
    assert 'macd' in indicators
    assert 'bollinger' in indicators
    
    # Test with different indicator values
    data['rsi_14'] = 30  # oversold
    data['MACD_12_26_9'] = -2.0  # negative
    data['MACDs_12_26_9'] = -1.0  # signal line
    indicators = analyzer._summarize_indicators()
    assert indicators['rsi'] == "Oversold"
    assert indicators['macd'] == "Bearish" 