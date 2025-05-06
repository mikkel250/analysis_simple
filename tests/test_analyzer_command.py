"""
Tests for the analyzer CLI command.
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner
from src.main import app


# Create test runner
runner = CliRunner()


@pytest.fixture
def mock_market_analyzer():
    """Create a mock MarketAnalyzer instance."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        # Create mock instance
        mock_instance = MagicMock()
        
        # Configure the mock instance to return test data
        mock_instance.fetch_data.return_value = MagicMock()
        mock_instance.run_analysis.return_value = {
            'data': MagicMock(),
            'performance': {
                'return': 5.2,
                'volatility': 2.3
            }
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': 'bullish',
            'indicators': {
                'sma': 'Price above SMA, suggesting uptrend',
                'rsi': 'RSI at 62, showing moderate bullish momentum',
                'macd': 'MACD crossed above signal line, bullish signal',
                'bollinger': 'Price near upper band, strong trend'
            }
        }
        
        # Configure visualizations
        mock_fig = MagicMock()
        mock_fig.to_html.return_value = "<div>Mock Chart</div>"
        mock_fig.write_html.return_value = None
        
        mock_instance.generate_visualizations.return_value = {
            'price': mock_fig,
            'technical': mock_fig,
            'candlestick': mock_fig
        }
        
        # Set the mock instance as the return value of the class constructor
        mock_analyzer_class.return_value = mock_instance
        
        yield mock_analyzer_class


def test_analyzer_command_exists():
    """Test that the analyzer command exists in the CLI."""
    result = runner.invoke(app, ["analyzer", "--help"])
    assert result.exit_code == 0
    assert "analyzer" in result.stdout
    assert "Market analyzer powered by the MarketAnalyzer class" in result.stdout


def test_analyzer_default_options(mock_market_analyzer):
    """Test analyzer command with default options."""
    result = runner.invoke(app, ["analyzer", "analyze"])
    
    assert result.exit_code == 0
    assert "Analyzing BTC with short timeframe" in result.stdout
    assert "MARKET ANALYSIS: BTC (SHORT TIMEFRAME)" in result.stdout
    assert "Current Price" in result.stdout
    assert "24H change: 5.20%" in result.stdout
    assert "TREND: BULLISH" in result.stdout
    assert "SMA: Price above SMA, suggesting uptrend" in result.stdout
    
    # Check that the correct methods were called
    mock_instance = mock_market_analyzer.return_value
    mock_instance.fetch_data.assert_called_once()
    mock_instance.run_analysis.assert_called_once()
    mock_instance.get_summary.assert_called_once()
    mock_instance.generate_visualizations.assert_not_called()


def test_analyzer_with_timeframe(mock_market_analyzer):
    """Test analyzer command with specified timeframe."""
    result = runner.invoke(app, ["analyzer", "analyze", "--timeframe", "short"])
    
    assert result.exit_code == 0
    assert "Analyzing BTC with short timeframe" in result.stdout
    
    # Check that the analyzer was called with the right parameters
    mock_market_analyzer.assert_called_with(symbol="BTC", timeframe="short")


def test_analyzer_json_output(mock_market_analyzer):
    """Test analyzer command with JSON output."""
    # Update the mock to return a serializable DataFrame
    mock_df = MagicMock()
    mock_df.to_dict.return_value = [{"date": "2023-01-01", "close": 50000}]
    
    # Get the mock instance
    mock_instance = mock_market_analyzer.return_value
    
    # Update the return values for this test
    mock_instance.run_analysis.return_value = {
        'data': mock_df,
        'performance': {
            'return': 5.2,
            'volatility': 2.3
        },
        'trading_style': {
            'name': 'short',
            'intervals': ['1h', '4h', '1d'],
            'periods': ['1mo', '3mo', '6mo']
        }
    }
    
    with patch('src.cli.commands.analyzer.json.dumps') as mock_dumps:
        # Make the mock return a valid JSON string
        mock_dumps.return_value = '{"summary": {"symbol": "BTC", "trend": "bullish"}, "analysis": {"performance": {"return": 5.2}}}'
        
        result = runner.invoke(app, ["analyzer", "analyze", "--output", "json"])
        
        assert result.exit_code == 0
        
        # Check that the json.dumps function was called
        mock_dumps.assert_called_once()
        
        # Directly use the test output
        assert "summary" in result.stdout
        assert "BTC" in result.stdout
        assert "bullish" in result.stdout


def test_analyzer_save_charts(mock_market_analyzer, tmpdir):
    """Test analyzer command with chart saving."""
    # Change to the temporary directory
    original_dir = os.getcwd()
    os.chdir(tmpdir)
    
    try:
        result = runner.invoke(app, ["analyzer", "analyze", "--save-charts"])
        
        assert result.exit_code == 0
        assert "Saved price chart" in result.stdout
        assert "Saved technical chart" in result.stdout
        assert "Saved candlestick chart" in result.stdout
        
        # Check that generate_visualizations was called
        mock_instance = mock_market_analyzer.return_value
        mock_instance.generate_visualizations.assert_called_once()
        
        # Check that write_html was called for each chart
        mock_fig = mock_instance.generate_visualizations.return_value['price']
        assert mock_fig.write_html.call_count == 3
    finally:
        # Restore the original directory
        os.chdir(original_dir)


def test_analyzer_html_output(mock_market_analyzer, tmpdir):
    """Test analyzer command with HTML output."""
    # Change to the temporary directory
    original_dir = os.getcwd()
    os.chdir(tmpdir)
    
    try:
        result = runner.invoke(app, ["analyzer", "analyze", "--output", "html"])
        
        assert result.exit_code == 0
        assert "HTML report saved" in result.stdout
        
        # Check if the HTML file was created
        assert os.path.exists("BTC_short_analysis.html")
    finally:
        # Restore the original directory
        os.chdir(original_dir)


def test_analyzer_explain_option(mock_market_analyzer):
    """Test analyzer command with the explain option."""
    result = runner.invoke(app, ["analyzer", "analyze", "--explain"])
    
    assert result.exit_code == 0
    # The category header is only printed when explain is True
    assert "TECHNICAL INDICATORS" in result.stdout
    
    # Mock the indicator explanation
    with patch('src.cli.commands.analyzer.get_indicator_explanation', 
               return_value="This indicator measures momentum"):
        result = runner.invoke(app, ["analyzer", "analyze", "--explain"])
        assert "This indicator measures momentum" in result.stdout


def test_analyzer_error_handling(mock_market_analyzer):
    """Test analyzer command error handling."""
    # Make the fetch_data method raise an exception
    mock_instance = mock_market_analyzer.return_value
    mock_instance.fetch_data.side_effect = ValueError("Invalid symbol")
    
    result = runner.invoke(app, ["analyzer", "analyze"])
    
    assert "Failed to analyze BTC" in result.stdout
    assert "Invalid symbol" in result.stdout 