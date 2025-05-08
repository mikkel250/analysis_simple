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


def test_analyzer_command_exists():
    """Test that the analyzer command exists in the CLI."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "--help"])
        assert result.exit_code == 0
        assert "analyzer" in result.stdout
        assert "Market analyzer powered by the MarketAnalyzer class" in result.stdout


def test_analyzer_default_options():
    """Test analyzer command with default options."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze"])
        assert result.exit_code == 0
        assert "MARKET ANALYSIS: BTC-USDT (SHORT - 15m)" in result.stdout
        assert "Current Price: $50,000.00" in result.stdout
        assert "Change over 1w: 5.20%" in result.stdout or "Change over 1w: 5.2%" in result.stdout
        assert "TREND: UPTREND" in result.stdout
        assert "Strength: Strong" in result.stdout
        assert "Confidence: High" in result.stdout
        assert "Recommended Action: BUY" in result.stdout
        assert "SMA: Bullish" in result.stdout


def test_analyzer_with_timeframe():
    """Test analyzer command with specified timeframe."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze", "--timeframe", "short"])
        assert result.exit_code == 0
        assert "MARKET ANALYSIS: BTC-USDT (SHORT - 15m)" in result.stdout


def test_analyzer_json_output():
    """Test analyzer command with JSON output."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze", "--output", "json"])
        assert result.exit_code == 0
        assert '"symbol": "BTC-USDT"' in result.stdout
        assert '"trend"' in result.stdout
        assert '"indicators"' in result.stdout


def test_analyzer_save_charts(tmpdir):
    """Test analyzer command with chart saving."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze", "--save-charts"])
        assert result.exit_code == 0
        assert "Saved price chart" in result.stdout or "Saved technical chart" in result.stdout or "Saved candlestick chart" in result.stdout


def test_analyzer_html_output(tmpdir):
    """Test analyzer command with HTML output."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze", "--output", "html"])
        assert result.exit_code == 0
        assert "html" in result.stdout.lower() or "analysis saved to" in result.stdout.lower()


def test_analyzer_explain_option():
    """Test analyzer command with the explain option."""
    result = runner.invoke(app, ["analyzer", "analyze", "--explain"])
    assert result.exit_code == 0
    assert "TECHNICAL INDICATORS" in result.stdout or "Trend:" in result.stdout


def test_analyzer_error_handling():
    """Test analyzer command error handling."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.fetch_data.side_effect = ValueError("Invalid symbol")
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': float('nan'),
            'period_return': float('nan'),
            'volatility': float('nan'),
            'trend': {
                'direction': '',
                'strength': '',
                'confidence': '',
                'signals': {},
                'explanation': ''
            },
            'indicators': {},
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Neutral',
            'cases': {
                'bullish': {'confidence': '0.0%', 'confidence_raw': 0.0, 'supporting_indicators': []},
                'bearish': {'confidence': '0.0%', 'confidence_raw': 0.0, 'supporting_indicators': []},
                'neutral': {'confidence': '100.0%', 'confidence_raw': 1.0, 'supporting_indicators': []}
            },
            'explanation': 'No valid indicators available for analysis. Defaulting to Neutral.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze"])
        assert "Failed to analyze BTC-USDT" in result.stdout
        assert "Invalid symbol" in result.stdout


def test_analyzer_cli_prints_open_interest_analytics():
    """Test that the CLI prints open interest analytics in the output."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_advanced_analytics.return_value = {
            'volatility_forecast': {},
            'regime': {},
            'strategy_suggestion': {},
            'watch_for_signals': [],
            'open_interest_analysis': {
                'regime': 'spike',
                'summary': 'Sudden OI spike detected, watch for breakout.'
            }
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze"])
        assert result.exit_code == 0
        assert "OPEN INTEREST ANALYTICS" in result.stdout
        assert "Regime: Spike" in result.stdout
        assert "Summary: Sudden OI spike detected, watch for breakout." in result.stdout


def test_analyzer_cli_open_interest_error_handling():
    """Test that the CLI handles errors in open interest analytics gracefully."""
    with patch('src.cli.commands.analyzer.MarketAnalyzer') as mock_analyzer_class:
        mock_instance = mock_analyzer_class.return_value
        mock_instance.symbol = "BTC-USDT"
        mock_instance.timeframe = "short"
        mock_instance.trading_style = {
            'periods': ['1d', '1w', '1mo'],
            'window_sizes': [14, 20, 50],
            'intervals': ['15m', '1h', '1d']
        }
        mock_instance.get_advanced_analytics.return_value = {
            'volatility_forecast': {},
            'regime': {},
            'strategy_suggestion': {},
            'watch_for_signals': [],
            'open_interest_analysis': {
                'regime': 'error',
                'summary': 'Error fetching or analyzing open interest: API unavailable'
            }
        }
        mock_instance.get_summary.return_value = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 50000.0,
            'period_return': 5.2,
            'volatility': 2.3,
            'trend': {
                'direction': 'Uptrend',
                'strength': 'Strong',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Bullish',
                    'action': 'Buy'
                },
                'explanation': 'Test trend explanation.'
            },
            'indicators': {
                'sma': {'interpretation': 'Bullish'},
                'rsi': {'interpretation': 'Neutral'},
                'macd': {'interpretation': 'Bullish'},
                'bollinger': {'interpretation': 'Neutral'}
            },
            'indicator_data': {}
        }
        mock_instance.present_cases.return_value = {
            'overall_sentiment': 'Bullish',
            'cases': {
                'bullish': {'confidence': '80.0%', 'confidence_raw': 0.8, 'supporting_indicators': [('sma', 'Bullish')]},
                'bearish': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []},
                'neutral': {'confidence': '10.0%', 'confidence_raw': 0.1, 'supporting_indicators': []}
            },
            'explanation': 'Bullish case has the highest confidence.'
        }
        mock_instance.get_full_analysis.return_value = {}
        mock_instance.generate_visualizations.return_value = {'price': MagicMock(), 'technical': MagicMock(), 'candlestick': MagicMock()}
        result = runner.invoke(app, ["analyzer", "analyze"])
        assert result.exit_code == 0
        assert "OPEN INTEREST ANALYTICS" in result.stdout
        assert "Regime: Error" in result.stdout
        assert "Error fetching or analyzing open interest: API unavailable" in result.stdout 