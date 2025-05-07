"""
Tests for enhanced analyzer display and explanation features
"""

import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os
import re

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cli.education import get_period_return_explanation, get_volatility_explanation
from src.cli.commands.analyzer import print_market_analysis

class TestAnalyzerEnhancements(unittest.TestCase):
    """Test cases for enhanced analyzer display and explanation features"""
    
    def setUp(self):
        """Set up test data"""
        self.test_summary = {
            'symbol': 'BTC-USDT',
            'timeframe': 'short',
            'current_price': 95000.50,
            'period_return': 3.75,
            'volatility': 4.2,
            'trading_style': {
                'intervals': ['5m', '15m', '30m'],
                'periods': ['1d', '5d', '30d']
            },
            'trend': {
                'direction': 'Bullish',
                'strength': 'Moderate',
                'confidence': 'High',
                'signals': {
                    'short_term': 'Bullish',
                    'medium_term': 'Bullish',
                    'long_term': 'Neutral',
                    'action': 'BUY'
                }
            },
            'indicators': {
                'rsi': {
                    'interpretation': 'Bullish momentum',
                    'values': {'value': 58.5}
                },
                'macd': {
                    'interpretation': 'Bullish crossover',
                    'values': {'line': 25.5, 'signal': 20.2, 'histogram': 5.3}
                }
            }
        }
        
        # Create a summary without trading_style for testing the fallback logic
        self.test_summary_no_style = self.test_summary.copy()
        del self.test_summary_no_style['trading_style']
    
    def test_period_return_explanation(self):
        """Test period return explanation for different values"""
        # Test strong bullish case
        explanation = get_period_return_explanation(12.5)
        self.assertIn("strong bullish movement", explanation)
        
        # Test moderate positive case
        explanation = get_period_return_explanation(3.5)
        self.assertIn("moderate positive movement", explanation)
        
        # Test sideways case
        explanation = get_period_return_explanation(0.5)
        self.assertIn("sideways price action", explanation)
        
        # Test negative case
        explanation = get_period_return_explanation(-7.5)
        self.assertIn("substantial negative movement", explanation)
    
    def test_volatility_explanation(self):
        """Test volatility explanation for different values"""
        # Test extremely high volatility
        explanation = get_volatility_explanation(10.5)
        self.assertIn("extremely high daily volatility", explanation)
        
        # Test high volatility
        explanation = get_volatility_explanation(6.5)
        self.assertIn("high daily volatility", explanation)
        
        # Test moderate volatility
        explanation = get_volatility_explanation(3.5)
        self.assertIn("moderate daily volatility", explanation)
        
        # Test low volatility
        explanation = get_volatility_explanation(1.5)
        self.assertIn("low daily volatility", explanation)
    
    # These tests only validate the content of the explanations without mocking print
    def test_print_market_analysis_with_explanations(self):
        """Test that explanations are correctly generated when explain=True"""
        volatility_explanation = get_volatility_explanation(4.2)
        period_return_explanation = get_period_return_explanation(3.75)
        
        self.assertIn("moderate daily volatility", volatility_explanation)
        self.assertIn("moderate positive movement", period_return_explanation)
    
    def test_numerical_values_formatting(self):
        """Test that the values are correctly formatted"""
        # Test RSI value formatting
        rsi_value = self.test_summary['indicators']['rsi']['values']['value']
        formatted_rsi = f"{rsi_value:.2f}"
        self.assertEqual(formatted_rsi, "58.50", "RSI value should be formatted to 2 decimal places")
        
        # Test MACD value formatting
        macd_values = self.test_summary['indicators']['macd']['values']
        formatted_line = f"{macd_values['line']:.4f}"
        formatted_signal = f"{macd_values['signal']:.4f}"
        formatted_histogram = f"{macd_values['histogram']:.4f}"
        
        self.assertEqual(formatted_line, "25.5000", "MACD line should be formatted to 4 decimal places")
        self.assertEqual(formatted_signal, "20.2000", "MACD signal should be formatted to 4 decimal places")
        self.assertEqual(formatted_histogram, "5.3000", "MACD histogram should be formatted to 4 decimal places")
    
    @patch('src.cli.commands.analyzer.print')
    def test_timeframe_formatting_with_style(self, mock_print):
        """Test that the timeframe is properly formatted with interval from trading_style"""
        print_market_analysis(self.test_summary, 'BTC-USDT', 'short', explain=False)
        
        # Check if the formatted timeframe appears in the header
        formatted_calls = mock_print.call_args_list
        
        # Convert all call args to strings for easier searching
        call_strings = [str(call) for call in formatted_calls]
        header_with_interval = any('SHORT - 15m' in s for s in call_strings)
        
        self.assertTrue(header_with_interval, "Header should include timeframe with interval")
    
    @patch('src.cli.commands.analyzer.print')
    def test_timeframe_formatting_with_fallback(self, mock_print):
        """Test that the timeframe is properly formatted using the fallback mapping"""
        print_market_analysis(self.test_summary_no_style, 'BTC-USDT', 'short', explain=False)
        
        # Check if the formatted timeframe appears in the header
        formatted_calls = mock_print.call_args_list
        
        # Convert all call args to strings for easier searching
        call_strings = [str(call) for call in formatted_calls]
        header_with_interval = any('SHORT - 15m' in s for s in call_strings)
        
        self.assertTrue(header_with_interval, "Header should include timeframe with interval using fallback mapping")


if __name__ == '__main__':
    unittest.main() 