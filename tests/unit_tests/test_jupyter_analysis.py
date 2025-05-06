import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.jupyter.analysis import get_price_data, run_analysis


class TestJupyterAnalysis(unittest.TestCase):
    """Tests for the Jupyter analysis functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample OHLCV data for testing
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Create sample data
        np.random.seed(42)  # For reproducibility
        close_prices = 30000 + np.random.randn(30).cumsum() * 100
        
        # Create some variation in open, high, low based on close
        open_prices = close_prices + np.random.randn(30) * 50
        high_prices = np.maximum(close_prices, open_prices) + np.abs(np.random.randn(30) * 70)
        low_prices = np.minimum(close_prices, open_prices) - np.abs(np.random.randn(30) * 70)
        volumes = np.random.randint(1000, 5000, 30)
        
        # Create the DataFrame
        self.sample_df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
    
    @patch('src.jupyter.analysis.get_historical_data')
    def test_get_price_data(self, mock_get_historical_data):
        """Test that get_price_data returns correctly formatted DataFrame."""
        # Mock the get_historical_data function to return our sample DataFrame
        mock_get_historical_data.return_value = self.sample_df
        
        # Call the function with test parameters
        result_df = get_price_data(
            symbol="bitcoin",
            days=30,
            timeframe="1d",
            force_refresh=True,  # Force refresh to bypass cache
            vs_currency="usd"
        )
        
        # Verify the result matches our sample data
        pd.testing.assert_frame_equal(result_df, self.sample_df)
        
        # Verify the function was called with correct parameters
        mock_get_historical_data.assert_called_once()
        args, kwargs = mock_get_historical_data.call_args
        self.assertEqual(kwargs["symbol"], "bitcoin")
        self.assertEqual(kwargs["vs_currency"], "usd")
    
    @patch('src.jupyter.analysis.get_price_data')
    @patch('src.services.indicators.calculate_indicator')
    def test_run_analysis(self, mock_calculate_indicator, mock_get_price_data):
        """Test that run_analysis correctly processes indicator data."""
        # Mock the get_price_data function to return our sample DataFrame
        mock_get_price_data.return_value = self.sample_df
        
        # Mock the calculate_indicator function to return sample RSI data
        mock_calculate_indicator.return_value = {
            "value": 65.5,
            "series": {d.isoformat(): 65.5 for d in self.sample_df.index}
        }
        
        # Call the function with test parameters
        result = run_analysis(
            symbol="bitcoin",
            indicator="rsi",
            days=30,
            timeframe="1d",
            force_refresh=True,  # Force refresh to bypass cache
            vs_currency="usd"
        )
        
        # Verify the result contains expected fields
        self.assertIn("value", result)
        self.assertIn("series", result)
        self.assertIn("signal", result)
        self.assertIn("type", result)
        
        # Verify correct signal interpretation (RSI > 30 and < 70 is neutral)
        self.assertEqual(result["signal"], "neutral")
        self.assertEqual(result["type"], "rsi")
        
        # Verify the mock functions were called correctly
        mock_get_price_data.assert_called_once_with(
            "bitcoin", 30, "1d", True, "usd"
        )
        mock_calculate_indicator.assert_called_once()
    
    @patch('src.jupyter.analysis.get_price_data')
    @patch('src.services.indicators.calculate_indicator')
    def test_run_analysis_macd(self, mock_calculate_indicator, mock_get_price_data):
        """Test that run_analysis correctly processes MACD data."""
        # Mock the get_price_data function to return our sample DataFrame
        mock_get_price_data.return_value = self.sample_df
        
        # Mock the calculate_indicator function to return sample MACD data
        mock_calculate_indicator.return_value = {
            "macd_line": 100.5,
            "signal_line": 90.2,
            "histogram": 10.3,
            "value": 10.3  # histogram value
        }
        
        # Call the function with test parameters
        result = run_analysis(
            symbol="bitcoin",
            indicator="macd",
            days=30,
            timeframe="1d",
            force_refresh=True,
            vs_currency="usd"
        )
        
        # Verify the result contains expected fields
        self.assertIn("macd_line", result)
        self.assertIn("signal_line", result)
        self.assertIn("histogram", result)
        self.assertIn("signal", result)
        
        # Verify correct signal interpretation (histogram > 0 is bullish)
        self.assertEqual(result["signal"], "bullish")
        self.assertEqual(result["type"], "macd")


if __name__ == '__main__':
    unittest.main() 