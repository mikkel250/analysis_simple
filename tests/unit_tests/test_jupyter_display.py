import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

from src.jupyter.display import create_price_chart, create_indicator_chart


class TestJupyterDisplay(unittest.TestCase):
    """Tests for the Jupyter display visualization functions."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data DataFrame for testing
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # Create sample OHLCV data
        np.random.seed(42)  # For reproducibility
        close_prices = 30000 + np.random.randn(30).cumsum() * 100
        
        # Create some variation in open, high, low based on close
        open_prices = close_prices + np.random.randn(30) * 50
        high_prices = np.maximum(close_prices, open_prices) + np.abs(np.random.randn(30) * 70)
        low_prices = np.minimum(close_prices, open_prices) - np.abs(np.random.randn(30) * 70)
        volumes = np.random.randint(1000, 5000, 30)
        
        # Create the DataFrame
        self.df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        # Create sample indicator data for testing - using ISO format dates
        self.rsi_data = {
            "type": "rsi",
            "value": 65.5,
            "signal": "neutral",
            "series": {
                d.isoformat(): 50 + np.sin(i/5) * 20 
                for i, d in enumerate(dates)
            }
        }
    
    def test_create_price_chart(self):
        """Test that create_price_chart returns a valid Plotly figure with expected traces."""
        # Call the function with test data
        fig = create_price_chart(self.df, "Test Price Chart")
        
        # Verify the result is a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Verify the figure contains the expected traces
        self.assertEqual(len(fig.data), 2)  # Should have candlestick and volume traces
        
        # Verify first trace is a candlestick chart
        self.assertEqual(fig.data[0].type, "candlestick")
        
        # Verify second trace is a bar chart (volume)
        self.assertEqual(fig.data[1].type, "bar")
        
        # Check that the title is included in the figure's subplot titles
        self.assertEqual(fig.layout.annotations[0].text, "Test Price Chart")
    
    def test_create_indicator_chart(self):
        """Test that create_indicator_chart returns a valid Plotly figure with expected traces."""
        # Call the function with test data
        fig = create_indicator_chart(
            self.df, 
            self.rsi_data, 
            "Test RSI Chart",
            include_price=True
        )
        
        # Verify the result is a Plotly figure
        self.assertIsInstance(fig, go.Figure)
        
        # Verify the figure contains the expected traces
        # Should have at least 2 traces: candlestick and RSI marker
        self.assertGreaterEqual(len(fig.data), 2)
        
        # Verify first trace is a candlestick chart
        self.assertEqual(fig.data[0].type, "candlestick")
        
        # Check that the title is included in the figure's subplot titles
        self.assertEqual(fig.layout.annotations[0].text, "Test RSI Chart")


if __name__ == '__main__':
    unittest.main() 