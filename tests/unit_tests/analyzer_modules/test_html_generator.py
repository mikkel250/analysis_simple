import unittest
from unittest.mock import patch, mock_open, MagicMock
import os

# Attempt to import from the new module structure
try:
    from src.cli.commands.analyzer_modules.html_generator import (
        generate_html_report,
        _save_visualizations, # If testing separately, though it's mainly internal to generate_html_report
        # display_info, display_success, display_warning, display_error # These are more like UI helpers
    )
    from src.cli.commands.analyzer_modules.common import OutputFormat # For context if needed
    from src.cli.education import get_indicator_explanation # Mocked
except ImportError:
    print("Error: Could not import html_generator or common module. Ensure PYTHONPATH set.")
    # Fallback definitions
    def generate_html_report(analysis_results: dict, symbol: str, timeframe: str, output_dir: str, explain: bool = False, save_charts: bool = True) -> str: return "<html></html>"
    def _save_visualizations(visualizations: dict, symbol: str, timeframe: str, output_dir: str) -> dict: return {}
    def get_indicator_explanation(indicator_name: str) -> str: return "Mocked explanation"
    class OutputFormat:
        HTML = "html"

class TestHtmlGenerator(unittest.TestCase):

    def setUp(self):
        self.analysis_results_mock = {
            "summary": {
                "symbol": "TEST-USD",
                "timeframe": "short",
                "current_price": 100.0,
                "period_return": 5.5,
                "volatility": 1.2,
                "trend": {
                    "direction": "Up", 
                    "strength": "Strong", 
                    "confidence": "High",
                    "signals": {"short_term": "Buy", "action": "Buy"},
                    "explanation": "Trend is looking good."
                },
                "indicators": {"rsi": "Bullish", "macd": "Bullish Crossover"},
                "indicator_data": {
                    "rsi": {"value": 65.0},
                    "macd": {"values": {"line": 1.0, "signal": 0.8, "histogram": 0.2}}
                }
            },
            "visualizations": {
                "price_chart": MagicMock() # Mock Plotly figure
            },
            "market_cases": { # Added from previous structure for completeness
                'bullish': {'confidence': 'High', 'supporting_indicators': [('RSI', 'Oversold')]}
            }
        }
        self.symbol = "TEST-USD"
        self.timeframe = "short"
        self.output_dir = "/mock_output/html"

    @patch('src.cli.commands.analyzer_modules.html_generator.get_indicator_explanation', return_value="Mocked explanation.")
    @patch('src.cli.commands.analyzer_modules.html_generator._save_visualizations')
    @patch('os.path.exists', return_value=True) # Assume output_dir exists
    @patch('os.makedirs') # To catch if it tries to create image dir
    def test_generate_html_report_structure_and_content(self, mock_makedirs, mock_os_exists, mock_save_viz, mock_get_ind_exp):
        """Test basic structure and content of the generated HTML report."""
        mock_save_viz.return_value = {"price_chart": "/mock_output/html/images/TEST-USD_short_price_chart.png"}
        
        # Mock the .to_html() method of the mock Plotly figure
        self.analysis_results_mock["visualizations"]["price_chart"].to_html.return_value = "<div>Mock Chart HTML</div>"

        html_content = generate_html_report(
            analysis_results=self.analysis_results_mock,
            symbol=self.symbol,
            timeframe=self.timeframe,
            output_dir=self.output_dir,
            explain=True,
            save_charts=True
        )

        self.assertIn("<html>", html_content)
        self.assertIn(f"<title>Market Analysis: {self.symbol} ({self.timeframe.upper()})</title>", html_content)
        self.assertIn(f"<h1>Market Analysis: {self.symbol} ({self.timeframe.upper()})</h1>", html_content)
        self.assertIn("Current Price:</span> $100.00", html_content) # Check for formatted price
        self.assertIn("Trend Analysis", html_content)
        self.assertIn("Direction:</span> Up", html_content)
        self.assertIn("Technical Indicators", html_content)
        self.assertIn("RSI", html_content)
        self.assertIn("Bullish", html_content) # RSI interpretation
        self.assertIn("Value:</span> 65.00", html_content) # RSI value
        self.assertIn("Mocked explanation.", html_content) # Check if explanation is included
        self.assertIn("Charts", html_content)
        self.assertIn("<div>Mock Chart HTML</div>", html_content) # Check for visualization
        
        mock_save_viz.assert_called_once_with(
            self.analysis_results_mock['visualizations'], 
            self.symbol, 
            self.timeframe, 
            self.output_dir
        )
        # Check if images directory would be created if _save_visualizations is called and needs it
        # In the new html_generator, _save_visualizations handles its own image directory creation
        # So, we only need to check if _save_visualizations was called.
        # mock_makedirs.assert_any_call(os.path.join(self.output_dir, 'images')) #This would be inside _save_visualizations

    @patch('os.makedirs')
    @patch('os.path.exists')
    def test_save_visualizations(self, mock_exists, mock_makedirs):
        """Test the _save_visualizations helper function."""
        mock_fig_plotly = MagicMock()
        mock_fig_plotly.to_html.return_value = "plotly_chart_html"
        # For saving, Plotly figs might use write_image or a similar method if saving as static image
        # Let's assume write_html is for embedding, and write_image for saving file if that were the case.
        # The current _save_visualizations in html_generator.py saves plotly charts as .png using fig.write_image.
        # And matplotlib figs using fig.savefig.

        visualizations = {
            "plotly_chart": mock_fig_plotly,
        }
        symbol = "TESTER"
        timeframe = "medium"
        output_dir = "/test_save_viz_output"
        images_dir_path = os.path.join(output_dir, 'images')

        # Scenario 1: Images directory doesn't exist
        mock_exists.return_value = False
        expected_plotly_path = os.path.join(images_dir_path, f"{symbol}_{timeframe}_plotly_chart.png")

        saved_paths = _save_visualizations(visualizations, symbol, timeframe, output_dir)

        mock_makedirs.assert_called_once_with(images_dir_path)
        mock_fig_plotly.write_image.assert_called_once_with(expected_plotly_path)
        self.assertEqual(saved_paths, {"plotly_chart": expected_plotly_path})

        mock_makedirs.reset_mock()
        mock_fig_plotly.reset_mock()
        
        # Scenario 2: Images directory already exists
        mock_exists.return_value = True
        saved_paths = _save_visualizations(visualizations, symbol, timeframe, output_dir)
        mock_makedirs.assert_not_called() # Should not be called if images_dir exists
        mock_fig_plotly.write_image.assert_called_once_with(expected_plotly_path)
        self.assertEqual(saved_paths, {"plotly_chart": expected_plotly_path})

    def test_generate_html_report_no_visualizations(self):
        """Test HTML generation when no visualizations are present."""
        self.analysis_results_mock['visualizations'] = {}
        html_content = generate_html_report(
            analysis_results=self.analysis_results_mock,
            symbol=self.symbol,
            timeframe=self.timeframe,
            output_dir=self.output_dir
        )
        self.assertNotIn("<h2>Charts</h2>", html_content)
        self.assertNotIn("<div>Mock Chart HTML</div>", html_content)

    def test_generate_html_report_no_explain(self):
        """Test HTML generation when explain is False."""
        html_content = generate_html_report(
            analysis_results=self.analysis_results_mock,
            symbol=self.symbol,
            timeframe=self.timeframe,
            output_dir=self.output_dir,
            explain=False
        )
        self.assertNotIn("Mocked explanation.", html_content)


if __name__ == '__main__':
    unittest.main() 