import unittest
from unittest.mock import patch, MagicMock
from rich.console import Console
import io

# Attempt to import from the new module structure
try:
    from src.cli.commands.analyzer_modules.text_renderer import format_text_analysis, _clean_text
    # Mocked imports that text_renderer.py uses:
    from src.cli.education import get_indicator_explanation, category_header, get_period_return_explanation, get_volatility_explanation
    from src.cli.display import format_price 
except ImportError:
    print("Error: Could not import text_renderer or its dependencies. Ensure PYTHONPATH set.")
    # Fallback definitions
    def format_text_analysis(analysis_results: dict, symbol: str, timeframe: str, explain: bool = False) -> str: return "Mocked text analysis"
    def _clean_text(text: str) -> str: return text
    def get_indicator_explanation(name): return "Expl for " + name
    def category_header(name): return "== " + name + " =="
    def get_period_return_explanation(val): return "Return expl"
    def get_volatility_explanation(val): return "Volatility expl"
    def format_price(val): return f"${val:.2f}"


class TestTextRenderer(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None # Show full diff on assertion failure
        self.analysis_results_mock = {
            "summary": {
                "symbol": "TXT-TEST",
                "timeframe": "medium",
                "current_price": 250.75,
                "period_return": -2.3,
                "volatility": 0.8,
                "trend": {
                    "direction": "Down", "strength": "Moderate", "confidence": "Medium",
                    "signals": {"short_term": "Sell", "medium_term": "Hold", "long_term": "Wait", "action": "Reduce"},
                    "explanation": "Overall trend is weakening."
                },
                "indicators": {
                    "rsi": "Bearish Divergence", 
                    "sma": "Price below SMA20"
                },
                "indicator_data": {
                    "rsi": {"value": 35.0},
                    "sma": {"sma20": 255.0, "sma50": 260.0, "price_action_sma20": "Below", "price_action_sma50": "Below" }
                }
            },
            "market_cases": { # From original analyzer structure
                'bearish': {'confidence': 'Medium', 'supporting_indicators': [('RSI', 'Overbought'), ('MACD', 'Bearish Cross')]}
            },
            "advanced_analytics": { # New structure for advanced analytics
                "volatility_forecast": {"1d": {"forecast": 1.5, "confidence": "High"}},
                "regime": {"trend_regime": "Choppy", "volatility_regime": "Expanding", "confidence": "Medium"},
                "strategy_suggestion": {"strategy": "Scalp", "educational_rationale": "Rationale here.", "actionable_advice": "Advice here."},
                "breakout_strategy": None, # Or a mock dict if testing that part specifically
                "open_interest_analysis": None, # Or a mock dict
            }
        }
        self.symbol = "TXT-TEST"
        self.timeframe_str = "medium"

    @patch('src.cli.commands.analyzer_modules.text_renderer.get_indicator_explanation')
    @patch('src.cli.commands.analyzer_modules.text_renderer.get_period_return_explanation')
    @patch('src.cli.commands.analyzer_modules.text_renderer.get_volatility_explanation')
    @patch('src.cli.commands.analyzer_modules.text_renderer.category_header', side_effect=lambda x: f"\n== {x.upper()} ==")
    @patch('src.cli.commands.analyzer_modules.text_renderer.format_price', side_effect=lambda x: f"${x:.2f}")
    def test_format_text_analysis_structure_and_content(self, mock_fmt_price, mock_cat_hdr, mock_vol_exp, mock_ret_exp, mock_ind_exp):
        mock_ind_exp.return_value = "Mocked indicator explanation."
        mock_ret_exp.return_value = "Mocked return explanation."
        mock_vol_exp.return_value = "Mocked volatility explanation."

        # Capture Rich Console output
        # The `format_text_analysis` in text_renderer.py uses an internal console and captures its output.
        # So we don't need to capture it here again, just check the returned string.
        
        # Test with explain = True
        text_output_explain = format_text_analysis(self.analysis_results_mock, self.symbol, self.timeframe_str, explain=True)

        self.assertIn("MARKET ANALYSIS: TXT-TEST (MEDIUM)", text_output_explain)
        self.assertIn("PRICE INFORMATION", text_output_explain)
        self.assertIn("Current Price: $250.75", text_output_explain)
        self.assertIn("Change over period: -2.30%", text_output_explain)
        self.assertIn("Volatility: 0.80%", text_output_explain)
        self.assertIn("Mocked return explanation.", text_output_explain)
        self.assertIn("Mocked volatility explanation.", text_output_explain)
        self.assertIn("TREND: Down", text_output_explain)
        self.assertIn("Strength: Moderate", text_output_explain)
        self.assertIn("Overall trend is weakening.", text_output_explain) # Trend explanation
        self.assertIn("TECHNICAL INDICATORS", text_output_explain)
        self.assertIn("== MOMENTUM ==", text_output_explain) # Example, depends on how text_renderer groups
        self.assertIn("RSI: Bearish Divergence", text_output_explain)
        self.assertIn("Value: 35.00", text_output_explain) # RSI value
        self.assertIn("Mocked indicator explanation.", text_output_explain)
        self.assertIn("== TREND ==", text_output_explain)
        self.assertIn("SMA: Price below SMA20", text_output_explain)
        self.assertIn("SMA20: $255.00", text_output_explain)
        self.assertIn("ADVANCED ANALYTICS", text_output_explain)
        self.assertIn("Volatility Forecast:", text_output_explain)
        self.assertIn("1d: 1.50% (Confidence: High)", text_output_explain)
        self.assertIn("Market Regime:", text_output_explain)
        self.assertIn("Trend: Choppy | Volatility: Expanding (Confidence: Medium)", text_output_explain)
        self.assertIn("Strategy Suggestion:", text_output_explain)
        self.assertIn("Strategy: Scalp", text_output_explain)
        self.assertIn("Rationale: Rationale here.", text_output_explain)
        self.assertIn("Advice: Advice here.", text_output_explain)

        # Test with explain = False
        text_output_no_explain = format_text_analysis(self.analysis_results_mock, self.symbol, self.timeframe_str, explain=False)
        self.assertNotIn("Mocked indicator explanation.", text_output_no_explain)
        self.assertNotIn("Mocked return explanation.", text_output_no_explain)
        self.assertNotIn("Mocked volatility explanation.", text_output_no_explain)
        # Trend explanation should still be there if it's part of core trend data
        self.assertIn("Overall trend is weakening.", text_output_no_explain)

    def test_clean_text(self):
        """Test the _clean_text helper function."""
        text_with_newlines = "Hello\nWorld\r\nThis is a test."
        expected_cleaned = "Hello World This is a test."
        self.assertEqual(_clean_text(text_with_newlines), expected_cleaned)

        text_with_tabs_spaces = "Hello\t World  multiple   spaces."
        expected_cleaned_tabs = "Hello World multiple spaces."
        self.assertEqual(_clean_text(text_with_tabs_spaces), expected_cleaned_tabs)

        already_clean = "This is clean."
        self.assertEqual(_clean_text(already_clean), already_clean)

        empty_text = ""
        self.assertEqual(_clean_text(empty_text), empty_text)

        none_text = None
        self.assertEqual(_clean_text(none_text), None)

if __name__ == '__main__':
    unittest.main() 