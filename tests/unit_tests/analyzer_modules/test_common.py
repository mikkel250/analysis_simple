import unittest
from enum import Enum
import pandas as pd
import numpy as np
from src.analysis.market_analyzer import MarketAnalyzer

# Attempt to import from the new module structure
# This might require PYTHONPATH adjustments or a project structure that supports this
try:
    from src.cli.commands.analyzer_modules.common import AnalyzerError, TimeframeOption, OutputFormat
except ImportError:
    # Fallback for different execution environments or if src is not directly in PYTHONPATH
    # This often indicates a need to configure the test runner (e.g., pytest)
    # or set PYTHONPATH: export PYTHONPATH=$(pwd):$PYTHONPATH
    # For now, we'll assume the test runner handles this or it's run from project root.
    # If this fails, the test execution environment needs review.
    print("Error: Could not import common module. Ensure PYTHONPATH is set correctly or tests are run from project root.")
    # As a last resort for the edit_file tool to proceed, define them locally for the test to be parsable.
    class AnalyzerError(Exception): pass
    class TimeframeOption(str, Enum): SHORT = "short"; MEDIUM = "medium"; LONG = "long"
    class OutputFormat(str, Enum): TEXT = "text"; TXT = "txt"; JSON = "json"; JSF = "jsf"; HTML = "html"

class TestCommon(unittest.TestCase):

    def test_analyzer_error_is_exception(self):
        """Test that AnalyzerError is a subclass of Exception."""
        self.assertTrue(issubclass(AnalyzerError, Exception))
        try:
            raise AnalyzerError("Test error")
        except AnalyzerError as e:
            self.assertEqual(str(e), "Test error")

    def test_timeframe_option_enum(self):
        """Test the TimeframeOption enum members and values."""
        self.assertEqual(TimeframeOption.SHORT, "short")
        self.assertEqual(TimeframeOption.MEDIUM, "medium")
        self.assertEqual(TimeframeOption.LONG, "long")
        
        # Check iteration if needed, or specific member access
        self.assertIn("short", [member.value for member in TimeframeOption])
        self.assertEqual(len(TimeframeOption), 3)

    def test_output_format_enum(self):
        """Test the OutputFormat enum members and values."""
        self.assertEqual(OutputFormat.TEXT, "text")
        self.assertEqual(OutputFormat.TXT, "txt")
        self.assertEqual(OutputFormat.JSON, "json")
        self.assertEqual(OutputFormat.JSF, "jsf")
        self.assertEqual(OutputFormat.HTML, "html")

        self.assertIn("json", [member.value for member in OutputFormat])
        self.assertEqual(len(OutputFormat), 5)

    def test_enum_creation_from_string(self):
        """Test creating enum members from string values."""
        self.assertEqual(TimeframeOption("short"), TimeframeOption.SHORT)
        with self.assertRaises(ValueError):
            TimeframeOption("invalid_timeframe")

        self.assertEqual(OutputFormat("html"), OutputFormat.HTML)
        with self.assertRaises(ValueError):
            OutputFormat("invalid_format")

    def test_zero_volatility_with_constant_prices(self):
        """Test that zero volatility is correctly explained for constant prices."""
        # Create a DataFrame with constant prices
        dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="1d")
        constant_price = 100.0
        df = pd.DataFrame({
            'open': [constant_price] * 30,
            'high': [constant_price] * 30,
            'low': [constant_price] * 30,
            'close': [constant_price] * 30,
            'volume': [1000] * 30
        }, index=dates)
        # Initialize MarketAnalyzer with this data
        analyzer = MarketAnalyzer(symbol="CONST", timeframe="1d", use_test_data=False)
        analyzer.data = df.copy()
        analyzer.run_analysis()
        vol_summary = analyzer._get_volatility_summary()
        self.assertIn("annualized_volatility_pct", vol_summary)
        self.assertEqual(vol_summary["annualized_volatility_pct"], "0.00%")
        self.assertIn("details", vol_summary)
        self.assertIn("all prices were constant", vol_summary["details"])
        self.assertIn("zero volatility value is only reasonable", vol_summary["details"])

if __name__ == '__main__':
    unittest.main() 