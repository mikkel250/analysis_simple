import unittest
from unittest.mock import patch, MagicMock, call # Added call
import typer # For Typer.Exit
import sys # For patching sys.argv

# Attempt to import from the new module structure
try:
    from src.cli.commands.analyzer import analyzer_app # The Typer app instance
    from src.cli.commands.analyzer_modules.common import OutputFormat, TimeframeOption, AnalyzerError
    # If we need to mock MarketAnalyzer instantiation or its methods:
    # from src.analysis.market_analyzer import MarketAnalyzer 
except ImportError:
    print("Error: Could not import analyzer_app or common modules. Ensure PYTHONPATH set.")
    # Fallback definitions for the test to be parsable
    class FakeTyperApp:
        def __call__(self, *args, **kwargs): pass # Mock __call__ for app()
    analyzer_app = FakeTyperApp()
    from enum import Enum
    class OutputFormat(str, Enum): TEXT = "text"; JSON = "json"; HTML = "html"
    class TimeframeOption(str, Enum): SHORT = "short"; MEDIUM = "medium"; LONG = "long"
    class AnalyzerError(Exception): pass

# To use Typer's test client, you usually do: from typer.testing import CliRunner; runner = CliRunner()
# However, the current structure of analyzer.py directly calls functions upon import/setup.
# For now, we will test by patching internals and simulating command-line args via sys.argv and calling a main/entry point if available,
# or directly invoking the command function if that's how analyzer.py is structured.
# The provided analyzer.py uses `analyzer_app = typer.Typer()` and then `@analyzer_app.command()`. 
# This means we should be able to use Typer's test runner.

from typer.testing import CliRunner

runner = CliRunner()

class TestAnalyzerCommand(unittest.TestCase):

    @patch('src.cli.commands.analyzer.MarketAnalyzer')
    @patch('src.cli.commands.analyzer.display_market_analysis')
    def test_analyze_command_success_text_output(self, mock_display_market_analysis, mock_market_analyzer):
        """Test the analyze command with basic valid inputs for TEXT output."""
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"data": "analysis_complete"}
        mock_market_analyzer.return_value = mock_analyzer_instance

        result = runner.invoke(analyzer_app, [
            "AAPL",
            "--timeframe", "short",
            "--output", "text",
            "--explain"
        ])

        self.assertEqual(result.exit_code, 0, f"CLI Errored: {result.stdout} {result.stderr} {result.exc_info}")
        mock_market_analyzer.assert_called_once_with(symbol="AAPL", timeframe_val="short", api_key=None)
        mock_analyzer_instance.analyze.assert_called_once()
        mock_display_market_analysis.assert_called_once_with(
            analysis_results={"data": "analysis_complete"},
            symbol="AAPL",
            timeframe_str="short",
            output_format_enum=OutputFormat.TEXT,
            explain=True,
            save_to_file=False
        )
        self.assertIn("Market analysis for AAPL (short) started.", result.stdout)

    @patch('src.cli.commands.analyzer.MarketAnalyzer')
    @patch('src.cli.commands.analyzer.display_market_analysis')
    def test_analyze_command_success_json_file_output(self, mock_display_market_analysis, mock_market_analyzer):
        """Test the analyze command with JSON output to file."""
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"json_data": "very_good_json"}
        mock_market_analyzer.return_value = mock_analyzer_instance

        result = runner.invoke(analyzer_app, [
            "MSFT",
            "--timeframe", "medium",
            "--output", "jsf" # JSF implies save_to_file=True
        ])

        self.assertEqual(result.exit_code, 0, f"CLI Errored: {result.stdout}")
        mock_market_analyzer.assert_called_once_with(symbol="MSFT", timeframe_val="medium", api_key=None)
        mock_analyzer_instance.analyze.assert_called_once()
        mock_display_market_analysis.assert_called_once_with(
            analysis_results={"json_data": "very_good_json"},
            symbol="MSFT",
            timeframe_str="medium",
            output_format_enum=OutputFormat.JSF,
            explain=False,
            save_to_file=True
        )

    @patch('src.cli.commands.analyzer.MarketAnalyzer')
    @patch('src.cli.commands.analyzer.display_error') # Patching the error display function
    def test_analyze_command_market_analyzer_failure(self, mock_display_error, mock_market_analyzer):
        """Test CLI behavior when MarketAnalyzer.analyze() raises an exception."""
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.side_effect = Exception("Network Error")
        mock_market_analyzer.return_value = mock_analyzer_instance

        result = runner.invoke(analyzer_app, [
            "GOOG",
            "--timeframe", "long",
            "--output", "text"
        ])
        
        self.assertEqual(result.exit_code, 1) # Typer.Exit(1)
        mock_market_analyzer.assert_called_once_with(symbol="GOOG", timeframe_val="long", api_key=None)
        mock_analyzer_instance.analyze.assert_called_once()
        # Check if our display_error was called with the specific message
        mock_display_error.assert_called_once()
        args, _ = mock_display_error.call_args
        self.assertIn("Error during market analysis for GOOG: Network Error", args[0])


    @patch('src.cli.commands.analyzer.MarketAnalyzer') # Still need to mock this, even if not directly used in this error path
    @patch('src.cli.commands.analyzer.display_market_analysis')
    @patch('src.cli.commands.analyzer.display_error')
    def test_analyze_command_display_failure(self, mock_display_error, mock_display_market_analysis, mock_market_analyzer):
        """Test CLI behavior when display_market_analysis raises an AnalyzerError."""
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"data": "analysis_complete"}
        mock_market_analyzer.return_value = mock_analyzer_instance
        mock_display_market_analysis.side_effect = AnalyzerError("Display system crashed")

        result = runner.invoke(analyzer_app, [
            "TSLA",
            "--timeframe", "short",
            "--output", "html"
        ])
        
        self.assertEqual(result.exit_code, 1)
        mock_display_market_analysis.assert_called_once()
        mock_display_error.assert_called_once_with("Error displaying analysis for TSLA: Display system crashed")

    def test_analyze_command_invalid_timeframe(self):
        """Test CLI behavior with an invalid timeframe option."""
        result = runner.invoke(analyzer_app, [
            "NVDA",
            "--timeframe", "invalid_time",
            "--output", "text"
        ])
        self.assertNotEqual(result.exit_code, 0)
        # Typer automatically handles enum validation for choices
        self.assertIn("Invalid value for '--timeframe'", result.stdout) # Typer >= 0.3.0
        # For older Typer versions, stderr might contain the message
        # self.assertIn("invalid choice: invalid_time. (choose from short, medium, long)", result.stderr) 

    def test_analyze_command_invalid_output_format(self):
        """Test CLI behavior with an invalid output format option."""
        result = runner.invoke(analyzer_app, [
            "AMZN",
            "--timeframe", "short",
            "--output", "invalid_format"
        ])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid value for '--output'", result.stdout)
        # self.assertIn("invalid choice: invalid_format. (choose from text, txt, json, jsf, html)", result.stderr)

    # Test for API Key (Optional, but good to have a placeholder if the feature is there)
    @patch('src.cli.commands.analyzer.MarketAnalyzer')
    @patch('src.cli.commands.analyzer.display_market_analysis')
    def test_analyze_command_with_api_key(self, mock_display_market_analysis, mock_market_analyzer):
        """Test that API key is passed to MarketAnalyzer if provided."""
        mock_analyzer_instance = MagicMock()
        mock_analyzer_instance.analyze.return_value = {"data": "api_analysis"}
        mock_market_analyzer.return_value = mock_analyzer_instance

        result = runner.invoke(analyzer_app, [
            "NFLX",
            "--timeframe", "long",
            "--output", "text",
            "--api-key", "TEST_API_KEY_123"
        ])
        self.assertEqual(result.exit_code, 0, f"CLI Errored: {result.stdout}")
        mock_market_analyzer.assert_called_once_with(symbol="NFLX", timeframe_val="long", api_key="TEST_API_KEY_123")
        mock_display_market_analysis.assert_called_once()

if __name__ == '__main__':
    unittest.main() 