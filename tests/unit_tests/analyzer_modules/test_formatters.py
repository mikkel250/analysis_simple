import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import json

# Attempt to import from the new module structure
try:
    from src.cli.commands.analyzer_modules.formatters import display_market_analysis
    from src.cli.commands.analyzer_modules.common import OutputFormat, AnalyzerError
    # These are also defined in formatters.py, but we might want to test their usage via formatters
    # from src.cli.commands.analyzer_modules.formatters import display_success, display_warning # etc.
except ImportError:
    print("Error: Could not import formatters or common module. Ensure PYTHONPATH set.")
    # Fallback definitions
    from enum import Enum
    class OutputFormat(str, Enum): TEXT = "text"; TXT = "txt"; JSON = "json"; JSF = "jsf"; HTML = "html"
    class AnalyzerError(Exception): pass
    def display_market_analysis(analysis_results: dict, symbol: str, timeframe_str: str, output_format_enum: OutputFormat, explain: bool = False, save_to_file: bool = False): pass


class TestFormatters(unittest.TestCase):

    def setUp(self):
        self.analysis_results_mock = {"summary": {"data": "mock_data"}, "visualizations": {}}
        self.symbol = "TESTER"
        self.timeframe = "test_timeframe"
        self.explain = False

    @patch('src.cli.commands.analyzer_modules.formatters.preprocess_for_json')
    @patch('src.cli.commands.analyzer_modules.formatters.NumpyEncoder')
    @patch('src.cli.commands.analyzer_modules.formatters.json.dumps')
    @patch('src.cli.commands.analyzer_modules.formatters.Console')
    def test_display_market_analysis_json_console(self, mock_console_rich, mock_json_dumps, mock_numpy_encoder, mock_preprocess):
        """Test JSON output to console."""
        mock_preprocess.return_value = {"processed": "data"}
        mock_json_dumps.return_value = '{"json": "output"}'
        
        display_market_analysis(
            self.analysis_results_mock, self.symbol, self.timeframe, 
            OutputFormat.JSON, self.explain, save_to_file=False
        )
        
        mock_preprocess.assert_called_once_with(self.analysis_results_mock)
        mock_json_dumps.assert_called_once_with({"processed": "data"}, indent=2, cls=mock_numpy_encoder)
        mock_console_rich.return_value.print.assert_called_once_with('{"json": "output"}')

    @patch('src.cli.commands.analyzer_modules.formatters._ensure_output_directory')
    @patch('src.cli.commands.analyzer_modules.formatters._generate_output_filename')
    @patch('src.cli.commands.analyzer_modules.formatters.preprocess_for_json')
    @patch('src.cli.commands.analyzer_modules.formatters.json.dumps')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.cli.commands.analyzer_modules.formatters.display_success')
    def test_display_market_analysis_jsf_file(self, mock_display_success, mock_file_open, mock_json_dumps, mock_preprocess, mock_gen_fname, mock_ensure_dir):
        """Test JSF (JSON to file) output."""
        mock_ensure_dir.return_value = "/mock/json_output_dir"
        mock_gen_fname.return_value = "/mock/json_output_dir/test.json"
        mock_preprocess.return_value = {"processed": "data"}
        mock_json_dumps.return_value = '{"json": "file_output"}'

        display_market_analysis(
            self.analysis_results_mock, self.symbol, self.timeframe,
            OutputFormat.JSF, self.explain, save_to_file=True
        )
        mock_ensure_dir.assert_called_once_with(OutputFormat.JSF.value)
        mock_gen_fname.assert_called_once_with(self.symbol, self.timeframe, OutputFormat.JSF, "/mock/json_output_dir")
        mock_preprocess.assert_called_once_with(self.analysis_results_mock)
        mock_json_dumps.assert_called_once_with({"processed": "data"}, indent=2, cls=unittest.mock.ANY) # NumpyEncoder
        mock_file_open.assert_called_once_with("/mock/json_output_dir/test.json", 'w', encoding='utf-8')
        mock_file_open().write.assert_called_once_with('{"json": "file_output"}')
        mock_display_success.assert_called_once()

    @patch('src.cli.commands.analyzer_modules.formatters.format_text_analysis')
    @patch('src.cli.commands.analyzer_modules.formatters.Console')
    def test_display_market_analysis_text_console(self, mock_console_rich, mock_format_text):
        """Test TEXT output to console."""
        mock_format_text.return_value = "Formatted text output"
        
        display_market_analysis(
            self.analysis_results_mock, self.symbol, self.timeframe,
            OutputFormat.TEXT, self.explain, save_to_file=False
        )
        mock_format_text.assert_called_once_with(self.analysis_results_mock, self.symbol, self.timeframe, self.explain)
        mock_console_rich.return_value.print.assert_called_once_with("Formatted text output")

    @patch('src.cli.commands.analyzer_modules.formatters._ensure_output_directory')
    @patch('src.cli.commands.analyzer_modules.formatters._generate_output_filename')
    @patch('src.cli.commands.analyzer_modules.formatters.format_text_analysis')
    @patch('src.cli.commands.analyzer_modules.formatters._strip_ansi_codes')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.cli.commands.analyzer_modules.formatters.display_success')
    def test_display_market_analysis_txt_file(self, mock_display_success, mock_file_open, mock_strip_ansi, mock_format_text, mock_gen_fname, mock_ensure_dir):
        """Test TXT (Text to file) output."""
        mock_ensure_dir.return_value = "/mock/txt_output_dir"
        mock_gen_fname.return_value = "/mock/txt_output_dir/test.txt"
        mock_format_text.return_value = "Formatted text with ANSI"
        mock_strip_ansi.return_value = "Cleaned text for file"

        display_market_analysis(
            self.analysis_results_mock, self.symbol, self.timeframe,
            OutputFormat.TXT, self.explain, save_to_file=True
        )
        mock_ensure_dir.assert_called_once_with(OutputFormat.TXT.value)
        mock_gen_fname.assert_called_once_with(self.symbol, self.timeframe, OutputFormat.TXT, "/mock/txt_output_dir")
        mock_format_text.assert_called_once_with(self.analysis_results_mock, self.symbol, self.timeframe, self.explain)
        mock_strip_ansi.assert_called_once_with("Formatted text with ANSI")
        mock_file_open.assert_called_once_with("/mock/txt_output_dir/test.txt", 'w', encoding='utf-8')
        mock_file_open().write.assert_called_once_with("Cleaned text for file")
        mock_display_success.assert_called_once()

    @patch('src.cli.commands.analyzer_modules.formatters._ensure_output_directory')
    @patch('src.cli.commands.analyzer_modules.formatters._generate_output_filename')
    @patch('src.cli.commands.analyzer_modules.formatters.generate_html_report')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.cli.commands.analyzer_modules.formatters.display_success')
    @patch('src.cli.commands.analyzer_modules.formatters.webbrowser.open')
    @patch('os.path.abspath', side_effect=lambda x: x) # Mock abspath to return input
    def test_display_market_analysis_html_file(self, mock_abspath, mock_webbrowser_open, mock_display_success, mock_file_open, mock_gen_html, mock_gen_fname, mock_ensure_dir):
        """Test HTML output to file and opening in browser."""
        mock_ensure_dir.return_value = "/mock/html_output_dir"
        mock_gen_fname.return_value = "/mock/html_output_dir/test.html"
        mock_gen_html.return_value = "<html>Generated HTML content</html>"

        display_market_analysis(
            self.analysis_results_mock, self.symbol, self.timeframe,
            OutputFormat.HTML, self.explain, save_to_file=True # HTML always saves
        )
        
        # HTML first ensures directory (can be called twice if save_to_file=False initially, then determined as True)
        mock_ensure_dir.assert_any_call(OutputFormat.HTML.value)
        
        mock_gen_fname.assert_called_once_with(self.symbol, self.timeframe, OutputFormat.HTML, "/mock/html_output_dir")
        mock_gen_html.assert_called_once_with(
            analysis_results=self.analysis_results_mock,
            symbol=self.symbol,
            timeframe=self.timeframe,
            output_dir="/mock/html_output_dir", # Passed for image saving within html_generator
            explain=self.explain,
            save_charts=True # Default for generate_html_report
        )
        mock_file_open.assert_called_once_with("/mock/html_output_dir/test.html", 'w', encoding='utf-8')
        mock_file_open().write.assert_called_once_with("<html>Generated HTML content</html>")
        mock_display_success.assert_called_once()
        mock_webbrowser_open.assert_called_once_with("file:///mock/html_output_dir/test.html")

    def test_display_market_analysis_unsupported_format(self):
        """Test error handling for unsupported output format."""
        class MockOutputFormat(Enum):
            UNSUPPORTED = "unsupported"
        
        with self.assertRaisesRegex(AnalyzerError, "Unsupported output format: unsupported"):
            display_market_analysis(
                self.analysis_results_mock, self.symbol, self.timeframe,
                MockOutputFormat.UNSUPPORTED, self.explain, save_to_file=False
            )
            
    @patch('src.cli.commands.analyzer_modules.formatters._ensure_output_directory', return_value=None)
    def test_display_market_analysis_dir_creation_failure(self, mock_ensure_dir):
        """Test error handling if output directory creation fails."""
        with self.assertRaisesRegex(AnalyzerError, "Could not create or access output directory for jsf"):
            display_market_analysis(
                self.analysis_results_mock, self.symbol, self.timeframe,
                OutputFormat.JSF, self.explain, save_to_file=True
            )

if __name__ == '__main__':
    unittest.main() 