import unittest
from unittest.mock import patch, mock_open, call
import os
import re
import datetime

# Attempt to import from the new module structure
try:
    from src.cli.commands.analyzer_modules.file_utils import (
        _ensure_output_directory,
        _generate_output_filename,
        _strip_ansi_codes
    )
    from src.cli.commands.analyzer_modules.common import OutputFormat # Needed for _generate_output_filename
except ImportError:
    print("Error: Could not import file_utils or common module. Ensure PYTHONPATH is set correctly or tests are run from project root.")
    # Fallback definitions for the test to be parsable
    def _ensure_output_directory(output_type: str) -> str: return "mock_dir/" + output_type
    def _generate_output_filename(symbol: str, timeframe: str, output_type_enum, base_path: str) -> str: 
        return os.path.join(base_path, f"{symbol}_{timeframe}.{output_type_enum.value}")
    def _strip_ansi_codes(text: str) -> str: return text
    from enum import Enum
    class OutputFormat(str, Enum): TXT = "txt"; JSF = "jsf"; HTML = "html"; TEXT = "text"; JSON = "json"


class TestFileUtils(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.getcwd', return_value='/mock_current_dir')
    def test_ensure_output_directory(self, mock_getcwd, mock_exists, mock_makedirs):
        """Test _ensure_output_directory for creating directories."""
        
        # Scenario 1: Base and type directory do not exist
        mock_exists.side_effect = [False, False] # base_dir, then type_dir
        expected_base_dir = '/mock_current_dir/saved_analysis'
        expected_type_dir = '/mock_current_dir/saved_analysis/txt'
        
        result_path = _ensure_output_directory("txt")
        self.assertEqual(result_path, expected_type_dir)
        mock_exists.assert_any_call(expected_base_dir)
        mock_exists.assert_any_call(expected_type_dir)
        mock_makedirs.assert_has_calls([
            call(expected_base_dir),
            call(expected_type_dir)
        ], any_order=False)
        mock_makedirs.reset_mock()
        mock_exists.reset_mock(side_effect=True)

        # Scenario 2: Base exists, type directory does not
        mock_exists.side_effect = [True, False] # base_dir, then type_dir
        expected_type_dir_json = '/mock_current_dir/saved_analysis/json' # for jsf
        result_path = _ensure_output_directory("jsf")
        self.assertEqual(result_path, expected_type_dir_json)
        mock_makedirs.assert_called_once_with(expected_type_dir_json)
        mock_makedirs.reset_mock()
        mock_exists.reset_mock(side_effect=True)

        # Scenario 3: Both exist
        mock_exists.side_effect = [True, True]
        expected_type_dir_html = '/mock_current_dir/saved_analysis/html'
        result_path = _ensure_output_directory("html")
        self.assertEqual(result_path, expected_type_dir_html)
        mock_makedirs.assert_not_called()
        mock_makedirs.reset_mock()
        mock_exists.reset_mock(side_effect=True)

        # Scenario 4: os.makedirs fails for base_dir (should fallback to cwd)
        mock_exists.side_effect = [False] # base_dir does not exist
        mock_makedirs.side_effect = [OSError("Permission denied"), None] # First call fails
        # The function is designed to fallback to current_dir if base_dir creation fails, 
        # and then it attempts to create type_dir inside this fallback.
        # However, the original code falls back to os.getcwd() for the *return value* if base_dir creation fails
        # Let's test the original code's behavior. The original `_ensure_output_directory` would return `os.getcwd()`
        # if `os.makedirs(base_dir)` fails. This means it doesn't attempt to create the type-specific dir in that case.
        # This test reflects the provided code's logic where it returns `os.getcwd()` if base_dir creation fails.
        # In file_utils.py, if base_dir creation fails, it logs an error and returns `os.getcwd()`
        
        # Reset mocks for clean state
        mock_exists.reset_mock(side_effect=True)
        mock_makedirs.reset_mock(side_effect=True)

        mock_exists.side_effect = [False] # base_dir fails
        mock_makedirs.side_effect = OSError("Cannot create base_dir")
        expected_fallback_path = '/mock_current_dir' # Fallback path
        result_path = _ensure_output_directory("txt")
        self.assertEqual(result_path, expected_fallback_path)
        mock_makedirs.assert_called_once_with(expected_base_dir)
        mock_makedirs.reset_mock(side_effect=True)
        mock_exists.reset_mock(side_effect=True)

        # Scenario 5: os.makedirs fails for type_dir (should fallback to base_dir path)
        mock_exists.side_effect = [True, False] # base_dir exists, type_dir does not
        mock_makedirs.side_effect = OSError("Cannot create type_dir")
        expected_fallback_path_to_base = '/mock_current_dir/saved_analysis'
        result_path = _ensure_output_directory("txt")
        self.assertEqual(result_path, expected_fallback_path_to_base)
        mock_makedirs.assert_called_once_with(expected_type_dir)

    @patch('src.cli.commands.analyzer_modules.file_utils._ensure_output_directory')
    @patch('datetime.datetime')
    def test_generate_output_filename(self, mock_datetime, mock_ensure_output_dir):
        """Test _generate_output_filename for correct name generation."""
        mock_ensure_output_dir.return_value = '/mock_output_dir/txt'
        
        mock_now = datetime.datetime(2023, 10, 26, 14, 30, 55)
        mock_datetime.now.return_value = mock_now
        timestamp_str = "20231026_143055"

        symbol = "BTC-USDT"
        timeframe = "short"
        output_type = OutputFormat.TXT

        expected_filename = f"/mock_output_dir/txt/{symbol.replace('-','_')}_{timeframe}_{timestamp_str}.txt"
        
        # Note: _generate_output_filename in file_utils.py takes output_format_enum and output_dir_path (which is base_path here)
        # The stub in this test file is simplified. The actual call will be:
        # _generate_output_filename(symbol, timeframe, output_type_enum, base_path_for_type_dir)
        # The mock_ensure_output_dir is for the directory where the file will be created.
        # The _generate_output_filename takes the *base_path* which is usually like 'saved_analysis/txt' etc.
        # Let's adjust the test to match file_utils.py version where _ensure_output_directory is called *inside* it.
        # So we don't mock _ensure_output_directory for the actual call of _generate_output_filename.
        # Instead, _generate_output_filename calls _ensure_output_directory itself.

        # Re-patch os.makedirs and os.path.exists for the _ensure_output_directory call inside _generate_output_filename
        with patch('os.makedirs'), patch('os.path.exists', return_value=True), patch('os.getcwd', return_value='/abs'):
            # The mock_ensure_output_dir will be called by the SUT (_generate_output_filename)
            # We need to make sure that the `_ensure_output_directory` inside `file_utils.py` behaves as expected.
            # For _generate_output_filename, it expects OutputFormat enum member for output_type
            # The `_generate_output_filename` in `file_utils` takes `output_dir_path` as an argument.

            # Let's assume the output_dir_path is correctly determined and passed.
            mock_output_dir_path = '/abs/saved_analysis/txt' # This would be returned by _ensure_output_directory normally
            
            generated_filename = _generate_output_filename(symbol, timeframe, OutputFormat.TXT, mock_output_dir_path)
            expected_filename = os.path.join(mock_output_dir_path, f"{symbol.replace('-','_')}_{timeframe}_{timestamp_str}.txt")
            self.assertEqual(generated_filename, expected_filename)

        # Test with different type (HTML)
        mock_output_dir_path_html = '/abs/saved_analysis/html'
        with patch('os.makedirs'), patch('os.path.exists', return_value=True), patch('os.getcwd', return_value='/abs'):
            generated_filename_html = _generate_output_filename("ETH/USD", "long", OutputFormat.HTML, mock_output_dir_path_html)
            expected_filename_html = os.path.join(mock_output_dir_path_html, f"ETH_USD_long_{timestamp_str}.html")
            self.assertEqual(generated_filename_html, expected_filename_html)


    def test_strip_ansi_codes(self):
        """Test _strip_ansi_codes for removing ANSI escape sequences."""
        ansi_text = "\033[1;31mHello\033[0m, \x1b[32mWorld\x1b[0m!"
        plain_text = "Hello, World!"
        self.assertEqual(_strip_ansi_codes(ansi_text), plain_text)

        text_with_other_codes = "Text with [34mcolor[0m and \x1b[1mbold\x1b[0m."
        expected_plain = "Text with color and bold."
        # The regex in file_utils might be more aggressive, let's test that one's behavior
        # Original regexes: r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])' and r'\[[0-9;]*[mGKH]'
        # If _strip_ansi_codes is the one from file_utils.py, it should handle these.
        self.assertEqual(_strip_ansi_codes(text_with_other_codes), expected_plain)
        
        already_plain = "This is plain text."
        self.assertEqual(_strip_ansi_codes(already_plain), already_plain)

        empty_string = ""
        self.assertEqual(_strip_ansi_codes(empty_string), empty_string)

if __name__ == '__main__':
    unittest.main() 