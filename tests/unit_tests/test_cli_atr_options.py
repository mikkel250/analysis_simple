import unittest
from unittest.mock import patch, mock_open, MagicMock
import sys
import io
import pandas as pd
import argparse # For ArgumentError

# Add src to path to allow direct import of cli
# This might be better handled by PYTHONPATH in the test execution command
import os
# Calculate the path to the 'src' directory relative to this test file's location
# tests/unit_tests/test_cli_atr_options.py -> tests/unit_tests -> tests -> project_root
# Then project_root/src
# More robustly, it's better if the test runner handles PYTHONPATH
# For now, let's assume PYTHONPATH is set up correctly or use a simpler relative path if possible.
# If src.cli cannot be found, this will need adjustment. A common pattern is:
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
# However, modifying sys.path directly in tests is sometimes frowned upon.
# Let's assume for now that the execution environment handles src module visibility.
from src import cli # This imports src/cli.py

class TestCliAtrOptions(unittest.TestCase):

    def setUp(self):
        # Redirect stdout and stderr to capture print and error messages
        self.mock_stdout = io.StringIO()
        self.mock_stderr = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self.mock_stdout
        sys.stderr = self.mock_stderr

        # Mock logger to prevent console output during tests, and allow assertions on log messages
        self.patcher_logger = patch('src.cli.logger')
        self.mock_logger = self.patcher_logger.start()
        
        # It's also good to mock functions that interact with the filesystem or external services by default
        # if they are not the specific target of a test.

    def tearDown(self):
        # Restore stdout and stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.mock_stdout.close()
        self.mock_stderr.close()
        self.patcher_logger.stop()


    @patch('src.cli.get_historical_data')
    @patch('src.cli.get_indicator')
    @patch('src.cli.calculate_atr_trade_parameters')
    def test_atr_from_symbol_success(self, mock_calculate_params, mock_get_indicator, mock_get_historical_data):
        # --- Arrange ---
        # Mock get_historical_data
        mock_df = pd.DataFrame({
            'open': [10, 11, 12, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'high': [11, 12, 13, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
            'low': [9, 10, 11, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'close': [10, 12, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 20]
        })
        mock_get_historical_data.return_value = mock_df

        # Mock get_indicator
        # The 'values' can be a Series or a dict. src.cli.py converts dict to Series.
        # Let's assume it returns a dict {'timestamp': value} which pandas_ta might do for some indicators
        # or a Series. The cli.py code handles pd.Series(indicator_output['values'])
        # Let's make it a series directly for simplicity here if get_indicator wrapper standardizes it.
        # Based on cli.py: indicator_output = get_indicator(...); atr_series = pd.Series(indicator_output['values'])
        # So, get_indicator returns a dict like: {'values': pd.Series([...]), 'metadata': ...}
        # Actually, src.services.indicators.get_indicator seems to return a dict like:
        # { 'name': ..., 'values': {pd.Timestamp: value, ...}, 'metadata': ... }

        mock_atr_series = pd.Series([1.0, 1.1, 1.05, 1.12, 1.15, 1.2, 1.22, 1.25, 1.3, 1.28, 1.32, 1.35, 1.4, 1.38])
        # We need to mock the structure that get_indicator returns, which is then processed.
        # It seems cli.py expects indicator_output['values'] to be the series or dict of values.
        mock_get_indicator.return_value = {'values': mock_atr_series, 'metadata': {'params': {'length': 14}}}
        
        # Mock calculate_atr_trade_parameters to verify it's called correctly
        mock_calculate_params.return_value = {
            'position_size': 100.0,
            'stop_loss_percentage': 5.0
        }

        # Prepare sys.argv
        test_args = [
            'src/cli.py', 
            '100', 'long', 
            '--atr-from-symbol', 'BTC-USD', 
            '--atr-period', '14', 
            '--atr-days', '30', # Needs to be >= atr_period
            '--risk_amount', '50'
        ]

        # --- Act ---
        with patch.object(sys, 'argv', test_args):
            # We expect parser.error to call sys.exit.
            # If main() completes without sys.exit, it means success for this path.
            # For error cases, we'll assert SystemExit.
            try:
                cli.main()
            except SystemExit as e:
                # If sys.exit is called, print stderr for debugging the test
                # and then re-raise to fail the test, unless we are specifically testing an error exit.
                # For a success case, SystemExit should not occur.
                print(f"Sys.exit called unexpectedly in success test: {e.code}")
                print(f"Stderr: {self.mock_stderr.getvalue()}")
                raise # This will fail the test

        # --- Assert ---
        # Check that get_historical_data was called correctly
        mock_get_historical_data.assert_called_once_with(
            symbol='BTC-USD', 
            timeframe='1d', 
            limit=30, # From --atr-days
            vs_currency='usd' # Default
        )

        # Check that get_indicator was called correctly
        # The DataFrame passed to get_indicator will have lowercase columns
        pd.testing.assert_frame_equal(mock_get_indicator.call_args[0][0], mock_df.rename(columns=str.lower))
        self.assertEqual(mock_get_indicator.call_args[1]['indicator'], 'atr')
        self.assertEqual(mock_get_indicator.call_args[1]['params'], {'length': 14})


        # Check that calculate_atr_trade_parameters was called with the determined ATR
        # The last value of the mock_atr_series is 1.38
        expected_atr = 1.38 
        mock_calculate_params.assert_called_once()
        call_args_to_calc = mock_calculate_params.call_args[0] # Positional arguments
        self.assertEqual(call_args_to_calc[0], 100.0) # entry_price
        self.assertAlmostEqual(call_args_to_calc[1], expected_atr, places=5) # atr
        self.assertEqual(call_args_to_calc[2], 'long') # position_type
        # Keyword arguments check
        call_kwargs_to_calc = mock_calculate_params.call_args[1]
        self.assertEqual(call_kwargs_to_calc['risk_amount'], 50.0)
        self.assertEqual(call_kwargs_to_calc['multiplier'], 1.0) # Default

        # Check stdout for expected informational messages
        output = self.mock_stdout.getvalue()
        self.assertIn(f"Calculating ATR for symbol: BTC-USD, days: 30, period: 14, vs_currency: usd", output)
        self.assertIn(f"Calculated ATR from symbol BTC-USD: {expected_atr:.8f}", output)
        self.assertIn("--- ATR Position Sizing Tool ---", output)
        
        # Check that no errors were printed to stderr
        # self.assertEqual(self.mock_stderr.getvalue(), '') # parser.error prints to stderr then exits.
                                                            # If we are here, it means no SystemExit.

    @patch('src.cli.get_historical_data')
    def test_atr_from_symbol_failure_insufficient_data(self, mock_get_historical_data):
        # --- Arrange ---
        # Mock get_historical_data to return an empty DataFrame
        mock_get_historical_data.return_value = pd.DataFrame()

        test_args = [
            'src/cli.py', 
            '100', 'long', 
            '--atr-from-symbol', 'BTC-FAIL', 
            '--atr-period', '14', 
            '--atr-days', '10' # Less than period
        ]

        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        
        self.assertEqual(cm.exception.code, 2) # argparse.error calls sys.exit(2)
        output_err = self.mock_stderr.getvalue()
        self.assertIn("Not enough historical data fetched for BTC-FAIL to calculate ATR with period 14", output_err)

    @patch('src.cli.get_historical_data')
    def test_atr_from_symbol_failure_api_error(self, mock_get_historical_data):
        # --- Arrange ---
        mock_get_historical_data.side_effect = Exception("Simulated API Error")

        test_args = [
            'src/cli.py', 
            '100', 'long', 
            '--atr-from-symbol', 'BTC-API-ERR'
        ]

        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        
        self.assertEqual(cm.exception.code, 2)
        output_err = self.mock_stderr.getvalue()
        self.assertIn("Error calculating ATR from symbol BTC-API-ERR: Simulated API Error", output_err)

    @patch('src.cli.get_historical_data')
    @patch('src.cli.get_indicator')
    def test_atr_from_symbol_failure_indicator_returns_none(self, mock_get_indicator, mock_get_historical_data):
        # --- Arrange ---
        mock_df = pd.DataFrame({
            'open': [10]*20, 'high': [11]*20, 'low': [9]*20, 'close': [10]*20
        })
        mock_get_historical_data.return_value = mock_df
        mock_get_indicator.return_value = None # Simulate indicator calculation failure

        test_args = [
            'src/cli.py', '100', 'long', '--atr-from-symbol', 'BTC-IND-FAIL'
        ]

        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        
        self.assertEqual(cm.exception.code, 2)
        output_err = self.mock_stderr.getvalue()
        self.assertIn("Failed to calculate ATR for symbol BTC-IND-FAIL. Output: None", output_err)

    @patch('src.cli.get_historical_data')
    @patch('src.cli.get_indicator')
    def test_atr_from_symbol_failure_indicator_returns_empty_values(self, mock_get_indicator, mock_get_historical_data):
        # --- Arrange ---
        mock_df = pd.DataFrame({
            'open': [10]*20, 'high': [11]*20, 'low': [9]*20, 'close': [10]*20
        })
        mock_get_historical_data.return_value = mock_df
        mock_get_indicator.return_value = {'values': pd.Series([], dtype=float), 'metadata': {}}

        test_args = [
            'src/cli.py', '100', 'long', '--atr-from-symbol', 'BTC-IND-EMPTY'
        ]

        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        
        self.assertEqual(cm.exception.code, 2)
        output_err = self.mock_stderr.getvalue()
        self.assertIn("ATR calculation for BTC-IND-EMPTY returned no values.", output_err)

    @patch('src.cli.get_historical_data')
    @patch('src.cli.get_indicator')
    def test_atr_from_symbol_failure_indicator_returns_all_nan_values(self, mock_get_indicator, mock_get_historical_data):
        # --- Arrange ---
        mock_df = pd.DataFrame({
            'open': [10]*20, 'high': [11]*20, 'low': [9]*20, 'close': [10]*20
        })
        mock_get_historical_data.return_value = mock_df
        mock_get_indicator.return_value = {'values': pd.Series([pd.NA]*5, dtype=float), 'metadata': {}}

        test_args = [
            'src/cli.py', '100', 'long', '--atr-from-symbol', 'BTC-IND-NAN'
        ]

        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        
        self.assertEqual(cm.exception.code, 2)
        output_err = self.mock_stderr.getvalue()
        # The error message in cli.py for this case is "Could not extract a valid ATR value... (all NaNs?)."
        self.assertIn("Could not extract a valid ATR value for BTC-IND-NAN after calculation (all NaNs?).", output_err)

    @patch('pandas.read_csv')
    @patch('src.cli.get_indicator')
    @patch('src.cli.calculate_atr_trade_parameters')
    def test_atr_from_file_csv_ohlc_success(self, mock_calculate_params, mock_get_indicator, mock_read_csv):
        # --- Arrange ---
        # Dummy CSV content
        csv_content = "Date,OpenPrice,HighPrice,LowPrice,ClosePrice\n"
        for i in range(20):
            csv_content += f"2023-01-{i+1:02d},{10+i},{11+i},{9+i},{10+i}\n"
        
        # Mock pd.read_csv
        mock_csv_df = pd.read_csv(io.StringIO(csv_content))
        mock_read_csv.return_value = mock_csv_df

        # Mock get_indicator (ATR calculated from the CSV's OHLC)
        mock_atr_series_csv = pd.Series([0.5 + i*0.01 for i in range(7)]) # ATR period 14, needs 14 data points from CSV (20 provided)
                                                                       # get_indicator will receive a df of 20 rows. ATR series has 20-14+1 = 7 values if period is 14.
                                                                       # Let's make it simpler: if ATR is 14, result is len(df) - 14 + 1 values if no NaNs start
                                                                       # The mock series here should match what TA-Lib/pandas_ta might produce.
                                                                       # The cli.py logic takes the *last* non-NaN value.
        mock_get_indicator.return_value = {'values': mock_atr_series_csv, 'metadata': {'params': {'length': 14}}}

        mock_calculate_params.return_value = {
            'position_size': 20.0,
            'stop_loss_percentage': 2.5
        }

        test_args = [
            'src/cli.py', '200', 'short',
            '--atr-from-file', 'dummy.csv',
            '--csv-ohlc', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice',
            '--csv-date-column', 'Date',
            '--atr-period', '14', # This will be used for get_indicator
            '--risk_amount', '100'
        ]

        # --- Act ---
        with patch.object(sys, 'argv', test_args):
            try:
                cli.main()
            except SystemExit as e:
                print(f"Sys.exit called unexpectedly in success test (CSV OHLC): {e.code}")
                print(f"Stderr: {self.mock_stderr.getvalue()}")
                raise

        # --- Assert ---
        mock_read_csv.assert_called_once_with('dummy.csv', parse_dates=['Date'], index_col='Date')
        
        # Check call to get_indicator
        # It receives a DataFrame with standardized lowercase columns: open, high, low, close
        # And only the necessary columns are passed after pd.to_numeric and dropna
        # For simplicity, we won't reconstruct the exact DF here but check key aspects.
        self.assertTrue(mock_get_indicator.called)
        call_df_to_indicator = mock_get_indicator.call_args[0][0]
        self.assertListEqual(list(call_df_to_indicator.columns), ['open', 'high', 'low', 'close'])
        self.assertTrue(len(call_df_to_indicator) >= 14) # Should have enough rows after processing
        self.assertEqual(mock_get_indicator.call_args[1]['indicator'], 'atr')
        self.assertEqual(mock_get_indicator.call_args[1]['params'], {'length': 14})

        expected_atr_csv = mock_atr_series_csv.dropna().iloc[-1]
        mock_calculate_params.assert_called_once()
        calc_args = mock_calculate_params.call_args[0]
        self.assertEqual(calc_args[0], 200.0) # entry_price
        self.assertAlmostEqual(calc_args[1], expected_atr_csv, places=5) # atr
        self.assertEqual(calc_args[2], 'short') # position_type
        self.assertEqual(mock_calculate_params.call_args[1]['risk_amount'], 100.0)

        output = self.mock_stdout.getvalue()
        self.assertIn(f"Calculating ATR from file: dummy.csv", output)
        self.assertIn(f"Calculated ATR from CSV file dummy.csv using OHLC columns: {expected_atr_csv:.8f}", output)

    @patch('pandas.read_csv')
    def test_atr_from_file_csv_ohlc_failure_file_not_found(self, mock_read_csv):
        # --- Arrange ---
        mock_read_csv.side_effect = FileNotFoundError("File not found: missing.csv")
        test_args = [
            'src/cli.py', '10', 'long',
            '--atr-from-file', 'missing.csv',
            '--csv-ohlc', 'O', 'H', 'L', 'C'
        ]
        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("CSV file not found: missing.csv", self.mock_stderr.getvalue())

    @patch('pandas.read_csv')
    def test_atr_from_file_csv_ohlc_failure_missing_column(self, mock_read_csv):
        # --- Arrange ---
        csv_content = "Date,OpenPrice,HighPrice,LowPrice\n2023-01-01,10,11,9"
        mock_csv_df = pd.read_csv(io.StringIO(csv_content))
        mock_read_csv.return_value = mock_csv_df
        test_args = [
            'src/cli.py', '10', 'long',
            '--atr-from-file', 'bad_cols.csv',
            '--csv-ohlc', 'OpenPrice', 'HighPrice', 'LowPrice', 'ClosePrice' # ClosePrice is missing
        ]
        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Missing OHLC columns in bad_cols.csv: ClosePrice", self.mock_stderr.getvalue())
    
    @patch('pandas.read_csv')
    @patch('src.cli.get_indicator')
    def test_atr_from_file_csv_ohlc_failure_insufficient_rows(self, mock_get_indicator, mock_read_csv):
        # --- Arrange ---
        # CSV with only 5 rows, ATR period 14
        csv_content = "Date,O,H,L,C\n" + "2023-01-01,1,2,0,1\n" * 5
        mock_csv_df = pd.read_csv(io.StringIO(csv_content))
        mock_read_csv.return_value = mock_csv_df
        # get_indicator won't be called if pre-check fails

        test_args = [
            'src/cli.py', '10', 'long',
            '--atr-from-file', 'short.csv',
            '--csv-ohlc', 'O', 'H', 'L', 'C',
            '--atr-period', '14'
        ]
        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Not enough valid OHLC data in short.csv (after parsing & NaN removal) to calculate ATR with period 14.", self.mock_stderr.getvalue())

    @patch('pandas.read_csv')
    @patch('src.cli.calculate_atr_trade_parameters')
    def test_atr_from_file_csv_atr_column_success(self, mock_calculate_params, mock_read_csv):
        # --- Arrange ---
        csv_content = "Date,SomeData,MyATR\n2023-01-01,10,1.5\n2023-01-02,11,1.4\n2023-01-03,12,1.6"
        mock_csv_df = pd.read_csv(io.StringIO(csv_content))
        mock_read_csv.return_value = mock_csv_df

        mock_calculate_params.return_value = {
            'position_size': 15.0,
            'stop_loss_percentage': 3.0
        }
        expected_atr_from_csv_col = 1.6 # Last value in MyATR column

        test_args = [
            'src/cli.py', '150', 'long',
            '--atr-from-file', 'data_with_atr.csv',
            '--csv-atr-column', 'MyATR',
            '--csv-date-column', 'Date'
        ]

        # --- Act ---
        with patch.object(sys, 'argv', test_args):
            try:
                cli.main()
            except SystemExit as e:
                print(f"Sys.exit called unexpectedly in success test (CSV ATR Col): {e.code}")
                print(f"Stderr: {self.mock_stderr.getvalue()}")
                raise

        # --- Assert ---
        mock_read_csv.assert_called_once_with('data_with_atr.csv', parse_dates=['Date'], index_col='Date')
        
        mock_calculate_params.assert_called_once()
        calc_args = mock_calculate_params.call_args[0]
        self.assertEqual(calc_args[0], 150.0) # entry_price
        self.assertAlmostEqual(calc_args[1], expected_atr_from_csv_col, places=5) # atr
        self.assertEqual(calc_args[2], 'long') # position_type

        output = self.mock_stdout.getvalue()
        self.assertIn(f"Calculating ATR from file: data_with_atr.csv", output)
        self.assertIn(f"Using ATR from CSV column 'MyATR': {expected_atr_from_csv_col:.8f}", output)

    @patch('pandas.read_csv')
    def test_atr_from_file_csv_atr_column_failure_column_not_found(self, mock_read_csv):
        # --- Arrange ---
        csv_content = "Date,SomeData\n2023-01-01,10"
        mock_csv_df = pd.read_csv(io.StringIO(csv_content))
        mock_read_csv.return_value = mock_csv_df
        test_args = [
            'src/cli.py', '10', 'long',
            '--atr-from-file', 'no_atr_col.csv',
            '--csv-atr-column', 'MissingATRCol'
        ]
        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("ATR column 'MissingATRCol' not found in no_atr_col.csv", self.mock_stderr.getvalue())

    @patch('pandas.read_csv')
    def test_atr_from_file_csv_atr_column_failure_empty_column(self, mock_read_csv):
        # --- Arrange ---
        csv_content = "Date,MyATR\n2023-01-01,\n2023-01-02,"
        mock_csv_df = pd.read_csv(io.StringIO(csv_content))
        mock_read_csv.return_value = mock_csv_df
        test_args = [
            'src/cli.py', '10', 'long',
            '--atr-from-file', 'empty_atr_col.csv',
            '--csv-atr-column', 'MyATR'
        ]
        # --- Act & Assert ---
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("ATR column 'MyATR' in empty_atr_col.csv contains no valid data.", self.mock_stderr.getvalue())

    def test_mutually_exclusive_atr_options(self):
        test_cases = [
            (['src/cli.py', '10', 'long', '--atr', '1.0', '--atr-from-symbol', 'BTC'], "argument --atr-from-symbol: not allowed with argument --atr"),
            (['src/cli.py', '10', 'long', '--atr', '1.0', '--atr-from-file', 'f.csv'], "argument --atr-from-file: not allowed with argument --atr"),
            (['src/cli.py', '10', 'long', '--atr-from-symbol', 'BTC', '--atr-from-file', 'f.csv'], "argument --atr-from-file: not allowed with argument --atr-from-symbol"),
        ]
        for args, error_msg_part in test_cases:
            with self.subTest(args=args):
                with patch.object(sys, 'argv', args):
                    with self.assertRaises(SystemExit) as cm:
                        cli.main()
                self.assertEqual(cm.exception.code, 2)
                # argparse error messages for mutually exclusive groups can vary slightly in phrasing
                # We check for a key part of the message.
                self.assertIn(error_msg_part, self.mock_stderr.getvalue())
                self.mock_stderr.truncate(0) # Clear stderr for next subtest
                self.mock_stderr.seek(0)

    def test_atr_from_file_requires_csv_option(self):
        test_args = ['src/cli.py', '10', 'long', '--atr-from-file', 'f.csv']
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--atr-from-file requires either --csv-ohlc or --csv-atr-column", self.mock_stderr.getvalue())

    def test_atr_from_file_cannot_use_both_csv_options(self):
        test_args = ['src/cli.py', '10', 'long', '--atr-from-file', 'f.csv', '--csv-ohlc', 'O','H','L','C', '--csv-atr-column', 'ATRCOL']
        with patch.object(sys, 'argv', test_args):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Cannot use both --csv-ohlc and --csv-atr-column with --atr-from-file", self.mock_stderr.getvalue())

    def test_final_atr_value_must_be_positive(self):
        # Test with --atr 0
        test_args_zero = ['src/cli.py', '10', 'long', '--atr', '0']
        with patch.object(sys, 'argv', test_args_zero):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Determined ATR value (0.0) must be positive.", self.mock_stderr.getvalue())
        self.mock_stderr.truncate(0)
        self.mock_stderr.seek(0)

        # Test with --atr -1
        test_args_negative = ['src/cli.py', '10', 'long', '--atr', '-1']
        with patch.object(sys, 'argv', test_args_negative):
            with self.assertRaises(SystemExit) as cm:
                cli.main()
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("Determined ATR value (-1.0) must be positive.", self.mock_stderr.getvalue())


if __name__ == '__main__':
    unittest.main() 