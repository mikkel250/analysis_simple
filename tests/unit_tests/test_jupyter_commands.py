import unittest
from unittest.mock import patch, MagicMock
import os
from pathlib import Path

from src.cli.commands.jupyter import start, demo, run, analysis

class TestJupyterCommands(unittest.TestCase):
    """Tests for the Jupyter CLI commands."""
    
    @patch('src.cli.commands.jupyter.subprocess.run')
    @patch('src.cli.commands.jupyter.shutil.which')
    @patch('src.cli.commands.jupyter.os.path.isdir')
    def test_start_command(self, mock_isdir, mock_which, mock_run):
        """Test the jupyter start command."""
        # Mock dependencies
        mock_which.return_value = "/usr/bin/jupyter"
        mock_isdir.return_value = True
        
        # Call the command
        result = start(example=False, port=9999, notebook_dir="")
        
        # Check result
        self.assertEqual(result, 0)
        mock_run.assert_called_once()
    
    @patch('src.cli.commands.jupyter.subprocess.run')
    @patch('src.cli.commands.jupyter.shutil.which')
    @patch('src.cli.commands.jupyter.Path.exists')
    def test_demo_command(self, mock_exists, mock_which, mock_run):
        """Test the jupyter demo command."""
        # Mock dependencies
        mock_which.return_value = "/usr/bin/jupyter"
        mock_exists.return_value = True
        
        # Call the command
        result = demo()
        
        # Check result
        self.assertEqual(result, 0)
        mock_run.assert_called_once()
    
    @patch('src.cli.commands.jupyter.subprocess.run')
    @patch('src.cli.commands.jupyter.shutil.which')
    @patch('src.cli.commands.jupyter.Path.exists')
    def test_run_existing_notebook(self, mock_exists, mock_which, mock_run):
        """Test running an existing notebook."""
        # Mock dependencies
        mock_which.return_value = "/usr/bin/jupyter"
        mock_exists.return_value = True
        
        # Call the command
        result = run(notebook="test_notebook", create_if_missing=False)
        
        # Check result
        self.assertEqual(result, 0)
        mock_run.assert_called_once()
    
    @patch('src.cli.commands.jupyter.subprocess.run')
    @patch('src.cli.commands.jupyter.shutil.which')
    @patch('src.cli.commands.jupyter.Path.exists')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('nbformat.write')
    @patch('src.cli.commands.jupyter.nbformat.v4', create=True)
    def test_run_create_notebook(self, mock_nbf, mock_write, mock_open, mock_exists, mock_which, mock_run):
        """Test creating and running a new notebook."""
        # Mock dependencies
        mock_which.return_value = "/usr/bin/jupyter"
        # First check if notebook exists (no), then check after creating it (yes)
        mock_exists.side_effect = [True, False, True]
        
        # Call the command
        result = run(notebook="new_notebook", create_if_missing=True)
        
        # Check result
        self.assertEqual(result, 0)
        mock_open.assert_called_once()
        mock_write.assert_called_once()
        mock_run.assert_called_once()
    
    @patch('src.cli.commands.jupyter.shutil.which')
    def test_jupyter_not_installed(self, mock_which):
        """Test behavior when Jupyter is not installed."""
        # Mock dependencies
        mock_which.return_value = None
        
        # Call the commands
        start_result = start(example=False, port=8888, notebook_dir="")
        demo_result = demo()
        run_result = run(notebook="test", create_if_missing=False)
        analysis_result = analysis(symbol="BTC")
        
        # Check results
        self.assertEqual(start_result, 1)
        self.assertEqual(demo_result, 1)
        self.assertEqual(run_result, 1)
        self.assertEqual(analysis_result, 1)
    
    @patch('src.cli.commands.jupyter.generate_and_launch_notebook')
    @patch('src.cli.commands.jupyter.generate_analysis_notebook')
    @patch('src.cli.commands.jupyter.Path.exists')
    @patch('src.cli.commands.jupyter.Path.stat')
    @patch('src.jupyter.launch_analysis.create_demo_template')
    def test_analysis_command(self, mock_create_template, mock_stat, mock_exists, 
                             mock_generate_analysis, mock_generate_and_launch):
        """Test the jupyter analysis command."""
        # Mock Path.exists() to return False so create_demo_template is called
        mock_exists.return_value = False
        
        # Mock stat return value with a small size
        stat_result = MagicMock()
        stat_result.st_size = 5  # Small file size to trigger template creation
        mock_stat.return_value = stat_result
        
        # Mock generate_analysis_notebook to return a path
        mock_generate_analysis.return_value = "/path/to/notebook.ipynb"
        
        # Mock generate_and_launch_notebook to return a path and a process
        mock_process = MagicMock()
        mock_generate_and_launch.return_value = ("/path/to/notebook.ipynb", mock_process)
        
        # Test no_launch=True (generate only)
        result1 = analysis(symbol="BTC", timeframe="short", output=None, no_launch=True)
        self.assertEqual(result1, 0)
        mock_generate_analysis.assert_called_once_with(
            symbol="BTC", timeframe="short", output_path=None
        )
        
        # Reset mocks
        mock_generate_analysis.reset_mock()
        mock_generate_and_launch.reset_mock()
        
        # Test no_launch=False (generate and launch)
        result2 = analysis(symbol="ETH", timeframe="medium", output="/custom/path.ipynb", no_launch=False)
        self.assertEqual(result2, 0)
        mock_generate_and_launch.assert_called_once_with(
            symbol="ETH", timeframe="medium", output_path="/custom/path.ipynb"
        )

    @patch('src.cli.commands.jupyter.generate_and_launch_notebook')
    def test_analysis_invalid_symbol(self, mock_generate_and_launch):
        """Test the analysis command with an invalid symbol."""
        # Test with invalid symbols
        result1 = analysis(symbol="BTC$", timeframe="medium")
        self.assertEqual(result1, 1)
        mock_generate_and_launch.assert_not_called()
        
        result2 = analysis(symbol="BTC!", timeframe="medium")
        self.assertEqual(result2, 1)
        mock_generate_and_launch.assert_not_called()
    
    @patch('src.cli.commands.jupyter.generate_and_launch_notebook')
    def test_analysis_invalid_timeframe(self, mock_generate_and_launch):
        """Test the analysis command with an invalid timeframe."""
        # Test with invalid timeframe
        result = analysis(symbol="BTC", timeframe="invalid")
        self.assertEqual(result, 1)
        mock_generate_and_launch.assert_not_called()
    
    @patch('src.cli.commands.jupyter.generate_and_launch_notebook')
    @patch('src.cli.commands.jupyter.Path.exists')
    @patch('src.cli.commands.jupyter.Path.stat')
    @patch('src.jupyter.launch_analysis.create_demo_template')
    def test_analysis_default_timeframe(self, mock_create_template, mock_stat, mock_exists, mock_generate_and_launch):
        """Test the jupyter analysis command uses 'short' as the default timeframe."""
        # Mock Path.exists() to return False so create_demo_template is called
        mock_exists.return_value = False
        
        # Mock stat return value with a small size
        stat_result = MagicMock()
        stat_result.st_size = 5  # Small file size to trigger template creation
        mock_stat.return_value = stat_result
        
        # Mock process for generate_and_launch_notebook
        mock_process = MagicMock()
        mock_generate_and_launch.return_value = ("/path/to/notebook.ipynb", mock_process)
        
        # Call analysis with only symbol parameter (should use default timeframe)
        result = analysis(symbol="BTC")
        
        # Check result and assert generate_and_launch_notebook was called with correct parameters
        self.assertEqual(result, 0)
        mock_generate_and_launch.assert_called_once_with(
            symbol="BTC", timeframe="short", output_path=None
        )


if __name__ == '__main__':
    unittest.main() 