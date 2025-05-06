"""
Tests for notebook generator functionality.

This test suite covers the notebook generator that creates
customized analysis notebooks from templates.
"""

import os
import json
import pytest
import tempfile
import nbformat
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

# Import the module to be tested (will be implemented later)
# from src.jupyter.notebook_generator import (
#     load_template, 
#     substitute_params,
#     generate_notebook,
#     launch_notebook,
#     NotebookGeneratorError
# )


# Mock template content
MOCK_SHORT_TEMPLATE = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Short-Term Analysis for {{SYMBOL}}"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": ["%short --verbose"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": ["symbol = \"{{SYMBOL}}\"", "\n", "print(f\"Analyzing {symbol}\")"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Financial Analysis",
            "language": "python",
            "name": "financial_analysis"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

MOCK_MEDIUM_TEMPLATE = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# Medium-Term Analysis for {{SYMBOL}}"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": ["%medium --verbose"]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "source": ["symbol = \"{{SYMBOL}}\"", "\n", "print(f\"Analyzing {symbol}\")"]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Financial Analysis",
            "language": "python",
            "name": "financial_analysis"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}


class TestTemplateLoading:
    """Tests for template loading functionality."""
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_existing_template(self, mock_json_load, mock_file_open):
        """Test loading an existing template."""
        # Setup mocks
        mock_json_load.return_value = MOCK_SHORT_TEMPLATE
        
        # Import the function if it's not at the module level
        from src.jupyter.notebook_generator import load_template
        
        # Execute function
        template = load_template('short')
        
        # Assertions
        assert template == MOCK_SHORT_TEMPLATE
        mock_file_open.assert_called_with(
            Path('notebooks/templates/short_term_analysis.ipynb'), 'r')
        mock_json_load.assert_called_once()
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    def test_load_with_file_extension(self, mock_json_load, mock_file_open):
        """Test loading a template when a file extension is provided."""
        # Setup mocks
        mock_json_load.return_value = MOCK_SHORT_TEMPLATE
        
        # Import the function if it's not at the module level
        from src.jupyter.notebook_generator import load_template
        
        # Execute function
        template = load_template('short_term_analysis.ipynb')
        
        # Assertions
        assert template == MOCK_SHORT_TEMPLATE
        mock_file_open.assert_called_with(
            Path('notebooks/templates/short_term_analysis.ipynb'), 'r')
    
    @patch('builtins.open')
    def test_load_missing_template(self, mock_file_open):
        """Test loading a non-existent template."""
        # Setup mock to raise FileNotFoundError
        mock_file_open.side_effect = FileNotFoundError("File not found")
        
        # Import the function and error class
        from src.jupyter.notebook_generator import load_template, NotebookGeneratorError
        
        # Test that function raises the appropriate error
        with pytest.raises(NotebookGeneratorError) as excinfo:
            load_template('nonexistent')
        
        # Verify error message
        assert "Template not found" in str(excinfo.value)
        assert "nonexistent" in str(excinfo.value)


class TestParameterSubstitution:
    """Tests for parameter substitution functionality."""
    
    def test_substitute_basic_params(self):
        """Test basic parameter substitution in notebook cells."""
        # Import the function
        from src.jupyter.notebook_generator import substitute_params
        
        # Create a simple template with parameterized text
        template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Analysis for {{SYMBOL}}"]
                },
                {
                    "cell_type": "code",
                    "source": ["symbol = \"{{SYMBOL}}\"", "\n", "period = \"{{PERIOD}}\""]
                }
            ]
        }
        
        # Parameters to substitute
        params = {
            "SYMBOL": "BTC",
            "PERIOD": "1y"
        }
        
        # Perform substitution
        result = substitute_params(template, params)
        
        # Assertions
        assert result["cells"][0]["source"][0] == "# Analysis for BTC"
        assert result["cells"][1]["source"][0] == "symbol = \"BTC\""
        assert result["cells"][1]["source"][2] == "period = \"1y\""
    
    def test_substitute_missing_params(self):
        """Test substitution with missing parameters."""
        # Import the function
        from src.jupyter.notebook_generator import substitute_params, NotebookGeneratorError
        
        # Create a template with a parameter
        template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Analysis for {{SYMBOL}}"]
                }
            ]
        }
        
        # Empty parameters dict
        params = {}
        
        # Test that function raises the appropriate error
        with pytest.raises(NotebookGeneratorError) as excinfo:
            substitute_params(template, params)
        
        # Verify error message
        assert "Missing required parameter" in str(excinfo.value)
        assert "SYMBOL" in str(excinfo.value)
    
    def test_substitute_list_source(self):
        """Test substitution when source is a list of strings."""
        # Import the function
        from src.jupyter.notebook_generator import substitute_params
        
        # Create a template with list source
        template = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "source": ["# Analysis for {{SYMBOL}}", "\n", "Period: {{PERIOD}}"]
                }
            ]
        }
        
        # Parameters to substitute
        params = {
            "SYMBOL": "BTC",
            "PERIOD": "1y"
        }
        
        # Perform substitution
        result = substitute_params(template, params)
        
        # Assertions
        assert result["cells"][0]["source"][0] == "# Analysis for BTC"
        assert result["cells"][0]["source"][2] == "Period: 1y"


class TestNotebookGeneration:
    """Tests for notebook generation functionality."""
    
    @patch('src.jupyter.notebook_generator.load_template')
    @patch('src.jupyter.notebook_generator.substitute_params')
    @patch('nbformat.write')
    def test_generate_notebook(self, mock_nbformat_write, mock_substitute, mock_load):
        """Test generating a notebook from a template with parameters."""
        # Setup mocks
        mock_load.return_value = MOCK_SHORT_TEMPLATE
        mock_substitute.return_value = MOCK_SHORT_TEMPLATE
        
        # Import the function
        from src.jupyter.notebook_generator import generate_notebook
        
        # Create parameters
        params = {"SYMBOL": "BTC"}
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix='.ipynb') as tmp:
            output_path = tmp.name
            
            # Generate notebook
            result_path = generate_notebook('short', params, output_path)
            
            # Assertions
            assert result_path == output_path
            mock_load.assert_called_with('short')
            mock_substitute.assert_called_with(MOCK_SHORT_TEMPLATE, params)
            mock_nbformat_write.assert_called_once()
    
    @patch('src.jupyter.notebook_generator.load_template')
    @patch('src.jupyter.notebook_generator.substitute_params')
    @patch('nbformat.write')
    def test_generate_with_default_output(self, mock_nbformat_write, mock_substitute, mock_load):
        """Test generating a notebook with default output path."""
        # Setup mocks
        mock_load.return_value = MOCK_SHORT_TEMPLATE
        mock_substitute.return_value = MOCK_SHORT_TEMPLATE
        
        # Import the function
        from src.jupyter.notebook_generator import generate_notebook
        
        # Create parameters
        params = {"SYMBOL": "BTC"}
        
        # Generate notebook with default output path
        result_path = generate_notebook('short', params)
        
        # Assertions
        assert "BTC_short_analysis.ipynb" in result_path
        mock_load.assert_called_with('short')
        mock_substitute.assert_called_with(MOCK_SHORT_TEMPLATE, params)
        mock_nbformat_write.assert_called_once()
    
    @patch('src.jupyter.notebook_generator.load_template')
    def test_generate_with_template_error(self, mock_load):
        """Test error handling when template loading fails."""
        # Setup mock to raise error
        from src.jupyter.notebook_generator import NotebookGeneratorError
        mock_load.side_effect = NotebookGeneratorError("Template not found")
        
        # Import the function
        from src.jupyter.notebook_generator import generate_notebook
        
        # Create parameters
        params = {"SYMBOL": "BTC"}
        
        # Test that function propagates the error
        with pytest.raises(NotebookGeneratorError) as excinfo:
            generate_notebook('nonexistent', params)
        
        # Verify error message
        assert "Template not found" in str(excinfo.value)
    
    @patch('src.jupyter.notebook_generator.substitute_params')
    @patch('src.jupyter.notebook_generator.load_template')
    def test_generate_with_substitution_error(self, mock_load, mock_substitute):
        """Test error handling when parameter substitution fails."""
        # Setup mocks
        mock_load.return_value = MOCK_SHORT_TEMPLATE
        from src.jupyter.notebook_generator import NotebookGeneratorError
        mock_substitute.side_effect = NotebookGeneratorError("Missing parameter")
        
        # Import the function
        from src.jupyter.notebook_generator import generate_notebook
        
        # Create parameters with missing required param
        params = {}
        
        # Test that function propagates the error
        with pytest.raises(NotebookGeneratorError) as excinfo:
            generate_notebook('short', params)
        
        # Verify error message
        assert "Missing parameter" in str(excinfo.value)


class TestNotebookLaunching:
    """Tests for notebook launching functionality."""
    
    @patch('subprocess.Popen')
    def test_launch_notebook(self, mock_popen):
        """Test launching a notebook."""
        # Setup mock
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Import the function
        from src.jupyter.notebook_generator import launch_notebook
        
        # Create a test notebook path
        notebook_path = "test_notebook.ipynb"
        
        # Launch notebook
        process = launch_notebook(notebook_path)
        
        # Assertions
        assert process == mock_process
        mock_popen.assert_called_once()
        
        # Check that command contains jupyter and the notebook path
        args = mock_popen.call_args[0][0]
        assert "jupyter" in args[0]
        assert "notebook" in args[0] or args[1] == "notebook"
        assert notebook_path in args
    
    @patch('subprocess.Popen')
    def test_launch_nonexistent_notebook(self, mock_popen):
        """Test launching a non-existent notebook."""
        # Setup mock to raise FileNotFoundError when command is not found
        mock_popen.side_effect = FileNotFoundError("jupyter not found")
        
        # Import the function
        from src.jupyter.notebook_generator import launch_notebook, NotebookGeneratorError
        
        # Test that function handles error properly
        with pytest.raises(NotebookGeneratorError) as excinfo:
            launch_notebook("nonexistent.ipynb")
        
        # Verify error message
        assert "Failed to launch notebook" in str(excinfo.value)


class TestIntegration:
    """Integration tests for notebook generator."""
    
    @patch('src.jupyter.notebook_generator.load_template')
    @patch('nbformat.write')
    @patch('subprocess.Popen')
    def test_end_to_end_workflow(self, mock_popen, mock_write, mock_load):
        """Test the complete workflow: generate and launch notebook."""
        # Setup mocks
        mock_load.return_value = MOCK_SHORT_TEMPLATE
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        # Import the functions
        from src.jupyter.notebook_generator import generate_notebook, launch_notebook
        
        # Create parameters
        params = {"SYMBOL": "BTC"}
        
        # Generate a notebook
        output_path = generate_notebook('short', params)
        
        # Launch the generated notebook
        process = launch_notebook(output_path)
        
        # Assertions
        assert mock_load.called
        assert mock_write.called
        assert mock_popen.called
        assert process == mock_process 