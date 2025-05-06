"""
Notebook Generator

This module provides functionality to generate customized analysis notebooks
from templates and launch them in Jupyter.
"""

import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Union, Optional, List

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


class NotebookGeneratorError(Exception):
    """Exception raised for errors in the notebook generator."""
    pass


def create_template_notebook(timeframe: str) -> nbformat.NotebookNode:
    """
    Create a template notebook for a specific timeframe.
    
    Args:
        timeframe: Trading timeframe (short, medium, long)
        
    Returns:
        NotebookNode containing the template notebook
    """
    # Create markdown cells
    title_cell = new_markdown_cell(f"# {timeframe.capitalize()}-Term Market Analysis for {{{{SYMBOL}}}}")
    
    fetch_data_cell = new_markdown_cell("## Fetch Data\n\nFetching market data for {{SYMBOL}}...")
    
    run_analysis_cell = new_markdown_cell("## Run Analysis\n\nPerforming market analysis...")
    
    # Create code cells
    magic_cell = new_code_cell(f"%{timeframe} --verbose")
    
    symbol_cell = new_code_cell('symbol = "{{SYMBOL}}"\nprint(f"Analyzing {symbol}")')
    
    analyzer_cell = new_code_cell(
        'from src.jupyter.market_analyzer import MarketAnalyzer\n\n'
        'analyzer = MarketAnalyzer(symbol=symbol, timeframe="{{TIMEFRAME}}")\n'
        'data = analyzer.fetch_data()\n\n'
        'data.tail()'
    )
    
    analysis_cell = new_code_cell(
        'analysis_results = analyzer.run_analysis()\n\n'
        'print(f"Analysis completed for {symbol}")\n'
        'analysis_results[\'performance\']'
    )
    
    # Create a new notebook
    nb = new_notebook(
        cells=[
            title_cell,
            magic_cell,
            symbol_cell,
            fetch_data_cell,
            analyzer_cell,
            run_analysis_cell,
            analysis_cell
        ],
        metadata={
            'kernelspec': {
                'display_name': 'Financial Analysis',
                'language': 'python',
                'name': 'financial_analysis'
            },
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.9.0'
            }
        }
    )
    
    return nb


def load_template(template_name: str) -> nbformat.NotebookNode:
    """
    Load a template notebook from the templates directory.
    
    Args:
        template_name: Name of the template, with or without file extension
    
    Returns:
        NotebookNode containing the notebook content
    
    Raises:
        NotebookGeneratorError: If the template cannot be found or loaded
    """
    # Map common names to template files
    template_map = {
        'short': 'short_term_analysis.ipynb',
        'medium': 'medium_term_analysis.ipynb',
        'long': 'long_term_analysis.ipynb'
    }
    
    # Get the template filename
    if template_name.endswith('.ipynb'):
        template_file = template_name
    else:
        template_file = template_map.get(template_name, f"{template_name}_analysis.ipynb")
    
    template_path = Path('notebooks/templates') / template_file
    timeframe = template_name.replace('_term_analysis', '').replace('.ipynb', '')
    if timeframe not in ['short', 'medium', 'long']:
        timeframe = 'medium'
    
    try:
        # Try to load the template, if it doesn't exist or has issues, create a new one
        try:
            notebook = nbformat.read(template_path, as_version=4)
            # Basic validation
            if 'cells' not in dir(notebook) or len(notebook.cells) == 0:
                raise ValueError("Invalid notebook structure")
            return notebook
        except (FileNotFoundError, ValueError) as e:
            # Create a new template notebook
            notebook = create_template_notebook(timeframe)
            
            # Ensure the directory exists
            os.makedirs(os.path.dirname(template_path), exist_ok=True)
            
            # Save the template
            nbformat.write(notebook, template_path)
            
            return notebook
        
    except Exception as e:
        raise NotebookGeneratorError(f"Error loading template: {str(e)}")


def substitute_params(template: nbformat.NotebookNode, params: Dict[str, str]) -> nbformat.NotebookNode:
    """
    Substitute parameters in a notebook template.
    
    Args:
        template: Notebook template content
        params: Parameters to substitute, with keys matching the placeholders
    
    Returns:
        NotebookNode containing the notebook with substituted parameters
    
    Raises:
        NotebookGeneratorError: If required parameters are missing
    """
    # Create a deep copy of the template to avoid modifying the original
    notebook_json = json.dumps(template, default=lambda obj: obj.__dict__ if hasattr(obj, '__dict__') else str(obj))
    notebook = nbformat.reads(notebook_json, as_version=4)
    
    # Find all placeholders in the template
    placeholders = set()
    pattern = r'{{(\w+)}}'
    
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' or cell.cell_type == 'code':
            source = cell.source
            placeholders.update(re.findall(pattern, source))
    
    # Check if all required parameters are provided
    missing_params = [param for param in placeholders if param not in params]
    if missing_params:
        raise NotebookGeneratorError(f"Missing required parameter(s): {', '.join(missing_params)}")
    
    # Substitute parameters in each cell
    for cell in notebook.cells:
        if cell.cell_type == 'markdown' or cell.cell_type == 'code':
            for param, value in params.items():
                cell.source = cell.source.replace(f"{{{{{param}}}}}", value)
    
    return notebook


def generate_notebook(
    template_name: str,
    params: Dict[str, str],
    output_path: Optional[str] = None
) -> str:
    """
    Generate a customized notebook from a template with parameters.
    
    Args:
        template_name: Name of the template to use
        params: Parameters to substitute in the template
        output_path: Path to save the generated notebook (optional)
    
    Returns:
        Path to the generated notebook
    
    Raises:
        NotebookGeneratorError: If there's an error loading the template or substituting parameters
    """
    # Load the template
    template = load_template(template_name)
    
    # Substitute parameters
    notebook = substitute_params(template, params)
    
    # Generate output path if not provided
    if output_path is None:
        # Create a sensible filename based on parameters
        symbol = params.get('SYMBOL', 'analysis')
        timeframe = template_name
        output_path = f"{symbol}_{timeframe}_analysis.ipynb"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Write the notebook to the output path
    try:
        nbformat.write(notebook, output_path)
    except Exception as e:
        raise NotebookGeneratorError(f"Failed to write notebook: {str(e)}")
    
    return output_path


def launch_notebook(notebook_path: str) -> subprocess.Popen:
    """
    Launch a notebook in Jupyter.
    
    Args:
        notebook_path: Path to the notebook to launch
    
    Returns:
        Subprocess object representing the Jupyter process
    
    Raises:
        NotebookGeneratorError: If Jupyter cannot be launched
    """
    try:
        # Launch Jupyter notebook
        process = subprocess.Popen(
            ['jupyter', 'notebook', notebook_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return process
    except Exception as e:
        raise NotebookGeneratorError(f"Failed to launch notebook: {str(e)}")


def generate_analysis_notebook(
    symbol: str,
    timeframe: str = 'medium',
    output_path: Optional[str] = None
) -> str:
    """
    Generate a complete market analysis notebook for a symbol.
    
    Args:
        symbol: Market symbol to analyze (e.g., 'BTC')
        timeframe: Trading timeframe ('short', 'medium', 'long')
        output_path: Path to save the generated notebook (optional)
    
    Returns:
        Path to the generated notebook
    
    Raises:
        NotebookGeneratorError: If there's an error generating the notebook
    """
    # Validate timeframe
    valid_timeframes = ['short', 'medium', 'long']
    if timeframe not in valid_timeframes:
        raise NotebookGeneratorError(
            f"Invalid timeframe: {timeframe}. Must be one of: {', '.join(valid_timeframes)}"
        )
    
    # Create parameters for substitution
    params = {
        'SYMBOL': symbol,
        'TIMEFRAME': timeframe
    }
    
    # Generate the notebook
    return generate_notebook(timeframe, params, output_path)


def generate_and_launch_notebook(
    symbol: str,
    timeframe: str = 'medium',
    output_path: Optional[str] = None
) -> tuple:
    """
    Generate and launch an analysis notebook.
    
    Args:
        symbol: Market symbol to analyze (e.g., 'BTC')
        timeframe: Trading timeframe ('short', 'medium', 'long')
        output_path: Path to save the generated notebook (optional)
    
    Returns:
        Tuple of (notebook_path, process)
    
    Raises:
        NotebookGeneratorError: If there's an error generating or launching the notebook
    """
    # Generate the notebook
    notebook_path = generate_analysis_notebook(symbol, timeframe, output_path)
    
    # Launch the notebook
    process = launch_notebook(notebook_path)
    
    return notebook_path, process 