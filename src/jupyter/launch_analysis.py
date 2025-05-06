#!/usr/bin/env python3
"""
Market Analysis Notebook Launcher

This script provides a user-friendly way to generate and launch
market analysis notebooks for any symbol and timeframe.
"""

import os
import sys
import argparse
import re
import json
from typing import Optional
from pathlib import Path

# Add the project root to the Python path if running as a script
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

from src.jupyter.notebook_generator import (
    generate_and_launch_notebook,
    NotebookGeneratorError
)


def validate_symbol(symbol: str) -> bool:
    """
    Validate that the symbol is properly formatted.
    
    Args:
        symbol: Market symbol to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Symbol should be alphanumeric, optionally with hyphens or underscores
    pattern = r'^[A-Za-z0-9_-]+$'
    return bool(re.match(pattern, symbol))


def validate_output_path(output_path: Optional[str]) -> bool:
    """
    Validate that the output path is valid.
    
    Args:
        output_path: Path to save the notebook
        
    Returns:
        True if valid, False otherwise
    """
    if output_path is None:
        return True
    
    try:
        # Check if the directory exists or can be created
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return True
    except Exception:
        return False


def create_demo_template(template_path: str, timeframe: str):
    """
    Create a demo template if it doesn't exist.
    
    Args:
        template_path: Path to the template file
        timeframe: Trading timeframe (short, medium, long)
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
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(template_path), exist_ok=True)
    
    # Write the template to the file
    with open(template_path, 'w') as f:
        nbformat.write(nb, f)


def ensure_templates_exist():
    """Ensure that all template notebooks exist."""
    templates_dir = Path('notebooks/templates')
    templates = {
        'short': templates_dir / 'short_term_analysis.ipynb',
        'medium': templates_dir / 'medium_term_analysis.ipynb',
        'long': templates_dir / 'long_term_analysis.ipynb'
    }
    
    for timeframe, path in templates.items():
        if not path.exists() or path.stat().st_size < 10:  # Check if file is empty or nearly empty
            print(f"Creating {timeframe} template notebook...")
            create_demo_template(str(path), timeframe)


def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate and launch market analysis notebooks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "symbol",
        type=str,
        help="Market symbol to analyze (e.g., BTC, ETH)"
    )
    
    parser.add_argument(
        "-t", "--timeframe",
        type=str,
        choices=["short", "medium", "long"],
        default="medium",
        help="Trading timeframe for analysis"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for the generated notebook (optional)"
    )
    
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Generate the notebook but don't launch it"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Ensure template notebooks exist
    ensure_templates_exist()
    
    # Parse command-line arguments
    args = parse_args()
    
    # Validate symbol
    if not validate_symbol(args.symbol):
        print(f"Error: Invalid symbol '{args.symbol}'")
        print("Symbol should contain only letters, numbers, hyphens, or underscores")
        return 1
    
    # Validate output path if provided
    if args.output and not validate_output_path(args.output):
        print(f"Error: Invalid output path '{args.output}'")
        print("Please provide a valid path where you have write permissions")
        return 1
    
    try:
        # Generate and optionally launch the notebook
        if args.no_launch:
            # Only generate without launching
            from src.jupyter.notebook_generator import generate_analysis_notebook
            notebook_path = generate_analysis_notebook(
                symbol=args.symbol,
                timeframe=args.timeframe,
                output_path=args.output
            )
            print(f"Generated notebook: {notebook_path}")
            print(f"Use 'jupyter notebook {notebook_path}' to open it")
        else:
            # Generate and launch
            notebook_path, process = generate_and_launch_notebook(
                symbol=args.symbol,
                timeframe=args.timeframe,
                output_path=args.output
            )
            print(f"Generated and launched notebook: {notebook_path}")
            print("Jupyter server is running. Press Ctrl+C to stop.")
            
            # Wait for the process to complete (or user interrupt)
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nStopping Jupyter server...")
                process.terminate()
        
        return 0
    
    except NotebookGeneratorError as e:
        print(f"Error: {str(e)}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 