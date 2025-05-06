"""
Jupyter Command Handler

Handles the 'jupyter' command for launching Jupyter notebooks with the analysis modules.
"""

import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path

import typer
from src.jupyter.notebook_generator import (
    generate_analysis_notebook,
    generate_and_launch_notebook,
    NotebookGeneratorError
)

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
jupyter_app = typer.Typer()


@jupyter_app.callback()
def callback():
    """
    Launch Jupyter notebooks for interactive cryptocurrency analysis.
    """
    pass


@jupyter_app.command()
def start(
    example: bool = typer.Option(False, "--example", "-e", help="Open example notebooks"),
    port: int = typer.Option(8888, "--port", "-p", help="Port to run Jupyter server on"),
    notebook_dir: str = typer.Option("", "--dir", "-d", help="Custom directory to start Jupyter in")
):
    """
    Start a Jupyter notebook server with the crypto analysis environment.
    """
    try:
        # Check if jupyter is installed
        if not shutil.which("jupyter"):
            typer.secho("Jupyter is not installed. Please install it with:", fg=typer.colors.RED)
            typer.secho("pip install -r requirements-jupyter.txt", fg=typer.colors.YELLOW)
            return 1
        
        # Get project root for PYTHONPATH
        project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from this file
        
        # Determine the directory to start Jupyter in
        if notebook_dir:
            jupyter_dir = notebook_dir
        elif example:
            # Use examples directory
            jupyter_dir = str(project_root / "notebooks" / "examples")
        else:
            # Use current directory
            jupyter_dir = os.getcwd()
        
        # Validate the directory
        if not os.path.isdir(jupyter_dir):
            typer.secho(f"Directory not found: {jupyter_dir}", fg=typer.colors.RED)
            return 1
        
        # Construct the command
        cmd = [
            "jupyter", "notebook",
            f"--port={port}",
            "--no-browser" if not example else "",
            jupyter_dir
        ]
        
        # Filter out empty strings
        cmd = [c for c in cmd if c]
        
        # Print info
        typer.secho("Starting Jupyter notebook server...", fg=typer.colors.GREEN)
        typer.secho(f"Directory: {jupyter_dir}", fg=typer.colors.BLUE)
        typer.secho(f"Port: {port}", fg=typer.colors.BLUE)
        
        # If showing examples, print the available examples
        if example:
            examples_path = Path(jupyter_dir)
            example_files = [f for f in examples_path.glob("*.ipynb") if f.is_file()]
            
            if example_files:
                typer.secho("\nAvailable example notebooks:", fg=typer.colors.GREEN)
                for i, f in enumerate(sorted(example_files), 1):
                    typer.secho(f"  {i}. {f.name}", fg=typer.colors.YELLOW)
            else:
                typer.secho("\nNo example notebooks found in the examples directory.", fg=typer.colors.YELLOW)
        
        # Start the process with PYTHONPATH set to include project root
        typer.secho("\nPress Ctrl+C to stop the server when finished.", fg=typer.colors.RED)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        subprocess.run(cmd, env=env)
        
        return 0
        
    except Exception as e:
        typer.secho(f"Error starting Jupyter: {str(e)}", fg=typer.colors.RED)
        logger.exception("Error in jupyter command")
        return 1


@jupyter_app.command()
def demo():
    """
    Launch a demo notebook with cryptocurrency visualizations.
    """
    try:
        # Get project root for PYTHONPATH
        project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from this file
        demo_path = project_root / "notebooks" / "examples" / "crypto_visualization_demo.ipynb"
        
        # Check if the demo notebook exists
        if not demo_path.exists():
            typer.secho(f"Demo notebook not found: {demo_path}", fg=typer.colors.RED)
            return 1
        
        # Construct the command
        cmd = [
            "jupyter", "notebook",
            str(demo_path)
        ]
        
        # Print info
        typer.secho("Launching visualization demo notebook...", fg=typer.colors.GREEN)
        
        # Start the process with PYTHONPATH set to include project root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        subprocess.run(cmd, env=env)
        
        return 0
        
    except Exception as e:
        typer.secho(f"Error launching demo: {str(e)}", fg=typer.colors.RED)
        logger.exception("Error in jupyter demo command")
        return 1


@jupyter_app.command()
def run(
    notebook: str = typer.Argument(..., help="Name of the example notebook to run"),
    create_if_missing: bool = typer.Option(False, "--create", "-c", help="Create the notebook if it doesn't exist")
):
    """
    Run a specific example notebook by name.
    
    Example: python -m src.main jupyter run price_analysis
    """
    try:
        # Check if jupyter is installed
        if not shutil.which("jupyter"):
            typer.secho("Jupyter is not installed. Please install it with:", fg=typer.colors.RED)
            typer.secho("pip install -r requirements-jupyter.txt", fg=typer.colors.YELLOW)
            return 1
        
        # Get the examples directory path
        project_root = Path(__file__).resolve().parents[3]  # Go up 3 levels from this file
        examples_dir = project_root / "notebooks" / "examples"
        
        # Ensure examples directory exists
        if not examples_dir.exists():
            examples_dir.mkdir(parents=True, exist_ok=True)
            typer.secho(f"Created examples directory: {examples_dir}", fg=typer.colors.GREEN)
        
        # Add .ipynb extension if not provided
        if not notebook.endswith(".ipynb"):
            notebook += ".ipynb"
            
        # Get the notebook path
        notebook_path = examples_dir / notebook
        
        # Check if the notebook exists
        if not notebook_path.exists():
            if create_if_missing:
                # Create a basic template notebook for the user
                from nbformat import v4 as nbf
                
                # Create a new notebook with our crypto analysis imports and example code
                new_notebook = nbf.new_notebook()
                
                # Add markdown cell with title
                new_notebook.cells.append(nbf.new_markdown_cell(f"# {notebook[:-6]} Example\n\nThis notebook demonstrates cryptocurrency analysis using the analysis_simple library."))
                
                # Add code cell for system path configuration
                path_setup_cell = """# Add project root to Python path to find the src module
import sys
import os
from pathlib import Path

# Try to find the project root relative to this notebook
notebook_dir = Path().absolute()
project_root = notebook_dir.parents[1] if notebook_dir.name == 'examples' and notebook_dir.parent.name == 'notebooks' else notebook_dir

# Add to Python path if not already there
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")
"""
                new_notebook.cells.append(nbf.new_code_cell(path_setup_cell))
                
                # Add code cell with imports
                imports_cell = """# Import necessary modules
try:
    from src.jupyter.display import create_price_chart, create_indicator_chart
    from src.jupyter.analysis import run_analysis, get_price_data
    import pandas as pd
    import plotly.io as pio

    # Enable Plotly in Jupyter
    pio.renderers.default = "notebook"
    print("Successfully imported modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the project root is in your Python path")
"""
                new_notebook.cells.append(nbf.new_code_cell(imports_cell))
                
                # Add example code cell
                example_cell = """# Get sample data for Bitcoin
btc_data = get_price_data("bitcoin", days=30, vs_currency="usd")

# Display price chart
create_price_chart(btc_data, title="Bitcoin Price (30 Days)")
"""
                new_notebook.cells.append(nbf.new_code_cell(example_cell))
                
                # Write the notebook to file
                import nbformat
                with open(notebook_path, 'w') as f:
                    nbformat.write(new_notebook, f)
                
                typer.secho(f"Created new notebook: {notebook_path}", fg=typer.colors.GREEN)
            else:
                # List available notebooks
                example_files = [f.name for f in examples_dir.glob("*.ipynb") if f.is_file()]
                
                typer.secho(f"Notebook not found: {notebook}", fg=typer.colors.RED)
                
                if example_files:
                    typer.secho("\nAvailable example notebooks:", fg=typer.colors.GREEN)
                    for i, f in enumerate(sorted(example_files), 1):
                        typer.secho(f"  {i}. {f}", fg=typer.colors.YELLOW)
                    
                    typer.secho("\nUse --create flag to create a new notebook.", fg=typer.colors.BLUE)
                else:
                    typer.secho("\nNo example notebooks found. Use --create flag to create a new notebook.", fg=typer.colors.BLUE)
                
                return 1
        
        # Construct the command
        cmd = [
            "jupyter", "notebook",
            str(notebook_path)
        ]
        
        # Print info
        typer.secho(f"Launching notebook: {notebook}", fg=typer.colors.GREEN)
        
        # Start the process with PYTHONPATH set to include project root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root)
        subprocess.run(cmd, env=env)
        
        return 0
        
    except Exception as e:
        typer.secho(f"Error running notebook: {str(e)}", fg=typer.colors.RED)
        logger.exception("Error in jupyter run command")
        return 1


@jupyter_app.command()
def analysis(
    symbol: str = typer.Argument(..., help="Symbol to analyze (e.g., BTC, ETH)"),
    timeframe: str = typer.Option(
        "short", 
        "--timeframe", 
        "-t", 
        help="Trading timeframe (short, medium, long)"
    ),
    output: str = typer.Option(
        None, 
        "--output", 
        "-o", 
        help="Output path for the generated notebook (optional)"
    ),
    no_launch: bool = typer.Option(
        False, 
        "--no-launch", 
        help="Generate the notebook but don't launch it"
    )
):
    """
    Generate and launch a market analysis notebook for a specific symbol.
    
    This command leverages Jupyter notebooks to provide interactive analysis
    of market data with customized templates for different trading timeframes.
    """
    try:
        # Validate symbol
        import re
        if not bool(re.match(r'^[A-Za-z0-9_-]+$', symbol)):
            typer.secho(f"Error: Invalid symbol '{symbol}'", fg=typer.colors.RED)
            typer.secho("Symbol should contain only letters, numbers, hyphens, or underscores", fg=typer.colors.YELLOW)
            return 1
            
        # Validate timeframe
        valid_timeframes = ["short", "medium", "long"]
        if timeframe not in valid_timeframes:
            typer.secho(f"Error: Invalid timeframe '{timeframe}'", fg=typer.colors.RED)
            typer.secho(f"Timeframe must be one of: {', '.join(valid_timeframes)}", fg=typer.colors.YELLOW)
            return 1
            
        # Ensure template notebooks exist
        from pathlib import Path
        templates_dir = Path('notebooks/templates')
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        templates = {
            'short': templates_dir / 'short_term_analysis.ipynb',
            'medium': templates_dir / 'medium_term_analysis.ipynb',
            'long': templates_dir / 'long_term_analysis.ipynb'
        }
        
        for tf, path in templates.items():
            if not path.exists() or path.stat().st_size < 10:  # Check if file is empty or nearly empty
                from src.jupyter.launch_analysis import create_demo_template
                typer.secho(f"Creating {tf} template notebook...", fg=typer.colors.BLUE)
                create_demo_template(str(path), tf)
        
        # Generate and optionally launch the notebook
        if no_launch:
            # Only generate without launching
            notebook_path = generate_analysis_notebook(
                symbol=symbol,
                timeframe=timeframe,
                output_path=output
            )
            typer.secho(f"Generated notebook: {notebook_path}", fg=typer.colors.GREEN)
            typer.secho(f"Use 'jupyter notebook {notebook_path}' to open it", fg=typer.colors.YELLOW)
        else:
            # Generate and launch
            notebook_path, process = generate_and_launch_notebook(
                symbol=symbol,
                timeframe=timeframe,
                output_path=output
            )
            typer.secho(f"Generated and launched notebook: {notebook_path}", fg=typer.colors.GREEN)
            typer.secho("Jupyter server is running. Press Ctrl+C to stop.", fg=typer.colors.YELLOW)
            
            # Wait for the process to complete (or user interrupt)
            try:
                process.wait()
            except KeyboardInterrupt:
                typer.secho("\nStopping Jupyter server...", fg=typer.colors.YELLOW)
                process.terminate()
        
        return 0
        
    except NotebookGeneratorError as e:
        typer.secho(f"Error: {str(e)}", fg=typer.colors.RED)
        return 1
    except Exception as e:
        import traceback
        typer.secho(f"Unexpected error: {str(e)}", fg=typer.colors.RED)
        logger.exception("Unexpected error in jupyter analysis command")
        return 1


if __name__ == "__main__":
    jupyter_app() 