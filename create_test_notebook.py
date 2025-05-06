#!/usr/bin/env python
# Generate a properly formatted test notebook for dashboard testing

import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = []

# Markdown cell for introduction
markdown1 = """# Dashboard Display Test

This notebook tests the enhanced dashboard display with dark background and improved formatting.
"""
cells.append(nbf.v4.new_markdown_cell(markdown1))

# Code cell for setup and imports
code1 = """# Import necessary modules
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Import the necessary modules
from src.jupyter.widgets import set_notebook_width
"""
cells.append(nbf.v4.new_code_cell(code1))

# Code cell for setting notebook width
code2 = """# Set notebook to full width for better display
set_notebook_width('100%')
print("Notebook width set to 100%")
"""
cells.append(nbf.v4.new_code_cell(code2))

# Code cell for running the test script
code3 = """# Run the test script to display the dashboard
import sys
import os

# Make sure to import from project root
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Import dashboard creation functions
from src.jupyter.display import create_summary_dashboard

# Generate mock analysis results for testing
def generate_mock_analysis():
    \"\"\"Generate mock analysis results for testing the dashboard\"\"\"
    
    return {
        "metadata": {
            "symbol": "BTC",
            "vs_currency": "usd",
            "timeframe": "1d",
            "last_updated": datetime.now().isoformat()
        },
        "price_data": {
            "current_price": 45000.25,
            "market_cap": 850000000000,
            "price_change_24h": 1250.75,
            "price_change_percentage_24h": 2.85
        },
        "summary": {
            "trend": {
                "direction": "bullish",
                "strength": "moderate",
                "analysis": \"\"\"
                Bitcoin has shown resilience above key support levels and appears to be gaining momentum.
                RSI indicates favorable conditions for continued upward movement, while volume profiles
                suggest accumulation by larger players. Watch for resistance around $48,000.
                \"\"\",
                "short_term": "bullish",
                "medium_term": "bullish",
                "long_term": "neutral"
            },
            "signals": {
                "action": "buy",
                "short_term": "bullish",
                "medium_term": "bullish",
                "long_term": "neutral",
                "confidence": "medium"
            }
        },
        "momentum_indicators": {
            "rsi": {
                "value": 65.23,
                "signal": "neutral",
                "params": {"length": 14}
            }
        },
        "trend_indicators": {
            "sma": {
                "value": 42000.50,
                "signal": "bullish",
                "params": {"length": 20}
            },
            "ema": {
                "value": 43500.75,
                "signal": "bullish", 
                "params": {"length": 20}
            },
            "macd": {
                "macd_line": 250.25,
                "signal_line": 180.50,
                "histogram": 69.75,
                "signal": "bullish"
            }
        },
        "volatility_indicators": {
            "bbands": {
                "upper": 47000.00,
                "middle": 43000.00,
                "lower": 39000.00,
                "width": 18.60,
                "signal": "neutral" 
            }
        }
    }

# Generate mock analysis
analysis_results = generate_mock_analysis()

# Create and display the dashboard
fig = create_summary_dashboard(analysis_results)
fig.show()

print("Dashboard displayed. Check for correct dark background and formatting.")
"""
cells.append(nbf.v4.new_code_cell(code3))

# Add cells to notebook
nb['cells'] = cells

# Set notebook metadata
nb['metadata'] = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3
        },
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.8.5'
    }
}

# Make sure the directory exists
os.makedirs("notebooks/examples", exist_ok=True)

# Write the notebook to a file
output_file = "notebooks/examples/test_dashboard.ipynb"
nbf.write(nb, output_file)

print(f"Test notebook created successfully at {output_file}") 