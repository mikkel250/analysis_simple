import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = []

# Markdown cell 1
markdown1 = """# Interactive Cryptocurrency Analysis Demo

This notebook demonstrates how to use the interactive widgets for cryptocurrency analysis. The widgets allow you to:

- Select symbols, timeframes, and date ranges
- Adjust indicator parameters
- Generate various chart types
- Create multi-currency comparisons

Let's start by importing the necessary modules."""
cells.append(nbf.v4.new_markdown_cell(markdown1))

# Code cell 1 - Add path fix
code1 = """# Add the project root to the Python path so we can import the src module
import sys
import os

# Go up two levels from the notebook directory to get to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))

# Add to path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Import the widgets module
from src.jupyter.widgets import (
    create_analysis_dashboard,
    create_quick_analysis_widget,
    create_comparison_widget,
    set_notebook_width
)

# Import the display functions if you want to use them directly
from src.jupyter.display import (
    create_price_chart,
    create_indicator_chart,
    create_summary_dashboard,
    create_multi_indicator_chart
)

# Import the analysis functions if you want to access data directly
from src.jupyter.analysis import (
    run_analysis,
    get_data,
    clear_cache,
    batch_analyze
)

# Set notebook to full width for better display
set_notebook_width('100%')
"""
cells.append(nbf.v4.new_code_cell(code1))

# Markdown cell 2
markdown2 = """## Quick Analysis Widget

Let's start with the simplest interface - the quick analysis widget. This creates a simplified UI with just the essential controls.

**Note:** If you want to use the forecasting feature, make sure to install the `statsmodels` package:
```
pip install statsmodels
```
"""
cells.append(nbf.v4.new_markdown_cell(markdown2))

# Code cell 2
code2 = """# Create the quick analysis widget
quick_widget = create_quick_analysis_widget()

# Display the widget
quick_widget"""
cells.append(nbf.v4.new_code_cell(code2))

# Markdown cell 3
markdown3 = """## Complete Analysis Dashboard

The `create_analysis_dashboard` function creates a more comprehensive UI with tabs for basic settings, indicator parameters, and display options. You can select a symbol, timeframe, and chart type, and then click "Run Analysis" to generate the chart."""
cells.append(nbf.v4.new_markdown_cell(markdown3))

# Code cell 3
code3 = """# Create the analysis dashboard
controls, dashboard = create_analysis_dashboard()

# Display the dashboard
dashboard"""
cells.append(nbf.v4.new_code_cell(code3))

# Markdown cell 4
markdown4 = """## Multi-Currency Comparison

The `create_comparison_widget` function creates a widget for comparing multiple cryptocurrencies. You can select multiple symbols and see a comparison table and mini-dashboards for each selected symbol."""
cells.append(nbf.v4.new_markdown_cell(markdown4))

# Code cell 4
code4 = """# Create the comparison widget
comparison_widget = create_comparison_widget()

# Display the widget
comparison_widget"""
cells.append(nbf.v4.new_code_cell(code4))

# Markdown cell 5
markdown5 = """## Direct API Usage Example

If you prefer to work with the API directly without using the widgets, you can use the analysis and display functions directly. Here's a simple example:"""
cells.append(nbf.v4.new_markdown_cell(markdown5))

# Code cell 5
code5 = """# Get data for Bitcoin
df, current_price_data = get_data('BTC', '1d', 100)

# Run analysis
analysis_results = run_analysis('BTC', '1d', 100)

# Create a price chart
price_chart = create_price_chart(df, title='BTC/USD Price Chart')
price_chart"""
cells.append(nbf.v4.new_code_cell(code5))

# Add cells to notebook
nb['cells'] = cells

# Write the notebook to a file
nbf.write(nb, 'notebooks/examples/crypto_widgets_demo.ipynb')

print("Notebook created successfully!") 