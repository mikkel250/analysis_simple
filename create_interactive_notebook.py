#!/usr/bin/env python
# Generate a properly formatted interactive analysis notebook

import nbformat as nbf
import os

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = []

# Markdown cell for introduction
markdown1 = """# Interactive Cryptocurrency Analysis

This notebook provides interactive analysis of cryptocurrency data using the enhanced display components.
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

# Import the modules
from src.jupyter.widgets import set_notebook_width, create_quick_analysis_widget
"""
cells.append(nbf.v4.new_code_cell(code1))

# Code cell for setting notebook width
code2 = """# Set notebook to full width for better display
set_notebook_width('100%')
print("Notebook width set to 100%")
"""
cells.append(nbf.v4.new_code_cell(code2))

# Code cell for creating quick analysis widget
code3 = """# Create and display the quick analysis widget
quick_widget = create_quick_analysis_widget()
quick_widget
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
output_file = "notebooks/examples/interactive_analysis.ipynb"
nbf.write(nb, output_file)

print(f"Interactive analysis notebook created successfully at {output_file}") 