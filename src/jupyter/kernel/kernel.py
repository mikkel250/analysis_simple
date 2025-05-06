"""
Core implementation of the Financial Analysis Jupyter kernel.

This module implements the main kernel class that extends IPython's Kernel
with custom functionality for financial market data analysis.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ipykernel.kernelbase import Kernel
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display, Javascript

# Pre-load common financial libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas_ta as ta

# Import market data and trading style modules
from src.jupyter.kernel import __version__
from src.jupyter.kernel import market_data
from src.jupyter.kernel import trading_styles

class FinancialAnalysisKernel(Kernel):
    """
    Custom Jupyter kernel for financial market data analysis.
    
    This kernel automatically includes project paths, pre-loads financial
    data libraries, and provides custom magics for different trading timeframes.
    """
    implementation = 'financial_analysis'
    implementation_version = __version__
    language = 'python'
    language_version = '3.8+'
    language_info = {
        'name': 'python',
        'mimetype': 'text/x-python',
        'file_extension': '.py',
        'pygments_lexer': 'ipython3',
        'version': '3.8+',
        'codemirror_mode': {
            'name': 'python',
            'version': 3
        }
    }
    banner = "Financial Analysis Kernel"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize the IPython shell
        self.shell = InteractiveShell.instance()
        
        # Add project path to sys.path
        self._add_project_path()
        
        # Register custom magics
        self._register_magics()
        
        # Set up auto-execution for cells
        self._setup_auto_execution()
        
    def _add_project_path(self):
        """Add the project root to the Python path."""
        # Get project path from environment or use current directory
        project_path = os.environ.get('PROJECT_PATH', os.getcwd())
        
        # Add to sys.path if not already there
        if project_path not in sys.path:
            sys.path.insert(0, project_path)
            
        # Also add the parent directories for proper imports
        parent_dir = str(Path(project_path).parent.absolute())
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
    
    def _register_magics(self):
        """Register custom magics for trading styles."""
        # Register line magics for trading styles
        self.shell.register_magic_function(
            trading_styles.short_magic, 
            magic_kind='line', 
            magic_name='short'
        )
        self.shell.register_magic_function(
            trading_styles.medium_magic,
            magic_kind='line',
            magic_name='medium'
        )
        self.shell.register_magic_function(
            trading_styles.long_magic,
            magic_kind='line',
            magic_name='long'
        )
    
    def _setup_auto_execution(self):
        """Set up auto-execution of cells when a notebook is loaded."""
        # JavaScript for auto-executing cells with advanced features
        auto_exec_js = """
        require(['base/js/namespace', 'jquery'], function(Jupyter, $) {
            // Configuration options
            var config = {
                autoRunOnLoad: true,           // Run all cells when notebook loads
                runCellsWithTags: true,        // Run cells with specific tags
                autoRunTags: ['autorun'],      // Tags for cells to auto-run
                skipTaggedCells: false,        // Skip cells with specific tags
                skipTags: ['skip', 'norun'],   // Tags for cells to skip
                runTimeout: 500,               // Timeout before running cells (ms)
                cellExecutionDelay: 100,       // Delay between cell executions (ms)
                notifyCompletion: true,        // Show notification when execution completes
                scrollToEnd: true              // Scroll to the end after execution
            };
            
            // Track if we've already auto-executed to avoid duplicates
            window._hasAutoExecutedCells = window._hasAutoExecutedCells || false;
            
            // Function to run all cells
            function autoRunCells() {
                if (window._hasAutoExecutedCells) {
                    console.log('Auto-execution already performed, skipping...');
                    return;
                }
                
                console.log('Auto-executing cells...');
                
                // If we need to be selective about which cells to run
                if (config.runCellsWithTags || config.skipTaggedCells) {
                    var cells = Jupyter.notebook.get_cells();
                    var cellsToRun = [];
                    
                    // Collect cells that should be run
                    for (var i = 0; i < cells.length; i++) {
                        var cell = cells[i];
                        var metadata = cell.metadata || {};
                        var tags = metadata.tags || [];
                        
                        var shouldSkip = config.skipTaggedCells && 
                            tags.some(tag => config.skipTags.includes(tag));
                            
                        var shouldRun = !config.runCellsWithTags || 
                            tags.some(tag => config.autoRunTags.includes(tag));
                            
                        if (shouldRun && !shouldSkip) {
                            cellsToRun.push(i);
                        }
                    }
                    
                    // Execute cells sequentially with delay
                    function runCellAt(index) {
                        if (index < cellsToRun.length) {
                            Jupyter.notebook.select(cellsToRun[index]);
                            Jupyter.notebook.execute_cell();
                            
                            setTimeout(function() {
                                runCellAt(index + 1);
                            }, config.cellExecutionDelay);
                        } else {
                            // All cells executed
                            if (config.scrollToEnd) {
                                $('html, body').animate({
                                    scrollTop: $(document).height()
                                }, 1000);
                            }
                            
                            if (config.notifyCompletion) {
                                var div = $('<div>')
                                    .addClass('alert alert-success')
                                    .text('Auto-execution complete!')
                                    .css({
                                        position: 'fixed',
                                        top: '10px',
                                        right: '10px',
                                        zIndex: 9999,
                                        padding: '10px',
                                        borderRadius: '5px',
                                        opacity: 0.9
                                    });
                                
                                $('body').append(div);
                                
                                setTimeout(function() {
                                    div.fadeOut(function() { $(this).remove(); });
                                }, 3000);
                            }
                        }
                    }
                    
                    // Start the sequence
                    if (cellsToRun.length > 0) {
                        runCellAt(0);
                    }
                } else {
                    // Just run all cells
                    Jupyter.actions.call('jupyter-notebook:run-all-cells');
                    
                    if (config.scrollToEnd) {
                        setTimeout(function() {
                            $('html, body').animate({
                                scrollTop: $(document).height()
                            }, 1000);
                        }, cells.length * config.cellExecutionDelay + 500);
                    }
                }
                
                window._hasAutoExecutedCells = true;
            }
            
            // Add a button to the toolbar for manual triggering
            function addAutorunButton() {
                if (!$('#autorun_button').length) {
                    var button = $('<button>')
                        .attr('id', 'autorun_button')
                        .addClass('btn btn-default')
                        .attr('title', 'Auto-run cells')
                        .text('Auto-run')
                        .click(function() {
                            window._hasAutoExecutedCells = false;
                            autoRunCells();
                        });
                    
                    $('#maintoolbar-container').append(button);
                }
            }
            
            // Run when the kernel is ready
            $(Jupyter.events).on("kernel_ready.Kernel", function() {
                console.log('Kernel ready, setting up auto-execution...');
                addAutorunButton();
                
                if (config.autoRunOnLoad) {
                    setTimeout(autoRunCells, config.runTimeout);
                }
            });
            
            // Also run on notebook loaded event as a backup
            $(Jupyter.events).on("notebook_loaded.Notebook", function() {
                console.log('Notebook loaded, setting up auto-execution...');
                addAutorunButton();
                
                if (config.autoRunOnLoad) {
                    setTimeout(autoRunCells, config.runTimeout);
                }
            });
        });
        """
        
        # Display the JavaScript to run when the notebook is loaded
        display(Javascript(auto_exec_js))
    
    def do_execute(
        self, 
        code: str, 
        silent: bool, 
        store_history: bool = True, 
        user_expressions: Optional[Dict[str, str]] = None, 
        allow_stdin: bool = False
    ) -> Dict[str, Any]:
        """
        Execute user code.
        
        This method is called when the user executes a code cell in the notebook.
        """
        # Execute the code in the IPython shell
        reply_content = self.shell.run_cell(code, silent=silent, store_history=store_history)
        
        if reply_content.error_before_exec or reply_content.error_in_exec:
            # If there was an error, return an error message
            return {
                'status': 'error',
                'execution_count': self.execution_count,
                'ename': str(reply_content.error_in_exec.__class__.__name__ 
                            if reply_content.error_in_exec else 'Error'),
                'evalue': str(reply_content.error_in_exec 
                             if reply_content.error_in_exec else 'Error during execution'),
                'traceback': []
            }
        
        # If execution was successful
        return {
            'status': 'ok',
            'execution_count': self.execution_count,
            'payload': [],
            'user_expressions': {}
        }

def main():
    """Launch the kernel."""
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=FinancialAnalysisKernel)

if __name__ == '__main__':
    main()
