#!/usr/bin/env python
# Test script for the enhanced display components

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from src.jupyter.widgets import set_notebook_width
from src.jupyter.display import create_summary_dashboard

# Generate mock analysis results for testing
def generate_mock_analysis():
    """Generate mock analysis results for testing the dashboard"""
    
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
                "analysis": """
                Bitcoin has shown resilience above key support levels and appears to be gaining momentum.
                RSI indicates favorable conditions for continued upward movement, while volume profiles
                suggest accumulation by larger players. Watch for resistance around $48,000.
                """,
                "short_term": "bullish",
                "medium_term": "bullish",
                "long_term": "neutral"
            },
            "signals": {
                "action": "buy",
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

# Function to run the test
def run_test():
    """Run the test for the enhanced display"""
    
    # Set up display settings for full width
    from IPython.display import display
    set_notebook_width('100%')
    
    # Generate mock analysis
    analysis_results = generate_mock_analysis()
    
    # Create and display the dashboard
    fig = create_summary_dashboard(analysis_results)
    display(fig)
    
    print("Test completed. Check the displayed dashboard for correct formatting.")

# When run directly
if __name__ == "__main__":
    # Try to display the test dashboard
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            # We're in a notebook environment
            run_test()
        else:
            print("This script needs to be run in a Jupyter notebook environment.")
            print("Please run this in a notebook with: %run test_enhanced_display.py")
    except ImportError:
        print("IPython not available. Please run this script in a Jupyter notebook.") 