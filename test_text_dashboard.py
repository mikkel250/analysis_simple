#!/usr/bin/env python
# Test script for the HTML-based text dashboard

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
from src.jupyter.text_display import create_text_dashboard
from IPython.display import display

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
            "current_price": 46750.25,
            "price_change_24h": 1250.75,
            "price_change_percentage_24h": 2.75,
            "market_cap": 912568000000,
            "volume_24h": 45689000000,
            "high_24h": 47150.00,
            "low_24h": 45280.50,
            "price_change_percentage_7d": 5.2,
            "price_change_percentage_30d": -2.8,
            "price_change_percentage_1y": 48.3
        },
        "market_trends": {
            "short_term": "bullish",
            "medium_term": "bullish",
            "long_term": "neutral"
        },
        "signals": {
            "trend_signals": {
                "moving_average": "bullish",
                "macd": "bullish",
                "parabolic_sar": "bullish",
                "bollinger_bands": "neutral",
                "ichimoku_cloud": "bullish"
            },
            "oscillator_signals": {
                "rsi": "neutral",
                "stochastic": "bearish",
                "cci": "bullish",
                "williams_r": "bearish",
                "awesome_oscillator": "bullish"
            }
        },
        "patterns": {
            "hammer": {
                "signal": "bullish",
                "strength": 0.8
            },
            "evening_star": {
                "signal": "bearish",
                "strength": 0.7
            },
            "bullish_engulfing": {
                "signal": "bullish",
                "strength": 0.9
            },
            "three_black_crows": {
                "signal": "bearish",
                "strength": 0.6
            }
        },
        "recommendation": {
            "action": "buy",
            "confidence": 0.75,
            "rationale": "Strong bullish trends in the short and medium term, confirmed by multiple technical indicators. However, some oscillators show overbought conditions, suggesting caution. Recent bullish candlestick patterns provide additional confirmation."
        }
    }

if __name__ == "__main__":
    # Only run this as a script, not when imported
    try:
        from IPython import get_ipython
        
        # If in Jupyter notebook, configure display and show dashboard
        if get_ipython() is not None:
            # Set notebook width for better display
            set_notebook_width('100%')
            
            # Generate mock analysis
            analysis_results = generate_mock_analysis()
            
            # Create and display the HTML dashboard
            dashboard = create_text_dashboard(analysis_results)
            display(dashboard)
            
            print("Dashboard displayed successfully!")
        else:
            print("This script is designed to be run in a Jupyter notebook environment.")
            print("Run 'jupyter notebook test_text_dashboard.py' to view the dashboard.")
    except ImportError:
        print("IPython not available. This script is designed to be run in a Jupyter notebook environment.")
        print("Run 'jupyter notebook test_text_dashboard.py' to view the dashboard.") 