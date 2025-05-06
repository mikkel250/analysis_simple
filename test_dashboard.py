import os
import sys
import json

# Add the src directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.jupyter.display import create_summary_dashboard

# Create a sample analysis_results dictionary
analysis_results = {
    "metadata": {
        "symbol": "BTC",
        "vs_currency": "usd",
        "timeframe": "1d"
    },
    "price_data": {
        "current_price": 50000,
        "price_change_24h": 1000,
        "price_change_percentage_24h": 2.0
    },
    "summary": {
        "trend": {
            "direction": "bullish",
            "strength": "moderate",
            "analysis": "The market is showing a moderate bullish trend."
        },
        "signals": {
            "action": "buy",
            "short_term": "bullish",
            "medium_term": "bullish",
            "long_term": "neutral"
        }
    }
}

# Create the dashboard
fig = create_summary_dashboard(analysis_results)

# Test if it works by showing the figure
fig.show()

print("Dashboard created successfully!") 