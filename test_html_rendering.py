import os
import sys

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.jupyter.display import create_summary_dashboard

# Create test data with varied market signals
analysis_results = {
    "metadata": {
        "symbol": "BTC",
        "vs_currency": "usd",
        "timeframe": "1d"
    },
    "price_data": {
        "current_price": 48750.25,
        "price_change_24h": 1200.50,
        "price_change_percentage_24h": 2.53
    },
    "summary": {
        "trend": {
            "direction": "bullish",
            "strength": "moderate",
            "analysis": "BTC is showing a moderate bullish trend with increased buying volume."
        },
        "signals": {
            "action": "buy",
            "short_term": "bullish",
            "medium_term": "neutral",
            "long_term": "bearish"
        }
    }
}

# Create the dashboard with fixed HTML rendering
fig = create_summary_dashboard(analysis_results)

# Display the figure
fig.show()

print("Dashboard with fixed HTML rendering created successfully!") 