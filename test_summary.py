#!/usr/bin/env python3
from src.jupyter.market_analyzer import MarketAnalyzer
import json

# Create analyzer with same settings as command
analyzer = MarketAnalyzer(symbol="BTC", timeframe="short")

# Get summary and display it
summary = analyzer.get_summary()
print("Summary keys:", summary.keys())
print("\nTrading style data:")
print(json.dumps(summary.get('trading_style', {}), indent=2))

print("\nTimeframe from summary:", summary.get('timeframe', 'None')) 