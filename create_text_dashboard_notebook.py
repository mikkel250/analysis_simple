#!/usr/bin/env python
# Generate a properly formatted text dashboard demo notebook

import nbformat as nbf
import os
import sys

# Create a new notebook
nb = nbf.v4.new_notebook()

# Create cells
cells = []

# Markdown cell for introduction
markdown1 = """# Text-Based Cryptocurrency Analysis Dashboard

This notebook demonstrates the HTML-based text dashboard for cryptocurrency analysis. This approach provides a simpler, more reliable alternative to complex Plotly visualizations while maintaining all the key information.

## Educational Accordions Feature

This dashboard includes educational accordions for each technical indicator. Click on any indicator name to expand an accordion that explains:
- What the indicator is and its purpose
- How it's calculated (with formula)
- What the current value means
- Why it's showing a particular signal (bullish/bearish/neutral)

This feature makes the dashboard both informative and educational for users of all experience levels."""
cells.append(nbf.v4.new_markdown_cell(markdown1))

# Code cell for setup and imports
code1 = """# Import necessary modules
import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname('__file__'), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Import the necessary modules
from src.jupyter.widgets import set_notebook_width
from src.jupyter.text_display import create_text_dashboard
from IPython.display import display

# Set notebook width for better display
set_notebook_width('100%')"""
cells.append(nbf.v4.new_code_cell(code1))

# Markdown cell for Generate Mock Analysis Data
markdown2 = """## Generate Mock Analysis Data

For demonstration purposes, we'll generate some mock cryptocurrency analysis data. Each indicator includes detailed educational content that will be displayed in accordions."""
cells.append(nbf.v4.new_markdown_cell(markdown2))

# Code cell for generating mock data
code2 = """def generate_mock_analysis():
    \"\"\"Generate mock analysis results for testing the dashboard\"\"\"
    
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
        "indicator_values": {
            "moving_average": {
                "latest_value": 45800.50,
                "previous_value": 44950.25,
                "change_percent": 1.89
            },
            "macd": {
                "latest_value": 235.75,
                "signal_line": 175.25,
                "histogram": 60.50
            },
            "parabolic_sar": {
                "latest_value": 44950.25,
                "trend_direction": "up"
            },
            "bollinger_bands": {
                "latest_value": 46750.25,
                "upper_band": 48500.75,
                "middle_band": 46200.50,
                "lower_band": 43900.25
            },
            "ichimoku_cloud": {
                "latest_value": 46100.00,
                "tenkan_sen": 46250.50,
                "kijun_sen": 45500.75
            },
            "rsi": {
                "latest_value": 58.5,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            },
            "stochastic": {
                "latest_value": 82.3,
                "signal_line": 78.5
            },
            "cci": {
                "latest_value": 120.5,
                "positive_threshold": 100,
                "negative_threshold": -100
            },
            "williams_r": {
                "latest_value": -85.2,
                "oversold_threshold": -80,
                "overbought_threshold": -20
            },
            "awesome_oscillator": {
                "latest_value": 175.5,
                "previous_value": 150.2
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

# Generate the mock analysis data
analysis_results = generate_mock_analysis()"""
cells.append(nbf.v4.new_code_cell(code2))

# Markdown cell for displaying the dashboard
markdown3 = """## Display the Text-Based Dashboard

Now we'll display the HTML-based text dashboard using our mock data. Try clicking on the indicator names to expand the educational accordions."""
cells.append(nbf.v4.new_markdown_cell(markdown3))

# Code cell for displaying the dashboard
code3 = """# Create and display the HTML dashboard
dashboard = create_text_dashboard(analysis_results)
display(dashboard)"""
cells.append(nbf.v4.new_code_cell(code3))

# Markdown cell for conflicting signals example
markdown4 = """## Creating a Modified Example with Conflicting Signals

Let's create another example with different data to show how conflicting signals are presented. This example also demonstrates the accordion functionality with different indicator values and signals."""
cells.append(nbf.v4.new_markdown_cell(markdown4))

# Code cell for conflicting signals
code4 = """def generate_conflicting_analysis():
    \"\"\"Generate analysis results with conflicting signals\"\"\"
    
    return {
        "metadata": {
            "symbol": "ETH",
            "vs_currency": "usd",
            "timeframe": "1d",
            "last_updated": datetime.now().isoformat()
        },
        "price_data": {
            "current_price": 2450.75,
            "price_change_24h": -75.25,
            "price_change_percentage_24h": -3.15,
            "market_cap": 289560000000,
            "volume_24h": 23580000000,
            "high_24h": 2510.00,
            "low_24h": 2412.50,
            "price_change_percentage_7d": -2.8,
            "price_change_percentage_30d": 8.5,
            "price_change_percentage_1y": 35.7
        },
        "market_trends": {
            "short_term": "bearish",
            "medium_term": "bullish",
            "long_term": "bullish"
        },
        "signals": {
            "trend_signals": {
                "moving_average": "neutral",
                "macd": "bearish",
                "parabolic_sar": "bearish",
                "bollinger_bands": "neutral",
                "ichimoku_cloud": "bullish"
            },
            "oscillator_signals": {
                "rsi": "bullish",
                "stochastic": "bearish",
                "cci": "bullish",
                "williams_r": "bearish",
                "awesome_oscillator": "neutral"
            }
        },
        "indicator_values": {
            "moving_average": {
                "latest_value": 2455.50,
                "previous_value": 2450.25,
                "change_percent": 0.21
            },
            "macd": {
                "latest_value": -45.25,
                "signal_line": -25.50,
                "histogram": -19.75
            },
            "parabolic_sar": {
                "latest_value": 2510.75,
                "trend_direction": "down"
            },
            "bollinger_bands": {
                "latest_value": 2450.75,
                "upper_band": 2580.25,
                "middle_band": 2452.50,
                "lower_band": 2324.75
            },
            "ichimoku_cloud": {
                "latest_value": 2400.50,
                "tenkan_sen": 2420.75,
                "kijun_sen": 2380.50
            },
            "rsi": {
                "latest_value": 38.2,
                "oversold_threshold": 30,
                "overbought_threshold": 70
            },
            "stochastic": {
                "latest_value": 75.6,
                "signal_line": 80.2
            },
            "cci": {
                "latest_value": 112.8,
                "positive_threshold": 100,
                "negative_threshold": -100
            },
            "williams_r": {
                "latest_value": -65.4,
                "oversold_threshold": -80,
                "overbought_threshold": -20
            },
            "awesome_oscillator": {
                "latest_value": 5.25,
                "previous_value": 4.90
            }
        },
        "patterns": {
            "shooting_star": {
                "signal": "bearish",
                "strength": 0.85
            },
            "morning_star": {
                "signal": "bullish",
                "strength": 0.7
            },
            "bearish_harami": {
                "signal": "bearish",
                "strength": 0.6
            },
            "doji": {
                "signal": "neutral",
                "strength": 0.5
            }
        },
        "recommendation": {
            "action": "hold",
            "confidence": 0.55,
            "rationale": "Mixed signals across different timeframes and indicators. Short-term bearish indicators suggest potential near-term downside, but medium and long-term trends remain bullish. This classic conflict between timeframes suggests a consolidation phase. Consider holding current positions or implementing a dollar-cost averaging strategy."
        }
    }

# Generate the conflicting analysis data
conflicting_results = generate_conflicting_analysis()

# Create and display the second dashboard
dashboard2 = create_text_dashboard(conflicting_results)
display(dashboard2)"""
cells.append(nbf.v4.new_code_cell(code4))

# Markdown cell for educational content explanation
markdown_edu = """## Understanding the Educational Accordions

The dashboard now includes educational accordions for each technical indicator:

1. **What it is**: Brief explanation of the indicator and its purpose
2. **How it's calculated**: Formula and calculation method
3. **Current Value**: The indicator's latest value (when available)
4. **What it means now**: Interpretation of the current signal (bullish/bearish/neutral)

These accordions make the dashboard more informative and educational, especially for users who are learning about technical analysis."""
cells.append(nbf.v4.new_markdown_cell(markdown_edu))

# Markdown cell for conclusion
markdown5 = """## Conclusion

This HTML-based text dashboard provides a clear, reliable way to present cryptocurrency analysis data. By focusing on textual representation with appropriate color coding and styling, we avoid the complexity and potential rendering issues of interactive Plotly charts while still conveying all the essential information.

The methodology section helps users understand how to interpret the data and develop their own analytical skills by considering multiple factors across different timeframes. The new accordion feature adds educational value to the dashboard, making it both informative and instructional."""
cells.append(nbf.v4.new_markdown_cell(markdown5))

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
        'version': '3.13.0'
    }
}

# Error handling
try:
    # Make sure the directory exists
    os.makedirs("notebooks/examples", exist_ok=True)

    # Write the notebook to a file
    output_file = "notebooks/examples/text_dashboard_demo.ipynb"
    nbf.write(nb, output_file)

    print(f"Text dashboard demo notebook created successfully at {output_file}")
except Exception as e:
    print(f"Error creating notebook: {e}")
    sys.exit(1) 