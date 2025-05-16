import os
import webbrowser
import re
from .common import OutputFormat  # Assuming OutputFormat is in common.py
from src.cli.education import get_indicator_explanation

# Helper functions for displaying messages (if they are simple enough to be here,
# otherwise they might also be in a shared utility module)
def display_info(message: str):
    print(f"[INFO] {message}")

def display_success(message: str):
    print(f"[SUCCESS] {message}")

def display_warning(message: str):
    print(f"[WARNING] {message}")

def display_error(message: str):
    print(f"[ERROR] {message}")

def _save_visualizations(visualizations: dict, symbol: str, timeframe: str, output_dir: str) -> dict:
    """Saves plots to files and returns a dictionary of paths."""
    saved_paths = {}
    if not visualizations:
        return saved_paths

    # Ensure the base directory for images exists
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    for key, fig in visualizations.items():
        if fig:
            # Sanitize filename
            safe_key = re.sub(r'[^a-zA-Z0-9_\-]', '_', key)
            filename = f"{symbol}_{timeframe}_{safe_key}.png"
            filepath = os.path.join(images_dir, filename)
            try:
                fig.savefig(filepath, bbox_inches='tight')
                saved_paths[key] = filepath
                # print(f"Saved {key} plot to {filepath}") # Optional: for debugging
            except Exception as e:
                print(f"Error saving plot {key}: {e}")
    return saved_paths

def generate_html_report(analysis_results: dict, symbol: str, timeframe: str, output_dir: str, explain: bool = False, save_charts: bool = True) -> str:
    """
    Generates an HTML report from the analysis results.

    Args:
        analysis_results: Dictionary containing the analysis data.
        symbol: The trading symbol (e.g., 'BTC-USD').
        timeframe: The timeframe of the analysis (e.g., '1d').
        output_dir: The directory to save supporting files like images.
        explain: Whether to include explanations for indicators.
        save_charts: Whether to save charts as images and embed them.

    Returns:
        str: The HTML content of the report.
    """
    visualizations = analysis_results.get('visualizations', {})
    image_paths = {}
    if save_charts and visualizations:
        image_paths = _save_visualizations(visualizations, symbol, timeframe, output_dir)

    # Start HTML content
    html_content = f"""<html>
<head>
    <title>Market Analysis for {symbol} ({timeframe})</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }}
        .container {{ background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f0f0f0; }}
        .indicator {{ margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
        .indicator h3 {{ margin-top: 0; }}
        .recommendation-HOLD {{ color: orange; }}
        .recommendation-BUY {{ color: green; }}
        .recommendation-SELL {{ color: red; }}
        .explanation {{ font-style: italic; color: #555; margin-top: 5px; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }}
        pre {{ background-color: #eee; padding: 10px; border-radius: 4px; white-space: pre-wrap; word-wrap: break-word; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Market Analysis Report: {symbol} ({timeframe})</h1>
"""

    # General Market Overview
    if 'general_overview' in analysis_results and analysis_results['general_overview']:
        html_content += "<h2>General Market Overview</h2>"
        html_content += f"<p>{analysis_results['general_overview']}</p>"

    # Price Action
    if 'price_action' in analysis_results and analysis_results['price_action']:
        html_content += "<h2>Price Action Analysis</h2>"
        pa_data = analysis_results['price_action']
        html_content += "<table>"
        for key, value in pa_data.items():
            html_content += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        html_content += "</table>"

    # Technical Indicators
    if 'indicators' in analysis_results:
        html_content += "<h2>Technical Indicators</h2>"
        for indicator_name, data in analysis_results['indicators'].items():
            if isinstance(data, dict):
                interpretation = data.get('interpretation', 'N/A')
                value = data.get('value', 'N/A')
                values = data.get('values', {})
                recommendation = data.get('recommendation', '')

                html_content += f"<div class='indicator'><h3>{indicator_name.upper()}</h3>"
                html_content += f"<p>Interpretation: {interpretation}</p>"
                
                if isinstance(value, (int, float)):
                     html_content += f"<p>Value: {value:.4f}</p>"
                elif value != 'N/A':
                     html_content += f"<p>Value: {value}</p>"
                
                if values:
                    html_content += "<ul>"
                    for k, v in values.items():
                        if isinstance(v, float):
                            html_content += f"<li>{k.replace('_', ' ').title()}: {v:.4f}</li>"
                        else:
                            html_content += f"<li>{k.replace('_', ' ').title()}: {v}</li>"
                    html_content += "</ul>"

                if recommendation:
                    html_content += f"<p>Recommendation: <span class='recommendation-{recommendation.upper()}'>{recommendation}</span></p>"
                
                if explain:
                    explanation = get_indicator_explanation(indicator_name)
                    if explanation:
                        html_content += f"<p class='explanation'><em>{explanation}</em></p>"
                html_content += "</div>"
            else:
                html_content += f"<div class='indicator'><h3>{indicator_name.upper()}</h3><p>{data}</p></div>"

    # Candlestick Patterns
    if 'candlestick_patterns' in analysis_results and analysis_results['candlestick_patterns']:
        html_content += "<h2>Candlestick Patterns Detected</h2>"
        html_content += "<ul>"
        for pattern_info in analysis_results['candlestick_patterns']:
            html_content += f"<li>{pattern_info['name']} (Date: {pattern_info['date']})</li>"
        html_content += "</ul>"

    # Chart Patterns
    if 'chart_patterns' in analysis_results and analysis_results['chart_patterns']:
        html_content += "<h2>Chart Patterns</h2>"
        html_content += "<ul>"
        for pattern in analysis_results['chart_patterns']:
            html_content += f"<li>{pattern['name']}: {pattern['description']}</li>"
        html_content += "</ul>"
    
    # Volume Analysis
    if 'volume_analysis' in analysis_results and analysis_results['volume_analysis']:
        html_content += "<h2>Volume Analysis</h2>"
        html_content += f"<p>{analysis_results['volume_analysis']}</p>"

    # Support and Resistance
    if 'support_resistance' in analysis_results and analysis_results['support_resistance']:
        html_content += "<h2>Support and Resistance Levels</h2>"
        sr_data = analysis_results['support_resistance']
        html_content += "<table><tr><th>Level Type</th><th>Price</th></tr>"
        for level in sr_data.get('support', []):
            html_content += f"<tr><td>Support</td><td>{level:.2f}</td></tr>"
        for level in sr_data.get('resistance', []):
            html_content += f"<tr><td>Resistance</td><td>{level:.2f}</td></tr>"
        html_content += "</table>"

    # Fibonacci Levels
    if 'fibonacci_levels' in analysis_results and analysis_results['fibonacci_levels']:
        html_content += "<h2>Fibonacci Retracement Levels</h2>"
        fib_data = analysis_results['fibonacci_levels']
        html_content += "<table><tr><th>Level</th><th>Price</th></tr>"
        for level, price in fib_data.items():
            html_content += f"<tr><td>{level}</td><td>{price:.2f}</td></tr>"
        html_content += "</table>"
    
    # Market Sentiment (Example - this would need actual data)
    if 'market_sentiment' in analysis_results and analysis_results['market_sentiment']:
        html_content += "<h2>Market Sentiment</h2>"
        html_content += f"<p>{analysis_results['market_sentiment']}</p>"
    
    # News and Events (Example - this would need actual data)
    if 'news_events' in analysis_results and analysis_results['news_events']:
        html_content += "<h2>Relevant News and Events</h2>"
        html_content += "<ul>"
        for news_item in analysis_results['news_events']:
            html_content += f"<li>{news_item}</li>"
        html_content += "</ul>"

    # Visualizations
    if image_paths:
        html_content += "<h2>Charts</h2>"
        for chart_name, chart_path in image_paths.items():
            # Make path relative for HTML if saved within a subfolder of the report
            relative_chart_path = os.path.join('images', os.path.basename(chart_path))
            html_content += f"<h3>{chart_name.replace('_', ' ').title()}</h3>"
            html_content += f"<img src='{relative_chart_path}' alt='{chart_name.replace('_', ' ').title()}'><br><br>"
    
    html_content += "</div></body></html>"
    return html_content 