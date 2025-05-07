"""
Analyzer Command Handler

Handles the 'analyzer' command for generating market analysis using the MarketAnalyzer class.
"""

import json
import logging
import time
import os
import datetime
import re
from enum import Enum
from typing import Optional, Dict, Any, Union, List
from pathlib import Path

import typer
import numpy as np
import pandas as pd
from rich import print
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from colorama import Fore, Style, init

init(autoreset=True)  # Initialize colorama

from src.jupyter.market_analyzer import MarketAnalyzer
from src.services import indicators
from src.cli.education import get_indicator_explanation, category_header, get_period_return_explanation, get_volatility_explanation
from src.cli.display import (
    format_price,
    display_info,
    display_success,
    display_warning,
    display_error
)

# Define analyzer error
class AnalyzerError(Exception):
    """Exception raised for analyzer errors."""
    pass

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
analyzer_app = typer.Typer()
console = Console()

# Create a custom JSON encoder to handle NumPy types and other special data types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types and other special data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # Handle any non-serializable objects by converting to string
        try:
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            return str(obj)

class TimeframeOption(str, Enum):
    """Trading timeframe options."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class OutputFormat(str, Enum):
    """
    Output format options.
    
    TEXT: Display formatted text in the terminal
    TXT: Display formatted text in the terminal and save to a text file
    JSON: Display JSON in the terminal
    JSF: Display JSON in the terminal and save to a JSON file
    HTML: Generate HTML and open in browser (saves to file)
    """
    TEXT = "text"
    TXT = "txt"
    JSON = "json"
    JSF = "jsf"
    HTML = "html"


@analyzer_app.callback()
def callback():
    """Market analyzer powered by the MarketAnalyzer class."""
    print("📊 Market Analyzer: Comprehensive market analysis using multiple timeframes")


def preprocess_for_json(obj):
    """
    Recursively preprocess data before JSON serialization to ensure it's serializable.
    
    Args:
        obj: Object to preprocess
        
    Returns:
        Preprocessed object
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: preprocess_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [preprocess_for_json(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return preprocess_for_json(obj.to_dict())
    else:
        # Try to serialize, if not possible, convert to string
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)


@analyzer_app.command()
def analyze(
    symbol: str = typer.Option("BTC-USDT", "--symbol", "-s", help="Symbol to analyze (e.g., BTC-USDT, ETH-USDT)"),
    timeframe: str = typer.Option("short", "--timeframe", "-t", 
                                help="Trading timeframe (short, medium, long)"),
    output: str = typer.Option("text", "--output", "-o", 
                              help="Output format (text/txt/json/jsf/html) - txt/jsf/html options save to file [default: text in the terminal]"),
    save_charts: bool = typer.Option(False, "--save-charts", "-c", 
                                    help="Save visualization charts to files"),
    explain: bool = typer.Option(False, "--explain", "-e", 
                                help="Include educational explanations for indicators"),
    debug: bool = typer.Option(False, "--debug", "-d", 
                              help="Enable debug logging"),
):
    """
    Analyze a market symbol using the specified timeframe.
    
    This command uses the MarketAnalyzer class to perform a comprehensive market analysis
    including technical indicators, visualizations, and a market summary.
    """
    try:
        # Set the log level to debug if debug is enabled
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Run analysis
        print(f"Analyzing {symbol} with {timeframe} timeframe...")
        analyzer = MarketAnalyzer(symbol=symbol, timeframe=timeframe)
        
        # Determine if this is a file-saving output type
        is_file_output = output.lower() in ['txt', 'jsf', 'html']
        output_format = output.lower()
        output_filename = None
        
        # If this is a file-saving type, prepare the file path
        if is_file_output:
            output_filename = _generate_output_filename(symbol, timeframe, output_format)
            logging.info(f"Output will be saved to: {output_filename}")
        
        # For JSON/JSF output format
        if output_format in ["json", "jsf"]:
            # Get the summary and cases
            summary = analyzer.get_summary()
            cases = analyzer.present_cases()
            
            # Preprocess both to ensure they're serializable
            summary_processed = preprocess_for_json(summary)
            cases_processed = preprocess_for_json(cases)
            
            # Add cases to the summary
            summary_processed['market_cases'] = cases_processed
            
            # Generate JSON output
            json_output = json.dumps(summary_processed, indent=2)
            
            # Always display in terminal
            print(json_output)
            
            # Save to file if JSF format
            if output_format == "jsf" and output_filename:
                with open(output_filename, 'w') as f:
                    f.write(json_output)
                print(f"[green]✓ Analysis saved to {output_filename}[/green]")
        
        # For TXT output format - capture terminal output
        elif output_format == "txt":
            # Use io.StringIO to capture the output
            import io
            import sys
            original_stdout = sys.stdout
            captured_output = io.StringIO()
            sys.stdout = captured_output
            
            try:
                # Display market analysis normally
                display_market_analysis(analyzer, "text", not save_charts, explain)
            finally:
                # Restore stdout
                sys.stdout = original_stdout
            
            # Get the captured output
            output_text = captured_output.getvalue()
            
            # Print to console (with colors)
            print(output_text)
            
            # Strip ANSI codes before saving to file
            clean_text = _strip_ansi_codes(output_text)
            
            # Debug: Check if clean_text still contains ANSI codes 
            if re.search(r'\[\d+m|\[\d+;\d+m', clean_text):
                logging.warning("Warning: ANSI codes may still be present after stripping")
            
            # Extra safety: Apply more aggressive stripping if needed
            clean_text = re.sub(r'\[[^]]*m', '', clean_text)
            
            # The nuclear option: remove all square brackets and their contents if they look like control codes
            clean_text = re.sub(r'\[(?:\d+[;m]|[a-zA-Z])[^]]*\]', '', clean_text)
            
            # Final safety: Remove anything that looks remotely like a control sequence
            clean_text = "".join(c for c in clean_text if ord(c) >= 32 or c in "\n\r\t")
            
            # Save to file
            with open(output_filename, 'w') as f:
                f.write(clean_text)
            print(f"[green]✓ Analysis saved to {output_filename}[/green]")
        
        # For HTML format, the _display_html_output function already saves to a file
        elif output_format == "html":
            # HTML output is handled by display_market_analysis, but we'll use our directory structure
            display_market_analysis(analyzer, "html", not save_charts, explain)
            # Note: We'll need to update _display_html_output in another task to use our directory structure
        
        # For non-file outputs (TEXT/JSON), just display normally
        else:
            display_market_analysis(analyzer, output_format, not save_charts, explain)
        
        # Print success message
        print(f"[green]✓ Analysis for {symbol} completed successfully[/green]")
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        print(f"[red]✗ Error: Failed to analyze {symbol}: {e}[/red]")


def _display_text_output(summary: dict, data=None, explain: bool = False, output_format: str = "text"):
    """
    Display market analysis in text format.
    
    Args:
        summary: Analysis summary dict
        data: Full analysis data (for advanced display)
        explain: Whether to include educational explanations
        output_format: Output format (text/txt)
    """
    # Extract key information
    symbol = summary.get('symbol', '')
    timeframe = summary.get('timeframe', '')
    
    # If data is an analyzer object, get necessary data directly
    if hasattr(data, 'symbol') and hasattr(data, 'timeframe'):
        # Use analyzer's symbol and timeframe if available
        symbol = data.symbol
        timeframe = data.timeframe
    
    # Call print_market_analysis to generate output
    print_market_analysis(summary, symbol, timeframe, explain=explain)
    
    # For TXT output, the file saving is handled in the analyze function through IO redirection


def _display_json_output(summary: dict, analysis_results: dict, output_format: str = "json"):
    """
    Display analysis results in JSON format.
    
    Args:
        summary: Analysis summary
        analysis_results: Full analysis results
        output_format: Output format (json/jsf)
    """
    # Clone the summary and results to avoid modifying the original
    display_summary = summary.copy() if summary else {}
    display_results = analysis_results.copy() if analysis_results else {}
    
    # Process any large arrays or data structures
    # Helper function to recursively process nested dictionaries and arrays
    def process_large_arrays(obj):
        if isinstance(obj, dict):
            for key, value in list(obj.items()):  # Convert to list to avoid modification during iteration
                if isinstance(value, (list, np.ndarray)) and len(value) > 5:
                    obj[key] = f"[Array with {len(value)} elements]"
                elif isinstance(value, (dict, list)):
                    process_large_arrays(value)
                # Handle NaN, Infinity values which are not JSON serializable
                elif isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    obj[key] = None
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (list, np.ndarray)) and len(item) > 5:
                    obj[i] = f"[Array with {len(item)} elements]"
                elif isinstance(item, (dict, list)):
                    process_large_arrays(item)
                # Handle NaN, Infinity values
                elif isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
                    obj[i] = None
    
    # Process both summary and results
    process_large_arrays(display_summary)
    process_large_arrays(display_results)
    
    # Extract key information for the output
    output = {
        "symbol": display_summary.get('symbol', ''),
        "timeframe": display_summary.get('timeframe', ''),
        "current_price": display_summary.get('current_price', 0),
        "period_return": display_summary.get('period_return', 0),
        "volatility": display_summary.get('volatility', 0),
        "trend": display_summary.get('trend', {}),  # Include full trend data with advanced recommendations
        "indicators": display_summary.get('indicators', {}),
        "indicator_data": display_summary.get('indicator_data', {})
    }
    
    # Add detailed analysis if available
    if display_results:
        output["detailed_analysis"] = display_results
    
    # Add market_cases if available
    if 'market_cases' in display_summary:
        output["market_cases"] = display_summary.get('market_cases', {})
    
    try:
        # Convert to JSON and print
        json_output = json.dumps(output, cls=NumpyEncoder, indent=2)
        print(json_output)
        
        # For JSF output, the file saving is handled in the analyze function
    except Exception as e:
        # If JSON serialization fails, try to provide a more helpful error
        logger.error(f"JSON serialization error: {e}")
        print(f"Error converting results to JSON: {e}")
        # Try a simpler output as fallback
        try:
            simple_output = {
                "symbol": str(display_summary.get('symbol', '')),
                "price": float(display_summary.get('current_price', 0)),
                "error": f"Full JSON serialization failed: {e}"
            }
            print(json.dumps(simple_output))
        except:
            print('{"error": "JSON serialization completely failed"}')


def _display_html_output(summary: dict, analysis_results: dict, visualizations: dict = None, explain: bool = False):
    """
    Display analysis results as HTML.
    
    Args:
        summary: Analysis summary
        analysis_results: Full analysis results
        visualizations: Plotly visualizations
        explain: Whether to include educational explanations
    """
    symbol = summary.get('symbol', '')
    timeframe = summary.get('timeframe', '')
    
    # Direct hardcoded formatting based on the timeframe string
    formatted_timeframe = timeframe.upper()
    
    # Force interval display for known timeframes
    if timeframe.lower() == "short":
        formatted_timeframe = "SHORT - 15m"
    elif timeframe.lower() == "medium":
        formatted_timeframe = "MEDIUM - 1h"
    elif timeframe.lower() == "long":
        formatted_timeframe = "LONG - 1d"
    
    # Get trend data (can be a string or dictionary)
    trend_data = summary.get('trend', {})
    if isinstance(trend_data, dict):
        trend_direction = trend_data.get('direction', 'Unknown')
        trend_strength = trend_data.get('strength', 'Unknown')
        trend_confidence = trend_data.get('confidence', 'Unknown')
        trend_signals = trend_data.get('signals', {})
        trend_explanation = trend_data.get('explanation', '')
    else:
        # For backward compatibility with older format
        trend_direction = str(trend_data)
        trend_strength = "Unknown"
        trend_confidence = "Unknown"
        trend_signals = {}
        trend_explanation = ""
    
    # Create a basic HTML structure
    html = f"""
    <html>
    <head>
        <title>Market Analysis: {symbol} ({formatted_timeframe})</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .indicator {{ margin-bottom: 15px; padding: 10px; border-left: 3px solid #ddd; }}
            .indicator h3 {{ margin-top: 0; }}
            .charts {{ margin-top: 20px; }}
            .trend {{ background: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .signals {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .signal-card {{ flex: 1; min-width: 200px; background: #f8f9fa; padding: 10px; border-radius: 5px; }}
            .indicator-section {{ margin-bottom: 30px; }}
            .indicator-details {{ margin-left: 20px; font-family: monospace; }}
            .explanation {{ font-style: italic; font-size: 0.9em; margin-top: 5px; margin-bottom: 15px; color: #555; }}
            .indicator-value {{ font-family: monospace; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Market Analysis: {symbol} ({formatted_timeframe})</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Current Price:</strong> {format_price(summary.get('current_price', 0))}</p>
                <p><strong>Period Return:</strong> {summary.get('period_return', 0):.2f}%</p>
    """
    
    # Add period return explanation if explain mode is enabled
    if explain:
        period_return = summary.get('period_return', 0)
        period_return_explanation = get_period_return_explanation(period_return)
        html += f"""
                <p class="explanation">{period_return_explanation}</p>
        """
    
    html += f"""
                <p><strong>Volatility:</strong> {summary.get('volatility', 0):.2f}%</p>
    """
    
    # Add volatility explanation if explain mode is enabled
    if explain:
        volatility = summary.get('volatility', 0)
        volatility_explanation = get_volatility_explanation(volatility)
        html += f"""
                <p class="explanation">{volatility_explanation}</p>
        """
    
    html += """
            </div>
            
            <div class="trend">
                <h2>Trend Analysis</h2>
    """
    
    html += f"""
                <p><strong>Direction:</strong> {trend_direction.upper()}</p>
                <p><strong>Strength:</strong> {trend_strength}</p>
                <p><strong>Confidence:</strong> {trend_confidence}</p>
    """
    
    # Add signals if available
    if trend_signals:
        html += """
                <h3>Signals</h3>
                <div class="signals">
        """
        
        for period, signal in trend_signals.items():
            if period != 'action':
                html += f"""
                    <div class="signal-card">
                        <h4>{period.replace('_', ' ').title()}</h4>
                        <p>{signal}</p>
                    </div>
                """
        
        # Add action recommendation if available
        action = trend_signals.get('action', 'Hold')
        html += f"""
                    <div class="signal-card" style="background: #e8f4f8; font-weight: bold;">
                        <h4>Recommended Action</h4>
                        <p>{action.upper()}</p>
                    </div>
                </div>
        """
    
    # Add trend explanation if available
    if trend_explanation:
        html += f"""
                <div style="margin-top: 15px; font-style: italic;">
                    <p><strong>Analysis:</strong> {trend_explanation}</p>
                </div>
        """
    
    # Check if advanced recommendation is available
    advanced_recommendation = trend_data.get('advanced_recommendation', None) if isinstance(trend_data, dict) else None
    
    if advanced_recommendation:
        market_condition = advanced_recommendation.get('market_condition', {})
        condition = market_condition.get('condition', 'unknown')
        sub_condition = market_condition.get('sub_condition', 'unknown')
        confidence = advanced_recommendation.get('confidence', 'low')
        strategy = advanced_recommendation.get('strategy', 'hold_cash')
        action = advanced_recommendation.get('action', 'hold')
        
        html += """
                <h2>Advanced Trading Recommendation</h2>
                <div class="advanced-recommendation">
                    <div class="recommendation-header">
        """
        
        # Determine color based on confidence
        confidence_color = "#FFD700" if confidence == "high" else "#A9A9A9" if confidence == "low" else "#1E90FF"
        
        html += f"""
                        <div class="recommendation-card" style="background: #f0f8ff; border-left: 5px solid {confidence_color};">
                            <h3>Market Condition: {condition.capitalize()} ({sub_condition.replace('_', ' ').capitalize()})</h3>
                            <p><strong>Strategy:</strong> {strategy.replace('_', ' ').capitalize()}</p>
                            <p><strong>Action:</strong> <span style="font-weight: bold; color: {'#006400' if action.lower() == 'buy' else '#8B0000' if action.lower() == 'sell' else '#000080'};">{action.upper()}</span></p>
                            <p><strong>Confidence:</strong> {confidence.capitalize()}</p>
                        </div>
        """
        
        # Entry and exit points
        entry_points = advanced_recommendation.get('entry_points', [])
        exit_points = advanced_recommendation.get('exit_points', {})
        take_profit = exit_points.get('take_profit', [])
        stop_loss = exit_points.get('stop_loss')
        
        if entry_points or take_profit or stop_loss is not None:
            html += """
                    <div class="trading-levels">
                        <div style="display: flex; flex-wrap: wrap;">
            """
            
            if entry_points:
                html += """
                            <div class="level-card" style="flex: 1; min-width: 200px; margin: 5px; background: #f0fff0; border-radius: 5px; padding: 10px;">
                                <h4>Entry Points</h4>
                                <ul>
                """
                
                for entry in entry_points:
                    price = entry.get('price')
                    condition = entry.get('condition', '').replace('_', ' ').capitalize()
                    if price is not None:
                        html += f"""<li><strong>{condition}:</strong> {price:.2f}</li>"""
                    else:
                        html += f"""<li><strong>{condition}</strong></li>"""
                
                html += """
                                </ul>
                            </div>
                """
            
            if take_profit or stop_loss is not None:
                html += """
                            <div class="level-card" style="flex: 1; min-width: 200px; margin: 5px; background: #fff0f0; border-radius: 5px; padding: 10px;">
                                <h4>Exit Points</h4>
                                <ul>
                """
                
                if stop_loss is not None:
                    html += f"""<li><strong>Stop Loss:</strong> {stop_loss:.2f}</li>"""
                
                if take_profit:
                    for i, target in enumerate(take_profit[:3], 1):
                        html += f"""<li><strong>Target {i}:</strong> {target:.2f}</li>"""
                
                html += """
                                </ul>
                            </div>
                """
            
            html += """
                        </div>
                    </div>
            """
        
        # Risk assessment
        risk_assessment = advanced_recommendation.get('risk_assessment', {})
        risk_reward = risk_assessment.get('risk_reward_ratio')
        
        if risk_reward is not None:
            html += """
                    <div class="risk-assessment" style="margin-top: 15px; background: #f8f8f8; border-radius: 5px; padding: 10px;">
                        <h4>Risk Assessment</h4>
                        <div style="display: flex; flex-wrap: wrap;">
            """
            
            html += f"""
                            <div style="flex: 1; min-width: 150px; margin: 5px; text-align: center;">
                                <p style="font-size: 0.9em;">Risk/Reward Ratio</p>
                                <p style="font-size: 1.2em; font-weight: bold;">{risk_reward:.2f}</p>
                            </div>
            """
            
            risk_pct = risk_assessment.get('risk_pct')
            if risk_pct is not None:
                html += f"""
                            <div style="flex: 1; min-width: 150px; margin: 5px; text-align: center;">
                                <p style="font-size: 0.9em;">Risk</p>
                                <p style="font-size: 1.2em; font-weight: bold;">{risk_pct:.2f}%</p>
                            </div>
                """
            
            position_size = risk_assessment.get('position_size')
            if position_size is not None:
                html += f"""
                            <div style="flex: 1; min-width: 150px; margin: 5px; text-align: center;">
                                <p style="font-size: 0.9em;">Suggested Position Size</p>
                                <p style="font-size: 1.2em; font-weight: bold;">{position_size:.2f}%</p>
                            </div>
                """
            
            html += """
                        </div>
                    </div>
            """
        
        # Supporting and contrary indicators
        supportive = advanced_recommendation.get('supportive_indicators', [])
        contrary = advanced_recommendation.get('contrary_indicators', [])
        
        if supportive or contrary:
            html += """
                    <div class="indicators-overview" style="margin-top: 15px;">
                        <div style="display: flex; flex-wrap: wrap;">
            """
            
            if supportive:
                html += """
                            <div style="flex: 1; min-width: 200px; margin: 5px; background: #e6ffe6; border-radius: 5px; padding: 10px;">
                                <h4>Supporting Indicators</h4>
                                <ul>
                """
                
                for indicator in supportive[:3]:
                    if len(indicator) >= 2:
                        ind_name, ind_value = indicator[0], indicator[1]
                        html += f"""<li><strong>{ind_name}:</strong> {ind_value}</li>"""
                
                html += """
                                </ul>
                            </div>
                """
            
            if contrary:
                html += """
                            <div style="flex: 1; min-width: 200px; margin: 5px; background: #ffe6e6; border-radius: 5px; padding: 10px;">
                                <h4>Contrary Indicators</h4>
                                <ul>
                """
                
                for indicator in contrary[:3]:
                    if len(indicator) >= 2:
                        ind_name, ind_value = indicator[0], indicator[1]
                        html += f"""<li><strong>{ind_name}:</strong> {ind_value}</li>"""
                
                html += """
                                </ul>
                            </div>
                """
            
            html += """
                        </div>
                    </div>
            """
        
        html += """
                    </div>
                </div>
        """
    
    html += """
            </div>
            
            <h2 style="margin-top: 30px;">Technical Indicators</h2>
    """
    
    # Group indicators by category for better organization
    indicator_categories = {
        "Trend": ["sma", "ema", "macd", "adx", "ichimoku"],
        "Momentum": ["rsi", "stochastic", "cci"],
        "Volatility": ["bollinger", "atr"],
        "Volume": ["volume", "obv"]
    }
    
    # Get indicators and detailed data
    indicators = summary.get('indicators', {})
    indicator_data = summary.get('indicator_data', {})
    
    # Add indicators by category
    for category, indicator_list in indicator_categories.items():
        category_indicators = {k: v for k, v in indicators.items() if k in indicator_list and k in indicators}
        
        if category_indicators:
            html += f"""
            <div class="indicator-section">
                <h3>{category} Indicators</h3>
            """
            
            for indicator, interpretation in category_indicators.items():
                detailed_data = indicator_data.get(indicator, {})
                
                html += f"""
                <div class="indicator">
                    <h3>{indicator.upper()}</h3>
                    <p>{interpretation}</p>
                """
                
                # Add detailed data based on indicator type
                if indicator == "rsi" and isinstance(detailed_data, dict):
                    value = detailed_data.get('value')
                    if value is not None:
                        html += f"""
                        <div class="indicator-details">
                            <p>Value: {value:.2f}</p>
                        </div>
                        """
                
                elif indicator == "macd" and isinstance(detailed_data, dict):
                    values = detailed_data.get('values', {})
                    line = values.get('line')
                    signal = values.get('signal')
                    histogram = values.get('histogram')
                    
                    if all(v is not None for v in [line, signal, histogram]):
                        html += f"""
                        <div class="indicator-details">
                            <p>Line: {line:.4f}</p>
                            <p>Signal: {signal:.4f}</p>
                            <p>Histogram: {histogram:.4f}</p>
                        </div>
                        """
                
                elif indicator == "bollinger" and isinstance(detailed_data, dict):
                    values = detailed_data.get('values', {})
                    upper = values.get('upper')
                    middle = values.get('middle')
                    lower = values.get('lower')
                    close = values.get('close')
                    percent = values.get('percent')
                    
                    if all(v is not None for v in [upper, middle, lower, close]):
                        html += f"""
                        <div class="indicator-details">
                            <p>Upper Band: {upper:.2f}</p>
                            <p>Middle Band: {middle:.2f}</p>
                            <p>Lower Band: {lower:.2f}</p>
                            <p>Price: {close:.2f}</p>
                        """
                        if percent is not None:
                            html += f"""<p>Position: {percent:.2f}% from middle</p>"""
                        html += """</div>"""
                
                elif indicator == "stochastic" and isinstance(detailed_data, dict):
                    values = detailed_data.get('values', {})
                    k_value = values.get('k')
                    d_value = values.get('d')
                    
                    if k_value is not None and d_value is not None:
                        html += f"""
                        <div class="indicator-details">
                            <p>%K: {k_value:.2f}</p>
                            <p>%D: {d_value:.2f}</p>
                        </div>
                        """
                
                elif indicator == "adx" and isinstance(detailed_data, dict):
                    value = detailed_data.get('value')
                    if value is not None:
                        html += f"""
                        <div class="indicator-details">
                            <p>Value: {value:.2f}</p>
                        </div>
                        """
                
                elif indicator == "cci" and isinstance(detailed_data, dict):
                    value = detailed_data.get('value')
                    if value is not None:
                        html += f"""
                        <div class="indicator-details">
                            <p>Value: {value:.2f}</p>
                        </div>
                        """
                
                elif indicator == "atr" and isinstance(detailed_data, dict):
                    value = detailed_data.get('value')
                    if value is not None:
                        html += f"""
                        <div class="indicator-details">
                            <p>Value: {value:.2f}</p>
                        </div>
                        """
                
                elif indicator == "obv" and isinstance(detailed_data, dict):
                    value = detailed_data.get('value')
                    if value is not None and isinstance(value, (int, float)):
                        html += f"""
                        <div class="indicator-details">
                            <p>Value: {value:.0f}</p>
                        </div>
                        """
                        
                elif indicator == "ichimoku" and isinstance(detailed_data, dict):
                    values = detailed_data.get('values', {})
                    tenkan = values.get('tenkan_sen')
                    kijun = values.get('kijun_sen')
                    senkou_a = values.get('senkou_span_a')
                    senkou_b = values.get('senkou_span_b')
                    chikou = values.get('chikou_span')
                    
                    if all(v is not None for v in [tenkan, kijun, senkou_a, senkou_b]):
                        html += f"""
                        <div class="indicator-details">
                            <p>Tenkan-sen (Conversion): {tenkan:.2f}</p>
                            <p>Kijun-sen (Base): {kijun:.2f}</p>
                            <p>Senkou Span A (Leading A): {senkou_a:.2f}</p>
                            <p>Senkou Span B (Leading B): {senkou_b:.2f}</p>
                        """
                        if chikou is not None:
                            html += f"""<p>Chikou Span (Lagging): {chikou:.2f}</p>"""
                        html += """</div>"""
                
                html += """
                </div>
                """
            
            html += """
            </div>
            """
    
    # Add visualizations if available
    if visualizations:
        html += """
            <h2>Charts</h2>
            <div class="charts">
        """
        
        for chart_name, fig in visualizations.items():
            chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            html += f"""
                <h3>{chart_name.capitalize()} Chart</h3>
                {chart_html}
            """
        
        html += "</div>"
    
    html += """
        </div>
    </body>
    </html>
    """
    
    # Generate output filename using the helper function
    output_file = _generate_output_filename(symbol, timeframe, "html")
    
    # Create an HTML file
    with open(output_file, "w") as f:
        f.write(html)
    
    print(f"[green]✓ HTML analysis saved to {output_file}[/green]")
    
    # Open in browser
    import webbrowser
    webbrowser.open(output_file)


def _save_visualizations(visualizations: dict, symbol: str, timeframe: str):
    """
    Save visualizations to files.
    
    Args:
        visualizations: Dictionary of visualizations
        symbol: Symbol being analyzed
        timeframe: Trading timeframe
    """
    # Create charts directory
    charts_dir = _ensure_output_directory("charts")
    
    for chart_name, fig in visualizations.items():
        # Check if this is a plotly figure (has write_html method) or matplotlib figure
        if hasattr(fig, 'write_html'):
            # Plotly figure
            filename = os.path.join(charts_dir, f"{symbol}_{timeframe}_{chart_name}.html")
            fig.write_html(filename)
            print(f"[green]✓ Saved {chart_name} chart to {filename}[/green]")
        else:
            # Matplotlib figure
            filename = os.path.join(charts_dir, f"{symbol}_{timeframe}_{chart_name}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"[green]✓ Saved {chart_name} chart to {filename}[/green]")


def display_market_analysis(analyzer: MarketAnalyzer, output_format: str = None, show_charts: bool = True, explain: bool = False) -> None:
    """
    Display market analysis results based on the specified output format.
    
    Args:
        analyzer: Market analyzer instance with analysis results
        output_format: Output format (text, json, html)
        show_charts: Whether to display charts
        explain: Whether to include educational explanations
    """
    try:
        # Get the analysis summary
        summary = analyzer.get_summary()
        if summary is None:
            logging.error(f"Failed to get analysis summary for {analyzer.symbol}")
            display_error(f"Failed to get analysis summary for {analyzer.symbol}")
            return
        
        # Add a debug print to show timeframe values
        print(f"DEBUG: timeframe value from analyzer: '{analyzer.timeframe}', type: {type(analyzer.timeframe)}")
        
        # Add a debug print to verify print_market_analysis is being called
        print(f"DEBUG: About to call print_market_analysis with timeframe='{analyzer.timeframe}'")
        
        # Handle output format
        if output_format and output_format.lower() in ['text', 'txt']:
            _display_text_output(summary, analyzer, explain=explain, output_format=output_format.lower())
        elif output_format and output_format.lower() in ['json', 'jsf']:
            _display_json_output(summary, analyzer.get_full_analysis() if hasattr(analyzer, 'get_full_analysis') else {}, output_format=output_format.lower())
        elif output_format and output_format.lower() == 'html':
            # Get visualizations if charts are requested
            visualizations = None
            if show_charts:
                try:
                    visualizations = analyzer.generate_visualizations() if hasattr(analyzer, 'generate_visualizations') else None
                except Exception as e:
                    logging.warning(f"Failed to generate visualizations: {str(e)}")
                    visualizations = None
            
            _display_html_output(summary, analyzer.get_full_analysis() if hasattr(analyzer, 'get_full_analysis') else {}, visualizations, explain=explain)
        else:
            # Default to text output
            print_market_analysis(summary, analyzer.symbol, analyzer.timeframe, explain=explain)
            
            # Calculate detailed support/resistance levels
            cases = analyzer.present_cases() if hasattr(analyzer, 'present_cases') else {}
            if not cases or not isinstance(cases, dict) or 'cases' not in cases:
                logging.warning("Support/resistance data is missing or invalid")
                return
                
            print("\n=== SUPPORT AND RESISTANCE SCENARIOS ===")
            for case_type in ['bullish', 'bearish', 'neutral']:
                case = cases['cases'][case_type]
                print(f"\n{case_type.upper()} CASE (Confidence: {case['confidence']})")
                if case['supporting_indicators']:
                    print("Supporting Indicators:")
                    for indicator, interp in case['supporting_indicators']:
                        print(f"  - {indicator.upper()}: {interp}")
                else:
                    print("No supporting indicators")
            
            print("======================")
            
            # Remove the display_charts call since it's not defined
            # if show_charts:
            #     display_charts(analyzer)
    except Exception as e:
        logging.error(f"Error analyzing {analyzer.symbol}: {str(e)}")
        raise AnalyzerError(f"Failed to analyze {analyzer.symbol}: {str(e)}")


def print_market_analysis(summary, symbol, timeframe, explain: bool = False):
    """
    Print market analysis in text format.
    
    Args:
        summary: Analysis summary dict
        symbol: Market symbol
        timeframe: Trading timeframe
        explain: Whether to include educational explanations
    """
    # Direct hardcoded formatting based on the timeframe string
    # Map standard timeframes to their intervals
    formatted_timeframe = timeframe.upper()
    
    # Force interval display for known timeframes
    if timeframe.lower() == "short":
        formatted_timeframe = "SHORT - 15m"
    elif timeframe.lower() == "medium":
        formatted_timeframe = "MEDIUM - 1h"
    elif timeframe.lower() == "long":
        formatted_timeframe = "LONG - 1d"
    
    # Display header
    print("\n" + "=" * 70)
    print(f"MARKET ANALYSIS: {symbol} ({formatted_timeframe})")
    print("=" * 70)
    
    # Price information
    price = summary.get('current_price', float('nan'))
    period_return = summary.get('period_return', float('nan'))
    volatility = summary.get('volatility', float('nan'))
    
    print("\nPRICE INFORMATION:")
    print(f"Current Price: {format_price(price)}")
    print(f"24H change: {period_return:.2f}%")
    print(f"Volatility: {volatility:.2f}%")
    
    # Add educational content if explain mode is enabled
    if explain:
        print(f"  └─ {get_volatility_explanation(volatility)}")
        print(f"  └─ {get_period_return_explanation(period_return)}")
    
    # Trend information
    trend = summary.get('trend', {})
    if isinstance(trend, dict):
        direction = trend.get('direction', 'UNKNOWN')
        strength = trend.get('strength', 'Unknown')
        confidence = trend.get('confidence', 'Unknown')
        signals = trend.get('signals', {})
        trend_explanation = trend.get('explanation', '')
        advanced_recommendation = trend.get('advanced_recommendation', None)
        
        print(f"\nTREND: {direction.upper()}")
        print(f"Strength: {strength}")
        print(f"Confidence: {confidence}")
        
        if signals:
            print("\nSIGNALS:")
            print(f"  Short-term: {signals.get('short_term', 'Neutral')}")
            print(f"  Medium-term: {signals.get('medium_term', 'Neutral')}")
            print(f"  Long-term: {signals.get('long_term', 'Neutral')}")
            print(f"  Recommended Action: {signals.get('action', 'HOLD').upper()}")
            
        # Display advanced recommendation information if available
        if advanced_recommendation:
            print("\nADVANCED TRADING RECOMMENDATION:")
            market_condition = advanced_recommendation.get('market_condition', {})
            condition = market_condition.get('condition', 'unknown')
            sub_condition = market_condition.get('sub_condition', 'unknown')
            confidence = advanced_recommendation.get('confidence', 'low')
            strategy = advanced_recommendation.get('strategy', 'hold_cash')
            action = advanced_recommendation.get('action', 'hold')
            
            print(f"  Market Condition: {condition.capitalize()} ({sub_condition.replace('_', ' ').capitalize()})")
            print(f"  Strategy: {strategy.replace('_', ' ').capitalize()}")
            print(f"  Action: {action.upper()}")
            print(f"  Confidence: {confidence.capitalize()}")
            
            # Display entry and exit points
            entry_points = advanced_recommendation.get('entry_points', [])
            if entry_points:
                print("\n  Entry Points:")
                for entry in entry_points:
                    price = entry.get('price')
                    condition = entry.get('condition', '').replace('_', ' ').capitalize()
                    if price is not None:
                        print(f"    • {condition} @ {price:.2f}")
                    else:
                        print(f"    • {condition}")
            
            exit_points = advanced_recommendation.get('exit_points', {})
            take_profit = exit_points.get('take_profit', [])
            stop_loss = exit_points.get('stop_loss')
            
            if take_profit or stop_loss is not None:
                print("\n  Exit Points:")
                if stop_loss is not None:
                    print(f"    • Stop Loss @ {stop_loss:.2f}")
                if take_profit:
                    for i, target in enumerate(take_profit[:3], 1):  # Show first 3 targets
                        print(f"    • Target {i} @ {target:.2f}")
            
            # Display risk assessment
            risk_assessment = advanced_recommendation.get('risk_assessment', {})
            risk_reward = risk_assessment.get('risk_reward_ratio')
            
            if risk_reward is not None:
                print("\n  Risk Assessment:")
                print(f"    • Risk/Reward Ratio: {risk_reward:.2f}")
                
                risk_pct = risk_assessment.get('risk_pct')
                if risk_pct is not None:
                    print(f"    • Risk: {risk_pct:.2f}%")
                
                position_size = risk_assessment.get('position_size')
                if position_size is not None:
                    print(f"    • Suggested Position Size: {position_size:.2f}%")
            
            # Display supporting indicators
            supportive = advanced_recommendation.get('supportive_indicators', [])
            contrary = advanced_recommendation.get('contrary_indicators', [])
            
            if supportive:
                print("\n  Supporting Indicators:")
                for indicator in supportive[:3]:  # Show first 3 supporting indicators
                    if len(indicator) >= 2:
                        ind_name, ind_value = indicator[0], indicator[1]
                        print(f"    • {ind_name}: {ind_value}")
            
            if contrary:
                print("\n  Contrary Indicators:")
                for indicator in contrary[:3]:  # Show first 3 contrary indicators
                    if len(indicator) >= 2:
                        ind_name, ind_value = indicator[0], indicator[1]
                        print(f"    • {ind_name}: {ind_value}")
        
        # Display trend explanation if available
        if trend_explanation and explain:
            print(f"\nTrend Analysis: {trend_explanation}")
    else:
        print(f"\nTREND: {trend}")
    
    # Display technical indicators
    print("\nTECHNICAL INDICATORS:")
    
    # Group indicators by category
    indicators = summary.get('indicators', {})
    indicator_data = summary.get('indicator_data', {})
    
    # Category order for display
    categories = ["Trend", "Momentum", "Volatility", "Volume"]
    indicator_categories = {
        "Trend": ["macd", "sma", "ema", "adx", "ichimoku"],
        "Momentum": ["rsi", "stochastic", "cci"],
        "Volatility": ["bollinger", "atr"],
        "Volume": ["volume", "obv"]
    }
    
    # Display indicators by category
    for category in categories:
        indicator_list = indicator_categories.get(category, [])
        category_indicators = {}
        
        # First collect indicators for this category
        for indicator_name in indicator_list:
            if indicator_name in indicators:
                # Handle different formats of indicator data
                indicator_info = indicators[indicator_name]
                if isinstance(indicator_info, dict) and "interpretation" in indicator_info:
                    interpretation = indicator_info["interpretation"]
                else:
                    interpretation = str(indicator_info)
                
                category_indicators[indicator_name] = interpretation
        
        # If we have indicators in this category, display them
        if category_indicators:
            print(f"\n{category}:")
            
            # Display each indicator in this category
            for indicator_name, interpretation in category_indicators.items():
                # Get detailed data if available
                detailed_data = indicator_data.get(indicator_name, {})
                
                # Format display based on indicator type
                if indicator_name == "macd":
                    values = detailed_data.get('values', {})
                    line = values.get('line')
                    signal = values.get('signal')
                    histogram = values.get('histogram')
                    
                    print(f"  - MACD: {interpretation}")
                    if all(v is not None for v in [line, signal, histogram]):
                        print(f"    ├─ Line: {line:.4f}")
                        print(f"    ├─ Signal: {signal:.4f}")
                        print(f"    └─ Histogram: {histogram:.4f}")
                
                elif indicator_name == "rsi":
                    value = detailed_data.get('value')
                    print(f"  - RSI: {interpretation}")
                    if value is not None:
                        print(f"    └─ Value: {value:.2f}")
                
                elif indicator_name == "adx":
                    value = detailed_data.get('value')
                    print(f"  - ADX: {interpretation}")
                    if value is not None:
                        print(f"    └─ Value: {value:.2f}")
                
                elif indicator_name == "stochastic":
                    values = detailed_data.get('values', {})
                    k_value = values.get('k')
                    d_value = values.get('d')
                    
                    print(f"  - STOCHASTIC: {interpretation}")
                    if k_value is not None and d_value is not None:
                        print(f"    ├─ %K: {k_value:.2f}")
                        print(f"    └─ %D: {d_value:.2f}")
                
                elif indicator_name == "bollinger":
                    values = detailed_data.get('values', {})
                    upper = values.get('upper')
                    middle = values.get('middle')
                    lower = values.get('lower')
                    close = values.get('close')
                    percent = values.get('percent')
                    
                    print(f"  - BOLLINGER: {interpretation}")
                    if all(v is not None for v in [upper, middle, lower, close]):
                        print(f"    ├─ Upper Band: {upper:.2f}")
                        print(f"    ├─ Middle Band: {middle:.2f}")
                        print(f"    ├─ Lower Band: {lower:.2f}")
                        print(f"    ├─ Price: {close:.2f}")
                        if percent is not None:
                            print(f"    └─ Position: {percent:.2f}% from middle")
                
                elif indicator_name == "atr":
                    value = detailed_data.get('value')
                    print(f"  - ATR: {interpretation}")
                    if value is not None:
                        print(f"    └─ Value: {value:.2f}")
                
                elif indicator_name == "cci":
                    value = detailed_data.get('value')
                    print(f"  - CCI: {interpretation}")
                    if value is not None:
                        print(f"    └─ Value: {value:.2f}")
                
                elif indicator_name == "ichimoku":
                    values = detailed_data.get('values', {})
                    tenkan = values.get('tenkan_sen')
                    kijun = values.get('kijun_sen')
                    senkou_a = values.get('senkou_span_a')
                    senkou_b = values.get('senkou_span_b')
                    chikou = values.get('chikou_span')
                    
                    print(f"  - ICHIMOKU: {interpretation}")
                    if all(v is not None for v in [tenkan, kijun, senkou_a, senkou_b]):
                        print(f"    ├─ Tenkan-sen (Conversion): {tenkan:.2f}")
                        print(f"    ├─ Kijun-sen (Base): {kijun:.2f}")
                        print(f"    ├─ Senkou Span A (Leading A): {senkou_a:.2f}")
                        print(f"    ├─ Senkou Span B (Leading B): {senkou_b:.2f}")
                        if chikou is not None:
                            print(f"    └─ Chikou Span (Lagging): {chikou:.2f}")
                
                elif indicator_name == "obv":
                    value = detailed_data.get('value')
                    print(f"  - OBV: {interpretation}")
                    if value is not None:
                        print(f"    └─ Value: {value}")
                
                else:
                    print(f"  - {indicator_name.upper()}: {interpretation}")
                
                # Add explanations when explain flag is set
                if explain:
                    explanation = get_indicator_explanation(indicator_name)
                    if explanation:
                        print(f"    {explanation}")
    
    print("\n" + "=" * 70)


def _ensure_output_directory(output_type: str) -> str:
    """
    Ensure the output directory exists for the given output type.
    
    Args:
        output_type: Type of output (txt, json, html)
        
    Returns:
        Full path to the output directory
    """
    # Convert output type to lowercase and handle special cases
    output_type_lower = output_type.lower()
    if output_type_lower == 'txt':
        directory = 'txt'
    elif output_type_lower == 'jsf':
        directory = 'json'
    elif output_type_lower == 'html':
        directory = 'html'
    else:
        directory = output_type_lower
    
    # Create base directory
    base_dir = os.path.join(os.getcwd(), 'saved_analysis')
    if not os.path.exists(base_dir):
        try:
            os.makedirs(base_dir)
            logging.info(f"Created base directory: {base_dir}")
        except OSError as e:
            logging.error(f"Error creating base directory: {e}")
            return os.getcwd()  # Fallback to current directory
    
    # Create output type subdirectory
    output_dir = os.path.join(base_dir, directory)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logging.error(f"Error creating output directory: {e}")
            return base_dir  # Fallback to base directory
    
    return output_dir

def _generate_output_filename(symbol: str, timeframe: str, output_type: str) -> str:
    """
    Generate a filename for the output file.
    
    Args:
        symbol: Market symbol
        timeframe: Trading timeframe
        output_type: Type of output (txt, json, html)
        
    Returns:
        Full path to the output file
    """
    # Clean up symbol and timeframe for filename
    clean_symbol = symbol.replace('-', '_').replace('/', '_')
    clean_timeframe = timeframe.lower()
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine file extension
    if output_type.lower() == 'txt':
        extension = 'txt'
    elif output_type.lower() == 'jsf':
        extension = 'json'
    elif output_type.lower() == 'html':
        extension = 'html'
    else:
        extension = output_type.lower()
    
    # Generate filename
    filename = f"{clean_symbol}_{clean_timeframe}_{timestamp}.{extension}"
    
    # Get output directory
    output_dir = _ensure_output_directory(output_type)
    
    # Return full path
    return os.path.join(output_dir, filename)


def _strip_ansi_codes(text: str) -> str:
    """
    Remove all ANSI color and formatting codes from text.
    
    Args:
        text: Text string that may contain ANSI codes
        
    Returns:
        Clean text with all ANSI codes removed
    """
    # Standard ANSI escape pattern with explicit escape char
    text = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)
    
    # Handle escaped brackets patterns
    ansi_patterns = [
        r'\\\[[\d;]*m\\\]',     # Escaped bracket color codes: \[32m\]
        r'\[[\d;]*m',           # Regular color codes: [32m, [1;36m, [0m
        r'\[[\d;]*[a-zA-Z]',    # Other formatting codes: [1m, [A, etc.
        r'\[32m\[1m',           # Specifically catch [32m[1m
        r'\[33mInterpretation', # Specifically catch [33mInterpretation
        r'\[0m'                 # Specifically catch [0m
    ]
    
    # Apply all patterns
    for pattern in ansi_patterns:
        text = re.sub(pattern, '', text)
    
    # Specific pattern for bold/color formatting seen in the output
    text = re.sub(r'\[\d+;\d+m|\[\d+m|\[m', '', text)
    
    # Extremely aggressive pattern to catch anything that looks like an ANSI code
    text = re.sub(r'\[[^]]*?m', '', text)
    
    # Final cleanup for any remaining control sequences
    text = re.sub(r'\x1B|\033', '', text)
    
    return text


if __name__ == "__main__":
    analyzer_app() 