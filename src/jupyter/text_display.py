"""
HTML-Based Text Display for Cryptocurrency Analysis

This module provides functions to create HTML-based text displays for cryptocurrency
analysis data, providing a simpler and more reliable alternative to complex
Plotly visualizations.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
from IPython.display import HTML
import pandas as pd


def get_color_for_value(value: float) -> str:
    """
    Get the appropriate color for a value based on whether it's positive or negative.
    
    Args:
        value: The numeric value to determine color for
        
    Returns:
        CSS color string (green for positive, red for negative, orange for zero)
    """
    if value > 0:
        return "chartreuse"  # Green for positive values
    elif value < 0:
        return "red"  # Red for negative values
    else:
        return "orange"  # Orange for zero/neutral values


def get_trend_color(trend: str) -> str:
    """
    Get the appropriate color for a market trend.
    
    Args:
        trend: Trend string ("bullish", "bearish", or "neutral")
        
    Returns:
        CSS color string
    """
    trend = trend.lower() if trend else "neutral"
    if trend == "bullish":
        return "chartreuse"
    elif trend == "bearish":
        return "red"
    else:
        return "orange"  # Default/neutral


def get_trend_symbol(trend: str) -> str:
    """
    Get the appropriate symbol for a market trend.
    
    Args:
        trend: Trend string ("bullish", "bearish", or "neutral")
        
    Returns:
        Unicode symbol representing the trend
    """
    trend = trend.lower() if trend else "neutral"
    if trend == "bullish":
        return "▲"  # Up triangle for bullish
    elif trend == "bearish":
        return "▼"  # Down triangle for bearish
    else:
        return "◆"  # Diamond for neutral


def get_indicator_education(indicator: str, current_value: float = None, signal: str = "neutral") -> str:
    """
    Generate educational content for a technical indicator, explaining what it is,
    how it's calculated, and what the current signal means.
    
    Args:
        indicator: The name of the indicator (e.g., "macd", "rsi")
        current_value: The current value of the indicator (optional)
        signal: The current signal from the indicator (bullish, bearish, neutral)
        
    Returns:
        HTML string with educational content about the indicator
    """
    # Normalize indicator name for matching
    indicator_lower = indicator.lower()
    signal_lower = signal.lower() if signal else "neutral"
    
    # Dictionary of educational content for common indicators
    indicator_info = {
        "macd": {
            "name": "Moving Average Convergence Divergence (MACD)",
            "description": "MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price.",
            "calculation": "MACD = 12-period EMA - 26-period EMA<br>Signal Line = 9-period EMA of MACD<br>MACD Histogram = MACD - Signal Line",
            "interpretation": {
                "bullish": "When MACD crosses above the signal line, it suggests upward momentum is increasing, indicating a potential buying opportunity.",
                "bearish": "When MACD crosses below the signal line, it suggests downward momentum is increasing, indicating a potential selling opportunity.",
                "neutral": "When MACD is moving parallel to the signal line, it suggests the current trend may be weakening or consolidating."
            }
        },
        "rsi": {
            "name": "Relative Strength Index (RSI)",
            "description": "RSI is a momentum oscillator that measures the speed and change of price movements on a scale from 0 to 100.",
            "calculation": "RSI = 100 - (100 / (1 + RS))<br>Where RS = Average Gain / Average Loss over a specified period (typically 14 days)",
            "interpretation": {
                "bullish": "When RSI moves out of oversold territory (below 30) and back above 30, it often signals a buying opportunity.",
                "bearish": "When RSI moves out of overbought territory (above 70) and back below 70, it often signals a selling opportunity.",
                "neutral": "When RSI is between 30 and 70, the market is considered to be in a neutral state without extreme conditions."
            }
        },
        "bollinger_bands": {
            "name": "Bollinger Bands",
            "description": "Bollinger Bands are volatility bands placed above and below a moving average, adaptively widening and narrowing when volatility increases and decreases.",
            "calculation": "Middle Band = 20-day simple moving average (SMA)<br>Upper Band = 20-day SMA + (20-day standard deviation of price x 2)<br>Lower Band = 20-day SMA - (20-day standard deviation of price x 2)",
            "interpretation": {
                "bullish": "When price moves from below the middle band to above it, especially after touching the lower band, it suggests a possible uptrend.",
                "bearish": "When price moves from above the middle band to below it, especially after touching the upper band, it suggests a possible downtrend.",
                "neutral": "When price oscillates around the middle band with low volatility (bands narrowing), it suggests consolidation before a significant move."
            }
        },
        "stochastic": {
            "name": "Stochastic Oscillator",
            "description": "The Stochastic Oscillator is a momentum indicator comparing a particular closing price to a range of prices over a certain period, generating overbought and oversold signals.",
            "calculation": "%K = 100 * ((Current Close - Lowest Low) / (Highest High - Lowest Low))<br>%D = 3-day SMA of %K",
            "interpretation": {
                "bullish": "When the %K line crosses above the %D line while both are below 20 (oversold), it generates a bullish signal.",
                "bearish": "When the %K line crosses below the %D line while both are above 80 (overbought), it generates a bearish signal.",
                "neutral": "When both %K and %D are between 20 and 80, the market is not at extreme levels and may be trending neutrally."
            }
        },
        "adx": {
            "name": "Average Directional Index (ADX)",
            "description": "ADX is a trend strength indicator that measures how strongly a price is trending, regardless of its direction.",
            "calculation": "ADX = 14-period smoothed average of the DI differences divided by the DI sum<br>Where DI+ and DI- are derived from Directional Movement (DM) calculations",
            "interpretation": {
                "bullish": "When ADX is above 25 and rising, and DI+ is above DI-, it indicates a strong uptrend.",
                "bearish": "When ADX is above 25 and rising, and DI- is above DI+, it indicates a strong downtrend.",
                "neutral": "When ADX is below 20, it suggests a weak or non-existent trend, and range-bound conditions may prevail."
            }
        },
        "obv": {
            "name": "On-Balance Volume (OBV)",
            "description": "OBV is a momentum indicator that uses volume flow to predict changes in stock price, measuring buying and selling pressure.",
            "calculation": "If today's close > yesterday's close: OBV = Previous OBV + Today's Volume<br>If today's close < yesterday's close: OBV = Previous OBV - Today's Volume<br>If today's close = yesterday's close: OBV = Previous OBV",
            "interpretation": {
                "bullish": "When OBV is rising while price is flat or rising, it suggests accumulation and confirms an uptrend or potential upward breakout.",
                "bearish": "When OBV is falling while price is flat or falling, it suggests distribution and confirms a downtrend or potential downward breakout.",
                "neutral": "When OBV moves sideways, it suggests no strong money flow in either direction, indicating possible price consolidation."
            }
        },
        "ema": {
            "name": "Exponential Moving Average (EMA)",
            "description": "EMA is a type of moving average that places greater weight on recent data points, making it more responsive to new information.",
            "calculation": "EMA = (Close - Previous EMA) * Multiplier + Previous EMA<br>Where Multiplier = 2 / (Time periods + 1)",
            "interpretation": {
                "bullish": "When price crosses above the EMA or a shorter-term EMA crosses above a longer-term EMA, it signals bullish momentum.",
                "bearish": "When price crosses below the EMA or a shorter-term EMA crosses below a longer-term EMA, it signals bearish momentum.",
                "neutral": "When price hovers around the EMA or EMAs are moving sideways, it suggests a neutral or consolidation phase."
            }
        },
        "sma": {
            "name": "Simple Moving Average (SMA)",
            "description": "SMA is the unweighted mean of a price over a specified number of time periods, providing a smooth price trend over time.",
            "calculation": "SMA = Sum of Closing Prices over N periods / N",
            "interpretation": {
                "bullish": "When price crosses above the SMA or a shorter-term SMA crosses above a longer-term SMA, it signals bullish momentum.",
                "bearish": "When price crosses below the SMA or a shorter-term SMA crosses below a longer-term SMA, it signals bearish momentum.",
                "neutral": "When price oscillates around the SMA or SMAs are moving sideways, it suggests a neutral or consolidation phase."
            }
        }
    }
    
    # Try to get info for the requested indicator
    indicator_data = None
    for key, data in indicator_info.items():
        if key in indicator_lower or indicator_lower in key:
            indicator_data = data
            break
    
    # If no specific info is found, use a generic template
    if indicator_data is None:
        display_name = " ".join(word.capitalize() for word in indicator.split("_"))
        indicator_data = {
            "name": display_name,
            "description": f"{display_name} is a technical indicator used in market analysis to help identify potential trends and signals.",
            "calculation": "The specific calculation depends on the parameters used for this indicator.",
            "interpretation": {
                "bullish": "A bullish signal typically suggests upward momentum or a potential buying opportunity.",
                "bearish": "A bearish signal typically suggests downward momentum or a potential selling opportunity.",
                "neutral": "A neutral signal suggests the market may be consolidating or lacks a clear direction."
            }
        }
    
    # Format the value if provided
    value_html = ""
    if current_value is not None:
        value_html = f"""
        <div class="indicator-value">
            <h4>Current Value:</h4>
            <p>{current_value:.2f}</p>
        </div>
        """
    
    # Build HTML content
    html = f"""
    <div class="indicator-education">
        <h3>{indicator_data["name"]}</h3>
        
        <div class="indicator-section">
            <h4>What it is:</h4>
            <p>{indicator_data["description"]}</p>
        </div>
        
        <div class="indicator-section">
            <h4>How it's calculated:</h4>
            <p>{indicator_data["calculation"]}</p>
        </div>
        
        {value_html}
        
        <div class="indicator-section">
            <h4>What it means now:</h4>
            <p>{indicator_data["interpretation"][signal_lower]}</p>
        </div>
    </div>
    """
    
    return html


def get_trend_education(trend_timeframe: str, signal: str = "neutral") -> str:
    """
    Generate educational content for market trends (short-term, medium-term, long-term),
    explaining what each timeframe means, how they're determined, and how to interpret signals.
    
    Args:
        trend_timeframe: The timeframe of the trend ("short_term", "medium_term", or "long_term")
        signal: The current signal for the trend (bullish, bearish, neutral)
        
    Returns:
        HTML string with educational content about the market trend
    """
    # Normalize trend timeframe and signal for matching
    timeframe_lower = trend_timeframe.lower()
    signal_lower = signal.lower() if signal else "neutral"
    
    # Dictionary of educational content for different trend timeframes
    trend_info = {
        "short_term": {
            "name": "Short-Term Trend (1-7 days)",
            "description": "Short-term trends reflect immediate market sentiment and are more sensitive to daily news, events, and trading activity. They are useful for timing entry and exit points for trades.",
            "determination": "Short-term trends are typically determined using moving averages (5, 10, 20-day), price patterns over recent sessions, intraday momentum indicators like RSI or MACD, and immediate support/resistance levels.",
            "interpretation": {
                "bullish": "A bullish short-term trend suggests immediate upward momentum with prices likely to continue rising over the next few days. This may present short-term buying opportunities but should be confirmed with other indicators.",
                "bearish": "A bearish short-term trend indicates immediate selling pressure with prices likely to decline over the next few days. Traders might consider reducing exposure or opening short positions, with appropriate risk management.",
                "neutral": "A neutral short-term trend suggests market indecision or consolidation. Prices may move sideways as buyers and sellers reach equilibrium. This could be a time to wait for clearer signals or focus on range-trading strategies."
            }
        },
        "medium_term": {
            "name": "Medium-Term Trend (1-4 weeks)",
            "description": "Medium-term trends reflect the market's direction over several weeks and are less affected by daily fluctuations. They're more reliable for position trades that last beyond a few days.",
            "determination": "Medium-term trends are identified using intermediate-length moving averages (20, 30, 50-day), multi-week chart patterns, broader support/resistance zones, and trend indicators like ADX (Average Directional Index).",
            "interpretation": {
                "bullish": "A bullish medium-term trend indicates sustainable upward momentum that could last several weeks. This may be suitable for swing trading or establishing positions with a multi-week horizon, especially if aligned with the longer-term trend.",
                "bearish": "A bearish medium-term trend signals a sustained downward move that may continue for weeks. Investors might consider hedging positions, while traders may look for shorting opportunities or avoid long positions until conditions improve.",
                "neutral": "A neutral medium-term trend indicates a transition period or consolidation phase. Markets may be preparing for a new trend or experiencing a pause in the existing trend. This could be an opportunity to reassess strategies or prepare for the next directional move."
            }
        },
        "long_term": {
            "name": "Long-Term Trend (1-6+ months)",
            "description": "Long-term trends represent the primary market direction over months to years. These trends are driven by fundamental factors, macroeconomic conditions, and broader market cycles in cryptocurrency adoption and development.",
            "determination": "Long-term trends are analyzed using long-duration moving averages (100, 200-day), major chart patterns, long-term support/resistance levels, on-chain metrics, fundamental analysis, and multi-month price structures.",
            "interpretation": {
                "bullish": "A bullish long-term trend suggests that the cryptocurrency is in a major upward cycle. This may represent a favorable environment for long-term investment, with short-term pullbacks potentially offering buying opportunities rather than reversal signals.",
                "bearish": "A bearish long-term trend indicates that the cryptocurrency is in a prolonged downward cycle. Long-term investors might reduce exposure, implement strong risk management, or look for alternatives until conditions change substantially.",
                "neutral": "A neutral long-term trend suggests the market is at a major inflection point or undergoing a lengthy consolidation. This may indicate the end of one market cycle and preparation for the next. Strategic planning and patience are advised during such phases."
            }
        }
    }
    
    # Try to get info for the requested timeframe
    timeframe_data = None
    for key, data in trend_info.items():
        if key in timeframe_lower or timeframe_lower in key:
            timeframe_data = data
            break
    
    # If no specific info is found, use a generic template
    if timeframe_data is None:
        display_name = " ".join(word.capitalize() for word in trend_timeframe.split("_"))
        timeframe_data = {
            "name": f"{display_name} Trend",
            "description": f"The {display_name.lower()} trend indicates the general market direction over its respective time period.",
            "determination": "Trends are typically determined using moving averages, price patterns, momentum indicators, and support/resistance levels appropriate for the timeframe.",
            "interpretation": {
                "bullish": "A bullish trend suggests upward momentum and potential buying opportunities, with appropriate risk management.",
                "bearish": "A bearish trend indicates downward momentum and potential selling pressure, suggesting caution for buyers.",
                "neutral": "A neutral trend suggests market indecision or consolidation, where prices may move sideways as buyers and sellers reach equilibrium."
            }
        }
    
    # Build HTML content
    html = f"""
    <div class="trend-education">
        <h3>{timeframe_data["name"]}</h3>
        
        <div class="trend-section">
            <h4>What it is:</h4>
            <p>{timeframe_data["description"]}</p>
        </div>
        
        <div class="trend-section">
            <h4>How it's determined:</h4>
            <p>{timeframe_data["determination"]}</p>
        </div>
        
        <div class="trend-section">
            <h4>What it means now:</h4>
            <p>{timeframe_data["interpretation"][signal_lower]}</p>
        </div>
    </div>
    """
    
    return html


def format_methodology_section() -> str:
    """
    Format the analysis methodology section to educate users on the approach.
    
    Returns:
        HTML string explaining the analysis methodology
    """
    html = """
    <div class="section methodology">
        <h2>Analysis Methodology</h2>
        <div class="methodology-content">
            <p>This analysis aims to both provide market insights and teach you how to perform 
            similar analysis yourself. Our approach uses multiple "puzzle pieces" to build a 
            comprehensive market picture:</p>
            
            <ul>
                <li><strong>Multiple Timeframes:</strong> Analysis across different time horizons 
                (short, medium, long-term) to identify trends that might be invisible when looking 
                at just one timeframe.</li>
                
                <li><strong>Multiple Indicators:</strong> We combine technical indicators, trend analysis, 
                candlestick patterns, and more to confirm or challenge our hypotheses.</li>
                
                <li><strong>Conflicting Signals:</strong> We deliberately present both confirming and 
                contradicting signals. For example, if technical indicators suggest a bullish trend 
                but candlestick patterns show bearish reversal patterns, both perspectives are presented 
                with their implications.</li>
            </ul>
            
            <p>This holistic approach helps you develop a more nuanced understanding of market 
            conditions rather than seeking a single "right answer." Learning to weigh multiple 
            factors is key to developing your own analysis skills.</p>
        </div>
    </div>
    """
    return html


def format_value_with_color(value: float, prefix: str = '', suffix: str = '') -> str:
    """
    Format a numeric value with appropriate color and optional prefix/suffix.
    
    Args:
        value: Numeric value to format
        prefix: Optional prefix string (e.g., "$" or "+")
        suffix: Optional suffix string (e.g., "%" or " USD")
        
    Returns:
        HTML string with the formatted value
    """
    color = get_color_for_value(value)
    # Add + sign for positive values when there's no prefix
    display_prefix = prefix if prefix else ("+" if value > 0 else "")
    return f'<span style="color: {color}; font-weight: bold;">{display_prefix}{value:.2f}{suffix}</span>'


def format_price_section(price_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """
    Format the price data and metadata section of the dashboard as HTML.
    
    Args:
        price_data: Dictionary containing price information
        metadata: Dictionary containing metadata (symbol, currency, timeframe)
        
    Returns:
        HTML string for the price section
    """
    # Extract metadata
    symbol = metadata.get("symbol", "BTC")
    vs_currency = metadata.get("vs_currency", "usd").upper()
    timeframe = metadata.get("timeframe", "1d")
    last_updated = metadata.get("last_updated", "")
    
    # Format last_updated if it exists
    last_updated_text = ""
    if last_updated:
        try:
            # Try to format the timestamp if it's ISO format
            from datetime import datetime
            dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
            last_updated_text = f"Last updated: {dt.strftime('%Y-%m-%d %H:%M:%S')}"
        except Exception:
            # If any error occurs, use the raw string
            last_updated_text = f"Last updated: {last_updated}"
    
    # Extract price data (handling missing data gracefully)
    current_price = price_data.get("current_price", 0)
    price_change_24h = price_data.get("price_change_24h", 0)
    price_change_pct_24h = price_data.get("price_change_percentage_24h", 0)
    market_cap = price_data.get("market_cap", 0)
    
    # Determine currency symbol
    currency_symbol = "$" if vs_currency == "USD" else "€" if vs_currency == "EUR" else "¥" if vs_currency == "JPY" else ""
    
    # Format values with colors
    formatted_price = f'<span class="price">{currency_symbol}{current_price:,.2f}</span>'
    formatted_change = format_value_with_color(price_change_pct_24h, suffix="%")
    formatted_change_value = format_value_with_color(price_change_24h, prefix=currency_symbol)
    
    # Create HTML section
    html = f"""
    <div class="section">
        <h2>{symbol}/{vs_currency} Price ({timeframe})</h2>
        <div class="metadata">{last_updated_text}</div>
        
        <div>
            <div>{formatted_price} <span class="change">24h: {formatted_change} ({formatted_change_value})</span></div>
        </div>
        
        <table>
            <tr>
                <th>Market Data</th>
                <th>Value</th>
            </tr>
    """
    
    # Add market cap if available
    if market_cap:
        html += f"""
            <tr>
                <td>Market Cap</td>
                <td>{currency_symbol}{market_cap:,.0f}</td>
            </tr>
        """
    
    # Add any additional price data available
    for key, value in price_data.items():
        # Skip keys we've already handled
        if key in ["current_price", "price_change_24h", "price_change_percentage_24h", "market_cap"]:
            continue
        
        # Format the key name for display (convert snake_case to Title Case)
        display_key = " ".join(word.capitalize() for word in key.split("_"))
        
        # Format the value based on its type
        if isinstance(value, (int, float)):
            # Format percentages appropriately
            if "percentage" in key or key.endswith("_pct"):
                formatted_value = format_value_with_color(value, suffix="%")
            # Format prices with currency symbol
            elif "price" in key or "value" in key:
                formatted_value = format_value_with_color(value, prefix=currency_symbol)
            # Format other numbers
            else:
                formatted_value = f"{value:,}"
        else:
            # Non-numeric values
            formatted_value = str(value)
        
        html += f"""
            <tr>
                <td>{display_key}</td>
                <td>{formatted_value}</td>
            </tr>
        """
    
    # Close table and section
    html += """
        </table>
    </div>
    """
    
    return html


def get_css_styles() -> str:
    """
    Get CSS styles for the HTML dashboard.
    
    Returns:
        String containing CSS styles
    """
    return """
    <style>
        .crypto-dashboard {
            font-family: Arial, sans-serif;
            background-color: rgba(30, 30, 30, 1);
            color: white;
            padding: 15px;
            border-radius: 8px;
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .section {
            background-color: rgba(40, 40, 40, 0.6);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .methodology {
            background-color: rgba(30, 30, 60, 0.6);
        }
        
        h2 {
            color: #eee;
            border-bottom: 1px solid #555;
            padding-bottom: 5px;
            margin-top: 5px;
            font-size: 18px;
        }
        
        h3 {
            color: #ddd;
            font-size: 16px;
            margin: 10px 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        
        th {
            text-align: left;
            padding: 8px;
            color: #aaa;
            border-bottom: 1px solid #555;
            font-weight: normal;
        }
        
        td {
            padding: 8px;
            border-bottom: 1px solid #444;
        }
        
        .price {
            font-size: 24px;
            font-weight: bold;
            color: white;
        }
        
        .change {
            margin-left: 15px;
            font-size: 16px;
        }
        
        .metadata {
            color: #888;
            font-size: 12px;
            margin-bottom: 10px;
        }
        
        .market-signals {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 20px;
        }
        
        .signal-block {
            flex: 1;
            min-width: 250px;
        }
        
        .methodology-content {
            color: #bbb;
            line-height: 1.4;
        }
        
        .methodology-content ul {
            padding-left: 20px;
        }
        
        .methodology-content li {
            margin-bottom: 10px;
        }
        
        /* Accordion Styles */
        details.indicator-details {
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 5px;
        }
        
        details.indicator-details summary {
            padding: 4px 0;
            cursor: pointer;
            list-style: none;
            font-weight: bold;
            position: relative;
            display: flex;
            align-items: center;
        }
        
        details.indicator-details summary::-webkit-details-marker {
            display: none;
        }
        
        details.indicator-details summary::before {
            content: "▶";
            display: inline-block;
            font-size: 10px;
            margin-right: 8px;
            transition: transform 0.3s;
            color: #888;
        }
        
        details.indicator-details[open] summary::before {
            transform: rotate(90deg);
        }
        
        .indicator-education {
            background: rgba(50, 50, 60, 0.7);
            border-radius: 4px;
            padding: 10px;
            margin-top: 5px;
            font-size: 14px;
            color: #ccc;
            line-height: 1.4;
        }
        
        .indicator-education h3 {
            margin-top: 0;
            color: #fff;
            font-size: 15px;
            border-bottom: 1px dotted #666;
            padding-bottom: 5px;
        }
        
        .indicator-education h4 {
            margin: 10px 0 5px 0;
            color: #9af;
            font-size: 14px;
        }
        
        .indicator-section {
            margin-bottom: 10px;
        }
        
        .indicator-value {
            background: rgba(0, 0, 0, 0.2);
            padding: 5px 10px;
            border-radius: 4px;
            display: inline-block;
            margin: 5px 0;
        }
        
        .indicator-value h4 {
            margin: 5px 0;
        }
        
        .indicator-value p {
            margin: 5px 0;
            font-weight: bold;
            color: #fff;
        }
    </style>
    """


def format_market_trend_analysis(analysis_results: Dict[str, Any]) -> str:
    """
    Format the market trend analysis section of the dashboard as HTML.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        HTML string for the market trend analysis section
    """
    # Extract trend data
    market_trends = analysis_results.get("market_trends", {})
    short_term = market_trends.get("short_term", "neutral")
    medium_term = market_trends.get("medium_term", "neutral")
    long_term = market_trends.get("long_term", "neutral")
    
    # Get symbols and colors for each trend
    short_term_symbol = get_trend_symbol(short_term)
    medium_term_symbol = get_trend_symbol(medium_term)
    long_term_symbol = get_trend_symbol(long_term)
    
    short_term_color = get_trend_color(short_term)
    medium_term_color = get_trend_color(medium_term)
    long_term_color = get_trend_color(long_term)
    
    # Format trend status with colors and symbols
    formatted_short_term = f'<span style="color: {short_term_color}; font-weight: bold;">{short_term_symbol} {short_term.upper()}</span>'
    formatted_medium_term = f'<span style="color: {medium_term_color}; font-weight: bold;">{medium_term_symbol} {medium_term.upper()}</span>'
    formatted_long_term = f'<span style="color: {long_term_color}; font-weight: bold;">{long_term_symbol} {long_term.upper()}</span>'
    
    # Create HTML section
    html = f"""
    <div class="section">
        <h2>Market Trend Analysis</h2>
        <div class="trend-analysis">
            <div><strong>Short-term (1-7 days):</strong> {formatted_short_term}</div>
            <div><strong>Medium-term (1-4 weeks):</strong> {formatted_medium_term}</div>
            <div><strong>Long-term (1-6 months):</strong> {formatted_long_term}</div>
        </div>
    </div>
    """
    
    return html


def format_technical_signals(analysis_results: Dict[str, Any]) -> str:
    """
    Format the technical signals section of the dashboard as HTML.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        HTML string for the technical signals section
    """
    # Extract signals data
    signals = analysis_results.get("signals", {})
    trend_signals = signals.get("trend_signals", {})
    oscillator_signals = signals.get("oscillator_signals", {})
    
    # Create tables for each signal type
    trend_html = """
        <div class="signal-block">
            <h3>Trend Indicators</h3>
            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Signal</th>
                </tr>
    """
    
    for indicator, signal in trend_signals.items():
        signal_color = get_trend_color(signal)
        signal_symbol = get_trend_symbol(signal)
        formatted_signal = f'<span style="color: {signal_color}; font-weight: bold;">{signal_symbol} {signal.upper()}</span>'
        display_indicator = " ".join(word.capitalize() for word in indicator.split("_"))
        
        # Get educational content
        indicator_values = analysis_results.get("indicator_values", {}).get(indicator, {})
        current_value = indicator_values.get("latest_value") if indicator_values else None
        
        education_content = get_indicator_education(indicator, current_value, signal)
        
        # Create accordion with educational content
        trend_html += f"""
            <tr>
                <td>
                    <details class="indicator-details">
                        <summary>{display_indicator}</summary>
                        {education_content}
                    </details>
                </td>
                <td>{formatted_signal}</td>
            </tr>
        """
    
    trend_html += """
            </table>
        </div>
    """
    
    oscillator_html = """
        <div class="signal-block">
            <h3>Oscillator Indicators</h3>
            <table>
                <tr>
                    <th>Indicator</th>
                    <th>Signal</th>
                </tr>
    """
    
    for indicator, signal in oscillator_signals.items():
        signal_color = get_trend_color(signal)
        signal_symbol = get_trend_symbol(signal)
        formatted_signal = f'<span style="color: {signal_color}; font-weight: bold;">{signal_symbol} {signal.upper()}</span>'
        display_indicator = " ".join(word.capitalize() for word in indicator.split("_"))
        
        # Get educational content
        indicator_values = analysis_results.get("indicator_values", {}).get(indicator, {})
        current_value = indicator_values.get("latest_value") if indicator_values else None
        
        education_content = get_indicator_education(indicator, current_value, signal)
        
        # Create accordion with educational content
        oscillator_html += f"""
            <tr>
                <td>
                    <details class="indicator-details">
                        <summary>{display_indicator}</summary>
                        {education_content}
                    </details>
                </td>
                <td>{formatted_signal}</td>
            </tr>
        """
    
    oscillator_html += """
            </table>
        </div>
    """
    
    # Create HTML section
    html = f"""
    <div class="section">
        <h2>Technical Signals</h2>
        <div class="market-signals">
            {trend_html}
            {oscillator_html}
        </div>
    </div>
    """
    
    return html


def format_candlestick_patterns(analysis_results: Dict[str, Any]) -> str:
    """
    Format the candlestick patterns section of the dashboard as HTML.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        HTML string for the candlestick patterns section
    """
    # Extract patterns data
    patterns = analysis_results.get("patterns", {})
    
    # If no patterns are found, return empty string
    if not patterns:
        return ""
    
    # Create HTML table for patterns
    patterns_html = """
        <table>
            <tr>
                <th>Pattern</th>
                <th>Signal</th>
                <th>Strength</th>
            </tr>
    """
    
    for pattern, data in patterns.items():
        signal = data.get("signal", "neutral")
        strength = data.get("strength", 0)
        
        signal_color = get_trend_color(signal)
        signal_symbol = get_trend_symbol(signal)
        formatted_signal = f'<span style="color: {signal_color}; font-weight: bold;">{signal_symbol} {signal.upper()}</span>'
        
        # Format pattern name (convert snake_case to Title Case)
        display_pattern = " ".join(word.capitalize() for word in pattern.split("_"))
        
        # Format strength as stars (1-5)
        stars = "★" * min(5, max(1, int(strength * 5)))
        formatted_strength = f'<span style="color: gold;">{stars}</span>'
        
        patterns_html += f"""
            <tr>
                <td>{display_pattern}</td>
                <td>{formatted_signal}</td>
                <td>{formatted_strength}</td>
            </tr>
        """
    
    patterns_html += """
        </table>
    """
    
    # Create HTML section
    html = f"""
    <div class="section">
        <h2>Candlestick Patterns</h2>
        {patterns_html}
    </div>
    """
    
    return html


def format_recommended_action(analysis_results: Dict[str, Any]) -> str:
    """
    Format the recommended action section of the dashboard as HTML.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        HTML string for the recommended action section
    """
    # Extract recommendation data
    recommendation = analysis_results.get("recommendation", {})
    action = recommendation.get("action", "hold")
    confidence = recommendation.get("confidence", 0.5)
    rationale = recommendation.get("rationale", "Based on current market conditions.")
    
    # Determine color based on action
    if action.lower() == "buy":
        action_color = "chartreuse"
        action_symbol = "▲"
    elif action.lower() == "sell":
        action_color = "red"
        action_symbol = "▼"
    else:  # hold or neutral
        action_color = "orange"
        action_symbol = "◆"
    
    # Format the action with color
    formatted_action = f'<span style="color: {action_color}; font-weight: bold; font-size: 24px;">{action_symbol} {action.upper()}</span>'
    
    # Format confidence as percentage
    confidence_pct = int(confidence * 100)
    formatted_confidence = f'<span style="font-weight: bold;">{confidence_pct}%</span>'
    
    # Create HTML section
    html = f"""
    <div class="section">
        <h2>Recommended Action</h2>
        <div style="margin-bottom: 10px;">
            {formatted_action} <span style="margin-left: 10px;">(Confidence: {formatted_confidence})</span>
        </div>
        <div style="color: #ddd;">
            <strong>Rationale:</strong> {rationale}
        </div>
    </div>
    """
    
    return html


def create_text_dashboard(analysis_results: Dict[str, Any]) -> HTML:
    """
    Create an HTML-based text dashboard for cryptocurrency analysis.
    
    Args:
        analysis_results: Dictionary containing analysis results
        
    Returns:
        IPython HTML object containing the dashboard
    """
    # Extract price data and metadata
    price_data = analysis_results.get("price_data", {})
    metadata = analysis_results.get("metadata", {})
    
    # Generate sections
    methodology_section = format_methodology_section()
    price_section = format_price_section(price_data, metadata)
    market_trend_section = format_market_trend_analysis(analysis_results)
    technical_signals_section = format_technical_signals(analysis_results)
    candlestick_patterns_section = format_candlestick_patterns(analysis_results)
    recommended_action_section = format_recommended_action(analysis_results)
    
    # Get CSS styles
    css_styles = get_css_styles()
    
    # Combine all sections
    dashboard_html = f"""
    {css_styles}
    <div class="crypto-dashboard">
        <h2 style="text-align: center; border-bottom: none; font-size: 28px; margin-bottom: 20px;">
            Cryptocurrency Analysis Dashboard
        </h2>
        
        {methodology_section}
        {price_section}
        {market_trend_section}
        {technical_signals_section}
        {candlestick_patterns_section}
        {recommended_action_section}
        
        <div class="metadata" style="text-align: center; margin-top: 20px; font-size: 12px;">
            This analysis is for educational purposes only and not financial advice.
        </div>
    </div>
    """
    
    # Return as IPython HTML object
    return HTML(dashboard_html) 