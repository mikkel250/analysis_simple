"""
Educational Content for Cryptocurrency Dashboard

This module provides functions that generate educational content explaining technical
indicators, market trends, and other cryptocurrency analysis concepts.
"""

from typing import Dict, Any, Union, List, Tuple, Optional


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
    
    # Build the HTML
    html = f"""
    <div class="indicator-education">
        <h3>{indicator_data["name"]}</h3>
        <div class="indicator-description">
            <h4>What is it?</h4>
            <p>{indicator_data["description"]}</p>
        </div>
        <div class="indicator-calculation">
            <h4>How is it calculated?</h4>
            <p>{indicator_data["calculation"]}</p>
        </div>
        {value_html}
        <div class="indicator-interpretation">
            <h4>What does the current signal mean?</h4>
            <p>{indicator_data["interpretation"][signal_lower]}</p>
        </div>
    </div>
    """
    
    return html


def get_trend_education(trend_timeframe: str, signal: str = "neutral") -> str:
    """
    Generate educational content explaining market trends for different timeframes.
    
    Args:
        trend_timeframe: The timeframe of the trend (e.g., "short_term", "medium_term", "long_term")
        signal: The current trend signal (bullish, bearish, neutral)
        
    Returns:
        HTML string with educational content about market trends
    """
    # Normalize inputs
    timeframe_lower = trend_timeframe.lower()
    signal_lower = signal.lower() if signal else "neutral"
    
    # Dictionary of educational content for different timeframes
    timeframe_info = {
        "short_term": {
            "name": "Short-Term Trend",
            "description": "Short-term trends typically reflect market sentiment over hours to days, capturing immediate reactions to news and events.",
            "timeframe": "Typically 1-5 days",
            "indicators": "Often analyzed using hourly charts, momentum oscillators, and short-term moving averages (5-20 periods).",
            "interpretation": {
                "bullish": "A short-term bullish trend suggests immediate buying pressure and could offer quick trading opportunities, but may reverse quickly without broader market support.",
                "bearish": "A short-term bearish trend indicates immediate selling pressure and could offer short-selling opportunities, but may reverse quickly if the broader trend remains counter to it.",
                "neutral": "A neutral short-term trend suggests consolidation or indecision in the market, often preceding a breakout in either direction."
            }
        },
        "medium_term": {
            "name": "Medium-Term Trend",
            "description": "Medium-term trends typically reflect ongoing market sentiment and more significant price movements over weeks to months.",
            "timeframe": "Typically 3 weeks to 3 months",
            "indicators": "Often analyzed using daily charts, MACD, RSI, and intermediate moving averages (20-50 periods).",
            "interpretation": {
                "bullish": "A medium-term bullish trend suggests sustained buying interest and could offer opportunities for swing trades or position building over several weeks.",
                "bearish": "A medium-term bearish trend indicates persistent selling pressure and may warrant defensive positioning or consideration of reduced exposure over a multi-week period.",
                "neutral": "A neutral medium-term trend often reflects a transitional market phase or ongoing consolidation, providing time for strategic planning rather than immediate action."
            }
        },
        "long_term": {
            "name": "Long-Term Trend",
            "description": "Long-term trends capture the overall market direction over months to years, reflecting fundamental factors and major market cycles.",
            "timeframe": "Typically 6+ months to years",
            "indicators": "Often analyzed using weekly or monthly charts, long-term moving averages (100-200 periods), and fundamental factors.",
            "interpretation": {
                "bullish": "A long-term bullish trend suggests a potential multi-month or multi-year upward trajectory, supporting investment strategies focused on accumulation and holding positions through shorter-term fluctuations.",
                "bearish": "A long-term bearish trend indicates potential for extended price deterioration, warranting caution with new investments and consideration of more defensive asset allocation strategies.",
                "neutral": "A neutral long-term trend may indicate a major transition period in the market cycle, requiring careful monitoring of potential directional shifts while maintaining balanced positioning."
            }
        }
    }
    
    # Try to get info for the requested timeframe
    timeframe_data = None
    for key, data in timeframe_info.items():
        if key in timeframe_lower or timeframe_lower in key:
            timeframe_data = data
            break
    
    # If no specific info is found, use a generic template
    if timeframe_data is None:
        display_name = " ".join(word.capitalize() for word in trend_timeframe.split("_"))
        timeframe_data = {
            "name": display_name,
            "description": f"{display_name} trend analysis looks at price movements over a specific period to identify directional bias in the market.",
            "timeframe": "The specific timeframe depends on the parameters used for this analysis.",
            "indicators": "Various technical indicators may be used depending on the specific analysis timeframe.",
            "interpretation": {
                "bullish": "A bullish trend suggests upward momentum or a potential buying opportunity over this timeframe.",
                "bearish": "A bearish trend suggests downward momentum or a potential selling opportunity over this timeframe.",
                "neutral": "A neutral trend suggests the market may be consolidating or lacks a clear direction over this timeframe."
            }
        }
    
    # Build the HTML
    html = f"""
    <div class="trend-education">
        <h3>{timeframe_data["name"]}</h3>
        <div class="trend-description">
            <h4>What is it?</h4>
            <p>{timeframe_data["description"]}</p>
        </div>
        <div class="trend-timeframe">
            <h4>Typical Duration:</h4>
            <p>{timeframe_data["timeframe"]}</p>
        </div>
        <div class="trend-indicators">
            <h4>Common Indicators:</h4>
            <p>{timeframe_data["indicators"]}</p>
        </div>
        <div class="trend-interpretation">
            <h4>What does the current trend signal mean?</h4>
            <p>{timeframe_data["interpretation"][signal_lower]}</p>
        </div>
    </div>
    """
    
    return html


def format_methodology_section() -> str:
    """
    Generate educational content explaining the methodology used in the analysis.
    
    Returns:
        HTML string with educational content about the methodology
    """
    html = """
    <div class="methodology-section">
        <h3>Methodology</h3>
        <p>
            This analysis combines multiple technical indicators and market structure analysis to provide a comprehensive view of market conditions.
        </p>
        <h4>Technical Indicators</h4>
        <p>
            We utilize a basket of momentum, trend, and volatility indicators including MACD, RSI, Bollinger Bands, and moving averages,
            applying appropriate weightings based on market conditions.
        </p>
        <h4>Market Structure Analysis</h4>
        <p>
            Price action is analyzed across multiple timeframes to identify key support/resistance levels, chart patterns, and market cycles.
        </p>
        <h4>Signal Confirmation</h4>
        <p>
            Signals are considered stronger when multiple indicators and timeframes align, with conflicting signals resulting in more neutral outputs.
        </p>
        <div class="methodology-disclaimer">
            <p>
                <strong>Note:</strong> All technical analysis has limitations. Past patterns may not predict future movements with certainty.
                This dashboard should be used as one tool among many in your decision-making process.
            </p>
        </div>
    </div>
    """
    return html 