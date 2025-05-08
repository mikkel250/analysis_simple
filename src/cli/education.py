"""
Educational content for technical indicators used in market analysis.

This module provides explanations for various technical indicators, their interpretation,
and their significance in cryptocurrency market analysis.
"""

# Helper functions for formatted text
def category_header(text):
    """Format a category header - plain text version."""
    return f"📊 {text}"

def indicator_name(text):
    """Format an indicator name - plain text version."""
    return f"{text}"

def highlight(text):
    """Highlight important text - plain text version."""
    return f"{text}"

# Category explanations
CATEGORY_EXPLANATIONS = {
    "trend": (
        "Trend indicators help identify the direction of price movement over time. "
        "These indicators smooth out price data to show the underlying trend, filtering out "
        "short-term fluctuations and market noise. In cryptocurrency markets, which can be highly "
        "volatile, trend indicators help traders distinguish between temporary price swings and "
        "meaningful directional movements."
    ),
    
    "momentum": (
        "Momentum indicators measure the rate of price change rather than the price itself. "
        "They help identify the strength or weakness of a trend and potential reversal points. "
        "In cryptocurrency trading, momentum indicators are particularly useful for identifying "
        "overbought or oversold conditions in volatile markets, potentially signaling upcoming "
        "corrections or trend changes."
    ),
    
    "volatility": (
        "Volatility indicators measure the rate and magnitude of price changes. "
        "They help traders assess market risk and potential price movement. "
        "Cryptocurrencies are known for their high volatility, making these indicators "
        "especially valuable for setting appropriate stop-loss levels and profit targets."
    ),
    
    "volume": (
        "Volume indicators analyze trading volume alongside price movements. "
        "Strong volume confirms price trends, while divergences between volume and price "
        "may signal weakness in the current trend. In cryptocurrency markets, volume "
        "indicators can help distinguish between genuine market movements and "
        "manipulated price actions common in less regulated markets."
    ),
    
    "market_summary": (
        "The market summary combines signals from multiple indicators to provide an overall "
        "assessment of market conditions. It considers trend direction, strength, and signals "
        "across different timeframes to generate a suggested action. While these signals can "
        "be valuable, remember that technical analysis is just one aspect of trading strategy "
        "and should be combined with fundamental analysis and risk management."
    )
}

# Individual indicator explanations
INDICATOR_EXPLANATIONS = {
    "sma": (
        f"{indicator_name('Simple Moving Average (SMA)')} calculates the average price over a specified "
        f"period, giving equal weight to each price point. {highlight('Interpretation:')} When price is "
        f"above the SMA, it suggests bullish sentiment; when below, bearish sentiment. Longer period "
        f"SMAs (50-200) indicate long-term trends, while shorter ones (5-20) show short-term momentum. "
        f"Cryptocurrency traders often watch for price crossing above/below key SMAs or for SMA crossovers "
        f"as potential entry/exit signals."
    ),
    
    "ema": (
        f"{indicator_name('Exponential Moving Average (EMA)')} is similar to SMA but gives more weight to "
        f"recent prices, making it more responsive to new information. {highlight('Interpretation:')} Price "
        f"above EMA suggests bullish momentum; price below suggests bearish momentum. EMAs respond faster "
        f"to price changes than SMAs, potentially providing earlier trend change signals. In crypto markets, "
        f"which can change direction rapidly, EMAs are often preferred for their greater sensitivity."
    ),
    
    "macd": (
        f"{indicator_name('Moving Average Convergence Divergence (MACD)')} consists of the MACD line (difference "
        f"between two EMAs), signal line (EMA of the MACD line), and histogram (difference between MACD and "
        f"signal lines). {highlight('Interpretation:')} When MACD crosses above the signal line, it's a potential "
        f"buy signal; when it crosses below, a potential sell signal. The histogram shows momentum: increasing "
        f"bars suggest strengthening momentum, decreasing bars suggest weakening momentum. MACD is particularly "
        f"valuable in crypto markets for identifying potential reversals and trend strength."
    ),
    
    "adx": (
        f"{indicator_name('Average Directional Index (ADX)')} measures trend strength regardless of direction. "
        f"{highlight('Interpretation:')} Values below 20 indicate a weak trend or ranging market; values above "
        f"25 indicate a strong trend. Higher values mean stronger trends, not higher prices. ADX doesn't show "
        f"trend direction, only strength. In crypto markets, which can experience extended periods of sideways "
        f"movement, ADX helps traders determine whether trend-following strategies are appropriate."
    ),
    
    "rsi": (
        f"{indicator_name('Relative Strength Index (RSI)')} measures the speed and magnitude of price movements "
        f"on a scale from 0 to 100. {highlight('Interpretation:')} Traditionally, values above 70 indicate "
        f"overbought conditions (potential sell signal); values below 30 indicate oversold conditions (potential "
        f"buy signal). In stronger trends, these thresholds may shift (80/20 for strong markets). Cryptocurrency "
        f"markets can maintain overbought/oversold conditions longer than traditional markets, so look for "
        f"divergences and pattern breaks rather than relying solely on threshold values."
    ),
    
    "stoch": (
        f"{indicator_name('Stochastic Oscillator')} compares a closing price to its price range over a given time "
        f"period. It consists of %K (fast stochastic) and %D (slow stochastic) lines. {highlight('Interpretation:')} "
        f"Readings above 80 indicate overbought conditions; readings below 20 indicate oversold conditions. "
        f"A %K crossing above %D is a potential buy signal; crossing below is a potential sell signal. "
        f"In crypto markets, the stochastic oscillator works well for identifying potential reversal points "
        f"during sideways movements."
    ),
    
    "cci": (
        f"{indicator_name('Commodity Channel Index (CCI)')} measures a security's price deviation from its "
        f"statistical average. {highlight('Interpretation:')} Values above +100 indicate potential overbought "
        f"conditions; values below -100 indicate potential oversold conditions. Crossing from negative to "
        f"positive territory may signal bullish momentum; crossing from positive to negative may signal "
        f"bearish momentum. CCI can be particularly useful in crypto markets for identifying potential "
        f"price extremes and trend reversals."
    ),
    
    "bbands": (
        f"{indicator_name('Bollinger Bands')} consist of a middle band (typically a 20-period SMA) and two outer "
        f"bands set at standard deviations from the middle band. {highlight('Interpretation:')} Price touching "
        f"or exceeding the upper band suggests overbought conditions; touching the lower band suggests oversold "
        f"conditions. Band width indicates volatility: wider bands mean higher volatility; narrower bands mean "
        f"lower volatility. In crypto markets, known for their volatility, Bollinger Bands help identify potential "
        f"price breakouts and extremes."
    ),
    
    "atr": (
        f"{indicator_name('Average True Range (ATR)')} measures market volatility by calculating the average range "
        f"between high and low prices. {highlight('Interpretation:')} Higher ATR values indicate higher volatility; "
        f"lower values indicate lower volatility. ATR doesn't provide directional signals but helps set appropriate "
        f"stop-loss levels and profit targets. Given cryptocurrency's high volatility, ATR is essential for risk "
        f"management, helping traders set stops that account for normal market fluctuations."
    ),
    
    "obv": (
        f"{indicator_name('On-Balance Volume (OBV)')} adds volume on up days and subtracts volume on down days to "
        f"show buying and selling pressure. {highlight('Interpretation:')} Rising OBV suggests positive volume "
        f"pressure (buying); falling OBV suggests negative volume pressure (selling). Divergences between OBV and "
        f"price may signal potential reversals. In crypto markets, where volume can indicate institutional interest, "
        f"OBV helps validate price movements and identify potential manipulated rallies or selloffs."
    )
}

# Market summary explanations
SUMMARY_EXPLANATIONS = {
    "trend_direction": {
        "bullish": "A bullish trend indicates prices are generally moving upward.",
        "bearish": "A bearish trend indicates prices are generally moving downward.",
        "neutral": "A neutral trend indicates prices are moving sideways without a clear direction."
    },
    
    "trend_strength": {
        "strong": "Strong trends are likely to continue and are more resistant to reversals.",
        "moderate": "Moderate trends have reasonable momentum but may be more susceptible to reversals.",
        "weak": "Weak trends lack conviction and may be more likely to reverse or move into a range."
    },
    
    "signal_terms": {
        "short_term": "Short-term signals reflect immediate market conditions (hours to days).",
        "medium_term": "Medium-term signals reflect intermediate market conditions (days to weeks).",
        "long_term": "Long-term signals reflect extended market conditions (weeks to months)."
    },
    
    "actions": {
        "buy": "A buy signal suggests favorable conditions for entering long positions.",
        "sell": "A sell signal suggests favorable conditions for entering short positions or exiting longs.",
        "hold": "A hold signal suggests maintaining current positions without new entries."
    }
}

def get_category_explanation(category):
    """Get the explanation for a category of indicators."""
    return CATEGORY_EXPLANATIONS.get(category, "")

def get_indicator_explanation(indicator):
    """Get the explanation for a specific indicator."""
    return INDICATOR_EXPLANATIONS.get(indicator, "")

def get_summary_explanation(summary_type, value):
    """Get the explanation for a market summary item."""
    return SUMMARY_EXPLANATIONS.get(summary_type, {}).get(value.lower(), "")

def get_period_return_explanation(period_return: float) -> str:
    """
    Get educational explanation for period return value.
    
    Args:
        period_return: Percentage return over the analyzed period
        
    Returns:
        Educational text explaining period return calculation and interpretation
    """
    # Base explanation of calculation method
    explanation = "Period return measures price change as a percentage over the analyzed timeframe, calculated as ((current_price - starting_price) / starting_price) * 100."
    
    # Range-specific interpretation
    if period_return > 10:
        range_text = "This value indicates a strong bullish movement, potentially suggesting overextension and profit-taking opportunities."
    elif period_return > 5:
        range_text = "This value indicates a substantial positive movement, showing significant buying pressure."
    elif period_return > 2:
        range_text = "This value indicates moderate positive movement, above typical daily fluctuations."
    elif period_return >= -2:
        range_text = "This value indicates sideways price action, typical of consolidation or indecision phases."
    elif period_return >= -5:
        range_text = "This value indicates moderate negative movement, suggesting selling pressure."
    elif period_return >= -10:
        range_text = "This value indicates substantial negative movement, showing significant selling pressure."
    else:
        range_text = "This value indicates a strong bearish movement, potentially suggesting oversold conditions and potential reversal opportunities."
    
    return f"{explanation} {range_text}"

def get_volatility_explanation(volatility: float) -> str:
    """
    Get educational explanation for volatility value.
    
    Args:
        volatility: Volatility percentage value
        
    Returns:
        Educational text explaining volatility calculation and interpretation
    """
    # Base explanation of calculation method
    explanation = "Volatility measures price fluctuation magnitude. It's calculated as the percentage range between high and low prices relative to the low price."
    
    # Range-specific interpretation
    if volatility > 8:
        range_text = "This value falls into the 'extremely high daily volatility' range (>8%), indicating potential market uncertainty."
    elif volatility > 5:
        range_text = "This value falls into the 'high daily volatility' range (5-8%), suggesting active market conditions."
    elif volatility > 2:
        range_text = "This value falls into the 'moderate daily volatility' range (2-5%), which is typical of normal market conditions."
    else:
        range_text = "This value falls into the 'low daily volatility' range (<2%), suggesting a consolidation phase."
    
    return f"{explanation} {range_text}" 