"""
Educational content for technical indicators used in market analysis.

This module provides explanations for various technical indicators, their
interpretation, and their significance in cryptocurrency market analysis.
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
        "These indicators smooth out price data to show the underlying trend, "
        "filtering out short-term fluctuations and market noise. In cryptocurrency "
        "markets, which can be highly volatile, trend indicators help traders "
        "distinguish between temporary price swings and meaningful directional "
        "movements."
    ),

    "momentum": (
        "Momentum indicators measure the rate of price change rather than the "
        "price itself. They help identify the strength or weakness of a trend and "
        "potential reversal points. In cryptocurrency trading, momentum indicators "
        "are particularly useful for identifying overbought or oversold conditions "
        "in volatile markets, potentially signaling upcoming corrections or trend "
        "changes."
    ),

    "volatility": (
        "Volatility indicators measure the rate and magnitude of price changes. "
        "They help traders assess market risk and potential price movement. "
        "Cryptocurrencies are known for their high volatility, making these "
        "indicators especially valuable for setting appropriate stop-loss levels "
        "and profit targets."
    ),

    "volume": (
        "Volume indicators analyze trading volume alongside price movements. "
        "Strong volume confirms price trends, while divergences between volume and "
        "price may signal weakness in the current trend. In cryptocurrency markets, "
        "volume indicators can help distinguish between genuine market movements "
        "and manipulated price actions common in less regulated markets."
    ),

    "market_summary": (
        "The market summary combines signals from multiple indicators to provide an "
        "overall assessment of market conditions. It considers trend direction, "
        "strength, and signals across different timeframes to generate a suggested "
        "action. While these signals can be valuable, remember that technical "
        "analysis is just one aspect of trading strategy and should be combined "
        "with fundamental analysis and risk management."
    )
}

# Individual indicator explanations
INDICATOR_EXPLANATIONS = {
    "sma": (
        f"{indicator_name('Simple Moving Average (SMA)')} calculates the average "
        f"price over a specified period, giving equal weight to each price point. "
        f"{highlight('Interpretation:')} When price is above the SMA, it suggests "
        f"bullish sentiment; when below, bearish sentiment. Longer period SMAs "
        f"(50-200) indicate long-term trends, while shorter ones (5-20) show "
        f"short-term momentum. Cryptocurrency traders often watch for price "
        f"crossing above/below key SMAs or for SMA crossovers as potential "
        f"entry/exit signals."
    ),

    "ema": (
        f"{indicator_name('Exponential Moving Average (EMA)')} is similar to SMA "
        f"but gives more weight to recent prices, making it more responsive to "
        f"new information. {highlight('Interpretation:')} Price above EMA suggests "
        f"bullish momentum; price below suggests bearish momentum. EMAs respond "
        f"faster to price changes than SMAs, potentially providing earlier trend "
        f"change signals. In crypto markets, which can change direction rapidly, "
        f"EMAs are often preferred for their greater sensitivity."
    ),

    "macd": (
        f"{indicator_name('Moving Average Convergence Divergence (MACD)')} consists "
        f"of the MACD line (difference between two EMAs), signal line (EMA of the "
        f"MACD line), and histogram (difference between MACD and signal lines). "
        f"{highlight('Interpretation:')} When MACD crosses above the signal line, "
        f"it's a potential buy signal; when it crosses below, a potential sell "
        f"signal. The histogram shows momentum: increasing bars suggest "
        f"strengthening momentum, decreasing bars suggest weakening momentum. "
        f"MACD is particularly valuable in crypto markets for identifying "
        f"potential reversals and trend strength."
    ),

    "adx": (
        f"{indicator_name('Average Directional Index (ADX)')} measures trend "
        f"strength regardless of direction. {highlight('Interpretation:')} Values "
        f"below 20 indicate a weak trend or ranging market; values above 25 "
        f"indicate a strong trend. Higher values mean stronger trends, not higher "
        f"prices. ADX doesn't show trend direction, only strength. In crypto "
        f"markets, which can experience extended periods of sideways movement, ADX "
        f"helps traders determine whether trend-following strategies are "
        f"appropriate."
    ),

    "rsi": (
        f"{indicator_name('Relative Strength Index (RSI)')} measures the speed and "
        f"magnitude of price movements on a scale from 0 to 100. "
        f"{highlight('Interpretation:')} Traditionally, values above 70 indicate "
        f"overbought conditions (potential sell signal); values below 30 indicate "
        f"oversold conditions (potential buy signal). In stronger trends, these "
        f"thresholds may shift (80/20 for strong markets). Cryptocurrency markets "
        f"can maintain overbought/oversold conditions longer than traditional "
        f"markets, so look for divergences and pattern breaks rather than relying "
        f"solely on threshold values."
    ),

    "stoch": (
        f"{indicator_name('Stochastic Oscillator')} compares a closing price to its price range over a given time period. It consists of %K (fast stochastic) and %D (slow stochastic) lines. "
        f"{highlight('Interpretation:')} Readings above 80 indicate overbought conditions; readings below 20 indicate oversold conditions. A %K crossing above %D is a potential buy signal; crossing below is a potential sell signal. In crypto markets, the stochastic oscillator works well for identifying potential reversal points during sideways movements."
    ),

    "cci": (
        f"{indicator_name('Commodity Channel Index (CCI)')} measures a security's price deviation from its statistical average. {highlight('Interpretation:')} Values above +100 indicate potential overbought conditions; values below -100 indicate potential oversold conditions. Crossing from negative to positive territory may signal bullish momentum; crossing from positive to negative may signal bearish momentum. CCI can be particularly useful in crypto markets for identifying potential price extremes and trend reversals."
    ),

    "bbands": (
        f"{indicator_name('Bollinger Bands')} consist of a middle band (typically a "
        f"20-period SMA) and two outer bands set at standard deviations from the "
        f"middle band. {highlight('Interpretation:')} Price touching or exceeding "
        f"the upper band suggests overbought conditions; touching the lower band "
        f"suggests oversold conditions. Band width indicates volatility: wider "
        f"bands mean higher volatility; narrower bands mean lower volatility. In "
        f"crypto markets, known for their volatility, Bollinger Bands help "
        f"identify potential price breakouts and extremes."
    ),

    "atr": (
        f"{indicator_name('Average True Range (ATR)')} measures market volatility "
        f"by calculating the average range between high and low prices. "
        f"{highlight('Interpretation:')} Higher ATR values indicate higher "
        f"volatility; lower values indicate lower volatility. ATR doesn't provide "
        f"directional signals but helps set appropriate stop-loss levels and profit "
        f"targets. Given cryptocurrency's high volatility, ATR is essential for "
        f"risk management, helping traders set stops that account for normal "
        f"market fluctuations."
    ),

    "obv": (
        f"{indicator_name('On-Balance Volume (OBV)')} adds volume on up days and "
        f"subtracts volume on down days to show buying and selling pressure. "
        f"{highlight('Interpretation:')} Rising OBV suggests positive volume "
        f"pressure (buying); falling OBV suggests negative volume pressure "
        f"(selling). Divergences between OBV and price may signal potential "
        f"reversals. In crypto markets, where volume can indicate institutional "
        f"interest, OBV helps validate price movements and identify potential "
        f"manipulated rallies or selloffs."
    ),

    "ichimoku": (
        f"{indicator_name('Ichimoku Cloud (Ichimoku Kinko Hyo)')} is a "
        f"comprehensive indicator defining support/resistance, momentum, and trend "
        f"direction. It consists of the Tenkan-sen (conversion line), Kijun-sen "
        f"(base line), Senkou Span A (leading span A), Senkou Span B (leading "
        f"span B), and Chikou Span (lagging span). The area between Senkou Span A "
        f"and B forms the 'Kumo' or Cloud. {highlight('Interpretation:')} Price "
        f"above the Cloud is bullish; below is bearish; inside is neutral/ranging. "
        f"Tenkan/Kijun crossovers can signal momentum shifts. A bullish Cloud "
        f"(Senkou A above B) and Chikou Span above price provide stronger bullish "
        f"confirmation. Due to its many components, Ichimoku provides a "
        f"multifaceted view, which is useful in dynamic crypto markets."
    ),

    "doji": (
        f"{indicator_name('Doji Candle')} forms when the open and close prices are nearly equal, resulting in a very small body. "
        f"{highlight('Interpretation:')} A doji signals indecision in the market and can precede a reversal or a pause in trend, especially after a strong move."
    ),

    "engulfing": (
        f"{indicator_name('Engulfing Pattern')} consists of two candles: a smaller candle followed by a larger one that completely engulfs the previous body. "
        f"{highlight('Interpretation:')} A bullish engulfing pattern (down candle followed by a strong up candle) can signal a reversal to the upside; a bearish engulfing (up candle followed by a strong down candle) can signal a reversal to the downside."
    ),

    "hammer": (
        f"{indicator_name('Hammer Candle')} has a small body near the top of the range and a long lower shadow. "
        f"{highlight('Interpretation:')} A hammer after a downtrend can signal a potential bullish reversal."
    ),

    "harami": (
        f"{indicator_name('Harami Pattern')} is a two-candle pattern where a small candle is contained within the prior candle's body. "
        f"{highlight('Interpretation:')} A bullish harami (small up candle within a large down candle) can signal a reversal to the upside; a bearish harami (small down candle within a large up candle) can signal a reversal to the downside."
    ),

    "morning_star": (
        f"{indicator_name('Morning Star Pattern')} is a three-candle pattern: a long down candle, a small-bodied candle (gap down), and a strong up candle. "
        f"{highlight('Interpretation:')} A morning star after a decline signals a potential bullish reversal."
    ),

    "psar": (
        f"{indicator_name('Parabolic SAR (Stop and Reverse)')} is a trend-following indicator that places dots above or below price to signal potential reversals. "
        f"{highlight('Interpretation:')} When price is above the SAR dots, it suggests a bullish trend; when price is below, it suggests a bearish trend. The SAR moves closer to price as the trend continues, and a crossover signals a possible reversal. In crypto markets, PSAR can help identify trend shifts and set trailing stops, but may generate whipsaws in choppy conditions."
    ),

    "williamsr": (
        f"{indicator_name('Williams %R')} is a momentum oscillator that measures overbought and oversold levels, ranging from 0 (overbought) to -100 (oversold). "
        f"{highlight('Interpretation:')} Readings above -20 indicate overbought conditions (potential reversal or pullback); readings below -80 indicate oversold conditions (potential upward reversal). Values between -20 and -80 are considered neutral. In crypto markets, Williams %R can help identify short-term extremes, but signals are more reliable when confirmed by trend or volume."
    ),

    "cmf": (
        f"{indicator_name('Chaikin Money Flow (CMF)')} measures the amount of Money Flow Volume over a specified period, combining price and volume to assess buying and selling pressure. "
        f"{highlight('Interpretation:')} CMF values above 0.1 indicate strong buying pressure (accumulation); values below -0.1 indicate strong selling pressure (distribution). Values between -0.1 and 0.1 are considered neutral. In crypto markets, CMF can help confirm trends and spot divergences between price and volume."
    ),

    "vwap": (
        f"{indicator_name('Volume Weighted Average Price (VWAP)')} calculates the average price of an asset, weighted by volume, over a specific period. "
        f"{highlight('Interpretation:')} Price above VWAP suggests bullish sentiment and institutional accumulation; price below VWAP suggests bearish sentiment and potential distribution. VWAP is often used by professional traders to gauge fair value and to identify intraday trend direction. In crypto markets, VWAP can help filter out noise and spot true price consensus."
    ),

    "heikinashi": (
        f"{indicator_name('Heikin Ashi')} is a type of candlestick chart that uses modified open, high, low, and close values to filter out market noise and highlight trend direction. "
        f"{highlight('Interpretation:')} A bullish Heikin Ashi candle (close > open) suggests upward momentum; a bearish candle (close < open) suggests downward momentum. Neutral candles indicate indecision. In crypto markets, Heikin Ashi can help traders stay in trends longer and avoid false signals from choppy price action."
    ),

    "dmi": (
        f"{indicator_name('Directional Movement Index (DMI)')} is a trend indicator that consists of three lines: +DI (positive directional indicator), -DI (negative directional indicator), and ADX (Average Directional Index). "
        f"{highlight('Interpretation:')} When +DI is above -DI and ADX is above 20, it suggests a bullish trend. When -DI is above +DI and ADX is above 20, it suggests a bearish trend. ADX measures trend strength, not direction. In crypto markets, DMI can help confirm trend direction and filter out weak or ranging periods."
    ),

    "kc": (
        f"{indicator_name('Keltner Channels')} are volatility-based envelopes set above and below an EMA, using the Average True Range (ATR) to set channel distance. "
        f"{highlight('Interpretation:')} Price above the upper channel suggests a bullish breakout; price below the lower channel suggests a bearish breakdown. Price within the channels indicates no strong directional signal. In crypto markets, Keltner Channels can help identify trending moves and filter out noise."
    ),

    "shooting_star": (
        f"{indicator_name('Shooting Star')} is a single-candle pattern with a small body, little or no lower shadow, and a long upper shadow. {highlight('Interpretation:')} It appears after an uptrend and signals a potential bearish reversal. In crypto, it can warn of exhaustion after a rally."
    ),

    "hanging_man": (
        f"{indicator_name('Hanging Man')} is a single-candle pattern with a small body near the top of the range and a long lower shadow. {highlight('Interpretation:')} It appears after an uptrend and signals a potential bearish reversal. In crypto, it can indicate profit-taking or a shift in sentiment."
    ),

    "three_white_soldiers": (
        f"{indicator_name('Three White Soldiers')} is a bullish reversal pattern of three consecutive long-bodied up candles, each closing higher. {highlight('Interpretation:')} It signals strong buying pressure and a potential trend reversal to the upside. In crypto, it can mark the start of a new rally."
    ),

    "three_black_crows": (
        f"{indicator_name('Three Black Crows')} is a bearish reversal pattern of three consecutive long-bodied down candles, each closing lower. {highlight('Interpretation:')} It signals strong selling pressure and a potential trend reversal to the downside. In crypto, it can mark the start of a new decline."
    ),

    "dark_cloud_cover": (
        f"{indicator_name('Dark Cloud Cover')} is a two-candle bearish reversal pattern: a strong up candle followed by a down candle that opens above the prior close and closes below the midpoint of the first. {highlight('Interpretation:')} It signals a potential shift from buying to selling pressure. In crypto, it can warn of a failed rally."
    ),

    "piercing_line": (
        f"{indicator_name('Piercing Line')} is a two-candle bullish reversal pattern: a strong down candle followed by an up candle that opens below the prior close and closes above the midpoint of the first. {highlight('Interpretation:')} It signals a potential shift from selling to buying pressure. In crypto, it can mark the start of a recovery."
    ),

    "evening_star": (
        f"{indicator_name('Evening Star')} is a three-candle bearish reversal pattern: a strong up candle, a small-bodied candle (gap up), and a strong down candle closing below the midpoint of the first. {highlight('Interpretation:')} It signals a potential top and reversal. In crypto, it can warn of a trend change after a rally."
    ),

    "spinning_top": (
        f"{indicator_name('Spinning Top')} is a single-candle pattern with a small real body and long upper and lower shadows. {highlight('Interpretation:')} It signals indecision and a possible pause or reversal. In crypto, it often appears during volatile, choppy markets."
    ),

    "marubozu": (
        f"{indicator_name('Marubozu')} is a single-candle pattern with a long body and little or no shadow. {highlight('Interpretation:')} A bullish marubozu (no upper/lower shadow) signals strong buying; a bearish marubozu signals strong selling. In crypto, it can indicate conviction and momentum."
    ),

    "open_interest": (
        f"{indicator_name('Open Interest (OI)')} counts only active, open positions in derivative contracts (like futures or options) that have been executed and are currently held open.\n"
        f"It does NOT include limit orders or any orders waiting to be filled.\n"
        f"OI increases when a new contract is opened (e.g., a buyer and a seller enter into a new futures contract).\n"
        f"OI decreases when a contract is closed (e.g., a position is settled, offset, or liquidated).\n"
        f"\n"
        f"{highlight('Interpretation:')} Rising OI with rising price is typically bullish (new money entering the market); rising OI with falling price is typically bearish (new shorts entering). Falling OI means positions are being closed, regardless of direction. OI is a measure of market participation, not direction.\n"
    ),

    "funding_rate": (
        f"{indicator_name('Funding Rate')} is a periodic payment exchanged between long and short positions in perpetual futures contracts. "
        f"A positive funding rate means longs pay shorts, indicating bullish sentiment or crowded long positions. "
        f"A negative funding rate means shorts pay longs, indicating bearish sentiment or crowded short positions. "
        f"Funding rates are typically small (e.g., 0.01% to 0.05% per 8 hours), but extreme values can signal market imbalance and potential for mean reversion or squeezes. "
        f"High positive rates may precede corrections as long positions become expensive to hold. High negative rates may precede short squeezes. "
        f"Funding rates help keep perpetual contract prices in line with spot prices. "
        f"{highlight('Interpretation:')} Watch for extreme funding rates as a sign of market crowding and possible reversals."
    ),

    # Advanced Technical Indicators (Batch 1)
    "williams_r": (
        f"{indicator_name('Williams %R')} is a momentum oscillator that measures overbought and oversold levels, ranging from 0 to -100. "
        f"It compares the closing price to the high-low range over a specified period. "
        f"{highlight('Interpretation:')} Values from 0 to -20 indicate overbought conditions (potential sell signal); "
        f"values from -80 to -100 indicate oversold conditions (potential buy signal). Unlike RSI, Williams %R is "
        f"inverted, with higher values being more bearish. In crypto markets, it's particularly useful for identifying "
        f"short-term reversal points and momentum shifts."
    ),

    "vortex": (
        f"{indicator_name('Vortex Indicator (VI)')} consists of two oscillators (VI+ and VI-) that capture positive "
        f"and negative trend movement by measuring the relationship between closing prices and true range. "
        f"{highlight('Interpretation:')} When VI+ crosses above VI-, it signals potential bullish momentum; when VI- "
        f"crosses above VI+, it signals potential bearish momentum. Values above 1.0 indicate strong directional "
        f"movement. In crypto markets, the Vortex Indicator helps identify trend changes and momentum shifts, "
        f"particularly effective during trending periods."
    ),

    "alma": (
        f"{indicator_name('Adaptive Linear Moving Average (ALMA)')} is a technical indicator that combines the "
        f"smoothness of a moving average with the responsiveness needed to track price changes effectively. "
        f"It uses a Gaussian filter and can be adjusted for smoothness and responsiveness. "
        f"{highlight('Interpretation:')} Price above ALMA suggests bullish momentum; price below suggests bearish "
        f"momentum. ALMA crossovers can signal trend changes. The indicator adapts to market conditions better "
        f"than traditional moving averages. In volatile crypto markets, ALMA provides smoother signals while "
        f"maintaining sensitivity to significant price movements."
    ),

    "kama": (
        f"{indicator_name('Kaufman Adaptive Moving Average (KAMA)')} is a moving average that adjusts its "
        f"smoothing based on market volatility and noise. It moves faster during trending periods and slower "
        f"during sideways markets. "
        f"{highlight('Interpretation:')} Price above KAMA indicates bullish conditions; price below indicates "
        f"bearish conditions. KAMA crossovers signal potential trend changes. The adaptive nature makes it "
        f"particularly valuable in crypto markets, which alternate between high volatility trending periods "
        f"and quieter consolidation phases."
    ),

    "trix": (
        f"{indicator_name('TRIX')} is a momentum oscillator that shows the rate of change of a triple exponentially "
        f"smoothed moving average. It filters out price movements that are considered insignificant or noise. "
        f"{highlight('Interpretation:')} When TRIX crosses above zero, it suggests bullish momentum; when it "
        f"crosses below zero, it suggests bearish momentum. TRIX signal line crossovers can provide entry/exit "
        f"signals. In crypto markets, TRIX helps filter out false signals common in volatile conditions while "
        f"identifying genuine trend changes."
    ),

    "ppo": (
        f"{indicator_name('Percentage Price Oscillator (PPO)')} is similar to MACD but shows the percentage "
        f"difference between two moving averages rather than the absolute difference. This makes it easier to "
        f"compare across different price levels. "
        f"{highlight('Interpretation:')} PPO above zero indicates bullish momentum; below zero indicates bearish "
        f"momentum. Signal line crossovers and histogram changes provide trading signals. In crypto markets, "
        f"PPO is particularly useful for comparing momentum across different cryptocurrencies or timeframes."
    ),

    "roc": (
        f"{indicator_name('Rate of Change (ROC)')} measures the percentage change in price over a specified "
        f"number of periods. It's a pure momentum oscillator that shows the velocity of price movement. "
        f"{highlight('Interpretation:')} Positive ROC values indicate upward momentum; negative values indicate "
        f"downward momentum. Extreme ROC values may signal overbought/oversold conditions. Zero-line crossovers "
        f"can signal momentum shifts. In crypto markets, ROC helps identify the strength of price movements "
        f"and potential reversal points."
    ),

    "aroon": (
        f"{indicator_name('Aroon Indicator')} consists of Aroon Up and Aroon Down lines that measure how long "
        f"it has been since the highest high and lowest low within a given period. Values range from 0 to 100. "
        f"{highlight('Interpretation:')} Aroon Up above 70 with Aroon Down below 30 suggests a strong uptrend; "
        f"Aroon Down above 70 with Aroon Up below 30 suggests a strong downtrend. When both are between 30-70, "
        f"the market is likely consolidating. In crypto markets, Aroon helps identify trend strength and "
        f"potential breakouts from consolidation periods."
    ),

    "fisher_transform": (
        f"{indicator_name('Fisher Transform')} converts price data into a Gaussian normal distribution, making "
        f"turning points easier to identify. It emphasizes when prices have moved to an extreme. "
        f"{highlight('Interpretation:')} Values above +1.5 suggest overbought conditions; values below -1.5 "
        f"suggest oversold conditions. The indicator crossing above/below zero can signal trend changes. "
        f"In crypto markets, the Fisher Transform is particularly effective at identifying reversal points "
        f"and extreme price movements that may be unsustainable."
    ),

    "awesome_oscillator": (
        f"{indicator_name('Awesome Oscillator (AO)')} measures market momentum by calculating the difference "
        f"between a 5-period and 34-period simple moving average of the midpoint (high+low)/2. "
        f"{highlight('Interpretation:')} Values above zero indicate bullish momentum; below zero indicates "
        f"bearish momentum. Color changes (green to red or vice versa) can signal momentum shifts. Zero-line "
        f"crossovers provide trend change signals. In crypto markets, AO helps identify momentum changes "
        f"and potential trend reversals, particularly effective in trending markets."
    ),

    "ultimate_oscillator": (
        f"{indicator_name('Ultimate Oscillator')} combines short, medium, and long-term price action into one "
        f"oscillator to avoid false signals common in single-timeframe indicators. It ranges from 0 to 100. "
        f"{highlight('Interpretation:')} Values above 70 indicate overbought conditions; values below 30 "
        f"indicate oversold conditions. Divergences between price and the oscillator can signal potential "
        f"reversals. In crypto markets, the Ultimate Oscillator's multi-timeframe approach helps filter "
        f"out noise and provides more reliable overbought/oversold signals."
    ),

    "cci_enhanced": (
        f"{indicator_name('Enhanced Commodity Channel Index (CCI)')} extends the traditional CCI with multiple "
        f"smoothing methods, multi-timeframe analysis, and divergence detection capabilities. "
        f"{highlight('Interpretation:')} Values above +100 indicate potential overbought conditions; values "
        f"below -100 indicate potential oversold conditions. The enhanced version provides additional insights "
        f"through different smoothing methods (SMA, EMA, WMA) and can analyze multiple timeframes simultaneously. "
        f"Divergence analysis helps identify potential reversal points. In crypto markets, the enhanced CCI "
        f"provides more comprehensive momentum analysis and better signal filtering."
    ),
}

# Market summary explanations
SUMMARY_EXPLANATIONS = {
    "trend_direction": {
        "bullish": "A bullish trend indicates prices are generally moving upward.",
        "bearish": "A bearish trend indicates prices are generally moving downward.",
        "neutral": "A neutral trend indicates prices are moving sideways without "
                   "a clear direction."
    },

    "trend_strength": {
        "strong": "Strong trends are likely to continue and are more resistant to "
                  "reversals.",
        "moderate": "Moderate trends have reasonable momentum but may be more "
                    "susceptible to reversals.",
        "weak": "Weak trends lack conviction and may be more likely to reverse or "
                "move into a range."
    },

    "signal_terms": {
        "short_term": "Short-term signals reflect immediate market conditions "
                      "(hours to days).",
        "medium_term": "Medium-term signals reflect intermediate market conditions "
                       "(days to weeks).",
        "long_term": "Long-term signals reflect extended market conditions "
                     "(weeks to months)."
    },

    "actions": {
        "buy": "A buy signal suggests favorable conditions for entering long "
               "positions.",
        "sell": "A sell signal suggests favorable conditions for entering short "
                "positions or exiting longs.",
        "hold": "A hold signal suggests maintaining current positions without new "
                "entries."
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
    explanation = (
        "Period return measures price change as a percentage over the analyzed "
        "timeframe, calculated as ((current_price - starting_price) / "
        "starting_price) * 100."
    )

    # Range-specific interpretation
    if period_return > 10:
        range_text = (
            "This value indicates a strong bullish movement, potentially suggesting "
            "overextension and profit-taking opportunities."
        )
    elif period_return > 5:
        range_text = (
            "This value indicates a substantial positive movement, showing "
            "significant buying pressure."
        )
    elif period_return > 2:
        range_text = (
            "This value indicates moderate positive movement, above typical daily "
            "fluctuations."
        )
    elif period_return >= -2:
        range_text = (
            "This value indicates sideways price action, typical of consolidation "
            "or indecision phases."
        )
    elif period_return >= -5:
        range_text = (
            "This value indicates moderate negative movement, suggesting selling "
            "pressure."
        )
    elif period_return >= -10:
        range_text = (
            "This value indicates substantial negative movement, showing "
            "significant selling pressure."
        )
    else:
        range_text = (
            "This value indicates a strong bearish movement, potentially suggesting "
            "oversold conditions and potential reversal opportunities."
        )

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
    explanation = (
        "Volatility measures price fluctuation magnitude. It's calculated as the "
        "percentage range between high and low prices relative to the low price."
    )

    # Range-specific interpretation
    if volatility > 8:
        range_text = (
            "This value falls into the 'extremely high daily volatility' range "
            "(>8%), indicating potential market uncertainty."
        )
    elif volatility > 5:
        range_text = (
            "This value falls into the 'high daily volatility' range (5-8%), "
            "suggesting active market conditions."
        )
    elif volatility > 2:
        range_text = (
            "This value falls into the 'moderate daily volatility' range (2-5%), "
            "which is typical of normal market conditions."
        )
    else:
        range_text = (
            "This value falls into the 'low daily volatility' range (<2%), "
            "suggesting a consolidation phase."
        )

    return f"{explanation} {range_text}"