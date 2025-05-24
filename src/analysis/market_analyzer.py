"""
Market analyzer module for financial data analysis.

This module provides a MarketAnalyzer class that can be used to fetch, analyze, and visualize
financial market data using different trading timeframes.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import talib

from src.analysis import market_data
from src.plotting import charts
from src.cli.education import get_volatility_explanation

# Attempt to import AdvancedMarketAnalyzer, make it optional for now
try:
    from src.analysis.advanced_analyzer import AdvancedMarketAnalyzer
except ImportError:
    AdvancedMarketAnalyzer = None
    logger = logging.getLogger(
        __name__
    )  # Ensure logger is defined if import fails early
    logger.warning(
        "AdvancedMarketAnalyzer could not be imported. Advanced analytics will be unavailable."
    )


# Set up logging
logger = logging.getLogger(__name__)

# Default number of candles to fetch for historical data
DEFAULT_CANDLE_LIMIT = 200


# Helper function to convert confidence levels to numeric values for comparison
def confidence_level(confidence: str) -> int:
    """Convert confidence string to numeric value for comparison."""
    levels = {"high": 3, "medium": 2, "low": 1}
    return levels.get(confidence.lower(), 0)


class MarketAnalyzer:
    """
    Market analyzer for financial data analysis.

    This class provides methods to fetch, analyze, and visualize financial market data
    using specific string timeframes (e.g., '1h', '1d').
    """

    def __init__(self, symbol: str, timeframe: str = "1d", use_test_data: bool = False):
        self.symbol = symbol
        self.timeframe = timeframe.lower()
        self.use_test_data = use_test_data
        self.data: Optional[pd.DataFrame] = None
        self.performance: Optional[Dict[str, float]] = None
        self.analysis_results: Dict[str, Any] = {}  # Stores all detailed analysis
        self.visualizations: Dict[str, go.Figure] = {}

        logger.info(
            f"Initialized MarketAnalyzer for {self.symbol} with {self.timeframe} timeframe"
        )

    def fetch_data(self) -> pd.DataFrame:
        if self.use_test_data:
            dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq="15min")
            base_price = 50000 + np.random.randint(-5000, 5000)
            trend_val = np.random.choice([-0.05, 0, 0.05])
            trend = np.linspace(0, trend_val, 100)
            noise = np.random.normal(0, 0.015, 100)
            close_prices: np.ndarray = base_price * (1 + trend + noise)
            open_prices: np.ndarray = close_prices * (1 + np.random.normal(0, 0.002, 100))
            high_prices: np.ndarray = np.maximum(close_prices, open_prices) * (
                1 + abs(np.random.normal(0, 0.005, 100))
            )
            low_prices: np.ndarray = np.minimum(close_prices, open_prices) * (
                1 - abs(np.random.normal(0, 0.005, 100))
            )
            volumes = np.random.randint(500, 2000, 100) * (
                1 + abs(noise) * 10
            )
            self.data = pd.DataFrame(
                {
                    "date": dates,
                    "open": open_prices,
                    "high": high_prices,
                    "low": low_prices,
                    "close": close_prices,
                    "volume": volumes.astype(float),
                }
            )
            self.data.set_index("date", inplace=True)
            if self.data.index.tz is not None:
                self.data.index = self.data.index.tz_convert(None)
            logger.info(
                f"Using test data for {self.symbol}, shape: {self.data.shape}"
            )
            return self.data
        logger.info(
            f"Fetching data for {self.symbol} with timeframe={self.timeframe}, "
            f"limit={DEFAULT_CANDLE_LIMIT}"
        )
        try:
            from src.services.data_fetcher import get_historical_data
            self.data = get_historical_data(
                symbol=self.symbol,
                timeframe=self.timeframe,
                limit=DEFAULT_CANDLE_LIMIT,
                use_cache=True,
            )
            if self.data is None or self.data.empty:
                logger.warning(
                    f"No data returned for {self.symbol} with timeframe "
                    f"{self.timeframe}"
                )
                self.data = pd.DataFrame()
            else:
                logger.debug(
                    f"Successfully fetched data for {self.symbol}, shape: "
                    f"{self.data.shape}. Columns: {self.data.columns.tolist()}"
                )
        except Exception as e:
            logger.error(
                f"Error fetching data for {self.symbol}, timeframe "
                f"{self.timeframe}: {e}",
                exc_info=True,
            )
            self.data = pd.DataFrame()
        return self.data

    def run_analysis(self) -> None:
        """
        Run core market analysis (indicators, performance).
        Populates self.data (with indicators) and self.performance.
        """
        if self.data is None:
            self.fetch_data()
        if self.data is None or self.data.empty:
            logger.warning(
                f"Data for {self.symbol} is empty. Skipping core analysis."
            )
            self.analysis_results["error"] = (
                "Data could not be fetched or is empty."
            )
            self.performance = {}
            return
        logger.info(f"Running core analysis for {self.symbol}")
        # Add technical indicators using MarketData class
        market_data_obj = market_data.MarketData(self.data.copy(), self.symbol, self.timeframe)
        market_data_obj.add_technical_indicators()
        self.data = market_data_obj.data
        self.performance = market_data.get_performance_summary(self.data)
        self.analysis_results["data_with_indicators"] = self.data
        self.analysis_results["performance"] = self.performance
        # Fetch open interest and store in analysis_results
        try:
            from src.services.open_interest import fetch_open_interest
            oi_data = fetch_open_interest(self.symbol, exchange="okx")
            value = oi_data.get("open_interest_value") or oi_data.get("value")
            prev_value = None
            if value is not None and "open_interest_change_24h" in oi_data:
                change_pct = oi_data["open_interest_change_24h"]
                try:
                    prev_value = value / (1 + change_pct / 100) if change_pct is not None else None
                except Exception:
                    prev_value = None
            self.analysis_results["open_interest"] = {"value": value, "prev_value": prev_value}
        except Exception as e:
            logger.warning(f"Could not fetch open interest for {self.symbol}: {e}")
            self.analysis_results["open_interest"] = None
        # Detect candlestick patterns
        self.analysis_results["candlestick_patterns"] = self._detect_candlestick_patterns()
        logger.info(
            f"Core analysis complete for {self.symbol}. Data shape with "
            f"indicators: {self.data.shape}"
        )

    def generate_visualizations(self) -> None:
        """
        Generate visualizations for the data. Populates self.visualizations.
        """
        if self.data is None or self.data.empty or "close" not in self.data.columns:
            logger.warning(
                f"Cannot generate visualizations for {self.symbol} due to missing or empty data."
            )
            self.visualizations = {}
            return

        if self.analysis_results.get("error"):  # If core analysis failed
            logger.warning(
                f"Skipping visualizations due to error in core analysis for {self.symbol}."
            )
            self.visualizations = {}
            return

        logger.info(f"Generating visualizations for {self.symbol}")
        try:
            price_fig = charts.plot_price_history(self.data, self.symbol)
            technical_fig = charts.plot_technical_analysis(self.data, self.symbol)
            candlestick_fig = charts.plot_candlestick(self.data, self.symbol)

            self.visualizations = {
                "price_history": price_fig,
                "technical_indicators": technical_fig,
                "candlestick": candlestick_fig,
            }
            logger.info(f"Visualizations generated for {self.symbol}.")
        except Exception as e:
            logger.error(
                f"Error generating visualizations for {self.symbol}: {e}", exc_info=True
            )
            self.visualizations = {}

    def _get_price_summary(self) -> Dict[str, Any]:
        if self.data is None or self.data.empty or "close" not in self.data.columns:
            return {"error": "Price data not available for summary."}

        current_price = self.data["close"].iloc[-1]
        high_price = (
            self.data["high"].iloc[-1] if "high" in self.data.columns else current_price
        )
        low_price = (
            self.data["low"].iloc[-1] if "low" in self.data.columns else current_price
        )

        price_summary = {
            "current_price": f"{current_price:.2f}",
            "last_high": f"{high_price:.2f}",
            "last_low": f"{low_price:.2f}",
        }
        if self.performance:
            price_summary["period_return_pct"] = (
                f"{self.performance.get('total_return_pct', 0):.2f}%"
            )
            price_summary["start_price"] = (
                f"{self.performance.get('start_price', 0):.2f}"
            )
            price_summary["end_price"] = f"{self.performance.get('end_price', 0):.2f}"

        return price_summary

    def _get_volatility_summary(self) -> Dict[str, Any]:
        if self.performance is None:
            return {"error": "Performance data not available for volatility summary."}

        # Try both keys for compatibility
        volatility_pct = self.performance.get(
            "annualized_volatility_pct",
            self.performance.get("annual_volatility_pct", self.performance.get("volatility", 0))
        )
        # Handle N/A or missing values
        if volatility_pct is None or (isinstance(volatility_pct, str) and volatility_pct.lower() in ["n/a", "na"]):
            volatility_pct = 0.0
        try:
            volatility_pct = float(volatility_pct)
        except Exception:
            volatility_pct = 0.0

        vol_level = "N/A"
        if volatility_pct > 100:
            vol_level = "Very High"
        elif volatility_pct > 60:
            vol_level = "High"
        elif volatility_pct > 30:
            vol_level = "Moderate"
        elif volatility_pct > 0:
            vol_level = "Low"
        elif volatility_pct == 0:
            vol_level = "None/Unavailable"

        # Educational explanation logic
        if volatility_pct == 0 or np.isnan(volatility_pct):
            explanation = (
                "Volatility is reported as zero or unavailable. This usually means there was insufficient price history, all prices were constant, or a calculation issue occurred. "
                "A zero volatility value is only reasonable if the asset price did not change at all during the analyzed period. Otherwise, this may indicate missing or invalid data. "
                + get_volatility_explanation(volatility_pct)
            )
        else:
            explanation = get_volatility_explanation(volatility_pct)

        return {
            "annualized_volatility_pct": f"{volatility_pct:.2f}%",
            "volatility_level": vol_level,
            "max_drawdown_pct": f"{self.performance.get('max_drawdown_pct', 0):.2f}%",
            "type": "Returns-Based Volatility",
            "details": explanation
        }

    def _get_trend_summary(self) -> Dict[str, Any]:
        trend_info = self._determine_trend()
        if isinstance(trend_info, str):  # Simple trend string
            return {
                "overall_trend": trend_info,
                "details": "Trend determined by simple method.",
            }
        # Detailed trend object
        return trend_info

    def _get_support_resistance_summary(self) -> Dict[str, Any]:
        # This is a simplified S/R identification. A more robust one would involve more complex algorithms.
        if (
            self.data is None or self.data.empty or len(self.data) < 10
        ):  # Need some data points
            return {"status": "Not enough data for S/R levels"}

        recent_lows = self.data["low"].rolling(window=10, min_periods=3).min().dropna()
        recent_highs = (
            self.data["high"].rolling(window=10, min_periods=3).max().dropna()
        )

        current_price = self.data["close"].iloc[-1]

        support_levels = sorted(
            list(set(recent_lows[recent_lows < current_price].tail(3).tolist())),
            reverse=True,
        )
        resistance_levels = sorted(
            list(set(recent_highs[recent_highs > current_price].head(3).tolist()))
        )

        return {
            "support": [f"{s:.2f}" for s in support_levels][:2],  # Max 2 levels
            "resistance": [f"{r:.2f}" for r in resistance_levels][:2],  # Max 2 levels
            "note": "Basic S/R based on recent rolling min/max.",
        }

    def _get_volume_analysis_summary(self) -> Dict[str, Any]:
        if (
            self.data is None
            or "volume" not in self.data.columns
            or self.data["volume"].empty
        ):
            return {"status": "Volume data not available."}

        avg_volume = (
            self.data["volume"].rolling(window=20, min_periods=5).mean().iloc[-1]
        )
        current_volume = self.data["volume"].iloc[-1]

        status = "Normal"
        details = f"Current volume ({current_volume:,.0f}) is near the 20-period average ({avg_volume:,.0f})."
        if pd.isna(avg_volume) or avg_volume == 0:
            details = f"Current volume is {current_volume:,.0f}. Average volume not available or zero."
        elif current_volume > avg_volume * 2:
            status = "Volume Spike"
            details = f"Volume spike: current ({current_volume:,.0f}) significantly above average ({avg_volume:,.0f})."
        elif current_volume < avg_volume * 0.5:
            status = "Low Volume"
            details = f"Low volume: current ({current_volume:,.0f}) significantly below average ({avg_volume:,.0f})."

        return {"volume_status": status, "details": details}

    def get_summary(self) -> Dict[str, Any]:
        """
        Generate a comprehensive textual summary of the market analysis, using enhanced trend and indicator confluence.
        """
        if self.data is None or self.data.empty:
            self.run_analysis()  # Ensure core analysis is run

        if self.analysis_results.get("error"):
            return {"error": self.analysis_results["error"]}

        if self.data is None or self.data.empty:
            return {
                "error": "Data could not be fetched or is still empty after analysis attempt."
            }

        logger.info(f"Generating summary for {self.symbol}")

        price_summary = self._get_price_summary()
        volatility_summary = self._get_volatility_summary()
        trend_summary = self._get_trend_summary()
        sr_summary = self._get_support_resistance_summary()
        volume_summary = self._get_volume_analysis_summary()
        indicators_summary = self._summarize_indicators()
        confluence_note = indicators_summary.get("overall_indicator_confluence", {}).get("note", "")

        general_overview = (
            f"Analysis for {self.symbol} ({self.timeframe}): "
            f"Current price is {price_summary.get('current_price', 'N/A')}. "
            f"The overall trend is considered '{trend_summary.get('overall_trend', 'N/A')}'. "
            f"Volatility is {volatility_summary.get('volatility_level', 'N/A')} "
            f"({volatility_summary.get('annualized_volatility_pct', 'N/A')}). "
            f"Indicator confluence: {confluence_note}"
        )

        summary_dict = {
            "general_overview": general_overview,
            "price_snapshot": price_summary,
            "trend_analysis": trend_summary,
            "volatility_analysis": volatility_summary,
            "support_resistance": sr_summary,
            "volume_analysis": volume_summary,
            "key_indicators": indicators_summary,
        }
        self.analysis_results.update(summary_dict)
        return summary_dict

    def _determine_trend(self) -> Dict[str, Any]:
        """
        Determine the market trend using multiple indicators, with educational ADX interpretation.
        """
        if (
            self.data is None or self.data.empty or len(self.data) < 20
        ):
            return {
                "overall_trend": "undetermined",
                "confidence": "low",
                "details": "Insufficient data for trend determination.",
                "explanation": "Not enough data to determine trend."
            }

        signals = []
        details = []
        explanations = []

        sma_short_col = next(
            (
                col
                for col in self.data.columns
                if col.startswith("SMA_") and int(col.split("_")[1]) <= 20
            ),
            None,
        )
        sma_medium_col = next(
            (
                col
                for col in self.data.columns
                if col.startswith("SMA_") and 20 < int(col.split("_")[1]) <= 50
            ),
            None,
        )
        sma_long_col = next(
            (
                col
                for col in self.data.columns
                if col.startswith("SMA_") and int(col.split("_")[1]) > 50
            ),
            None,
        )

        current_price = self.data["close"].iloc[-1]

        if sma_short_col and sma_medium_col:
            sma_short = self.data[sma_short_col].iloc[-1]
            sma_medium = self.data[sma_medium_col].iloc[-1]
            if pd.notna(sma_short) and pd.notna(sma_medium):
                if current_price > sma_short > sma_medium:
                    signals.append("bullish")
                    details.append(
                        f"Price above {sma_short_col} ({sma_short:.2f}) and {sma_medium_col} ({sma_medium:.2f}); {sma_short_col} above {sma_medium_col} (Bullish MA alignment)."
                    )
                    explanations.append("Bullish trend: Price is above both short and medium-term moving averages, and the short MA is above the medium MA, indicating upward momentum.")
                elif current_price < sma_short < sma_medium:
                    signals.append("bearish")
                    details.append(
                        f"Price below {sma_short_col} ({sma_short:.2f}) and {sma_medium_col} ({sma_medium:.2f}); {sma_short_col} below {sma_medium_col} (Bearish MA alignment)."
                    )
                    explanations.append("Bearish trend: Price is below both short and medium-term moving averages, and the short MA is below the medium MA, indicating downward momentum.")
                else:
                    signals.append("mixed")
                    details.append(
                        f"Price ({current_price:.2f}) shows mixed signals with {sma_short_col} ({sma_short:.2f}) and {sma_medium_col} ({sma_medium:.2f})."
                    )
                    explanations.append("Mixed trend: Price and moving averages do not show a clear alignment, suggesting indecision or transition.")

        # MACD
        macd_line_col = next(
            (
                col
                for col in self.data.columns
                if col.upper().startswith("MACD_")
                and not col.upper().endswith("_H")
                and not col.upper().endswith("_S")
            ),
            None,
        )
        macd_signal_col = next(
            (col for col in self.data.columns if col.upper().startswith("MACDS_")), None
        )
        macd_hist_col = next(
            (col for col in self.data.columns if col.upper().startswith("MACDH_")), None
        )

        if macd_line_col and macd_signal_col and macd_hist_col:
            macd_line = self.data[macd_line_col].iloc[-1]
            macd_signal_val = self.data[macd_signal_col].iloc[-1]
            macd_hist = self.data[macd_hist_col].iloc[-1]
            if (
                pd.notna(macd_line)
                and pd.notna(macd_signal_val)
                and pd.notna(macd_hist)
            ):
                if macd_line > macd_signal_val and macd_hist > 0:
                    signals.append("bullish")
                    details.append(
                        f"MACD line ({macd_line:.2f}) above signal ({macd_signal_val:.2f}), histogram positive ({macd_hist:.2f}) (Bullish MACD)."
                    )
                    explanations.append("Bullish MACD: MACD line is above the signal line and histogram is positive, indicating bullish momentum.")
                elif macd_line < macd_signal_val and macd_hist < 0:
                    signals.append("bearish")
                    details.append(
                        f"MACD line ({macd_line:.2f}) below signal ({macd_signal_val:.2f}), histogram negative ({macd_hist:.2f}) (Bearish MACD)."
                    )
                    explanations.append("Bearish MACD: MACD line is below the signal line and histogram is negative, indicating bearish momentum.")
                else:
                    signals.append("mixed")
                    details.append(
                        f"MACD ({macd_line:.2f}) and signal ({macd_signal_val:.2f}) show mixed/crossing signals."
                    )
                    explanations.append("Mixed MACD: MACD and signal line are close or crossing, suggesting indecision.")

        # ADX (Trend Strength)
        adx_col = next((col for col in self.data.columns if "ADX" in col.upper()), None)
        adx_strength = "weak"
        adx_value = None
        if adx_col:
            adx_value = self.data[adx_col].iloc[-1]
            if pd.notna(adx_value):
                if adx_value > 25:
                    adx_strength = "strong"
                    details.append(f"ADX at {adx_value:.2f} indicates a strong trend.")
                    explanations.append("ADX above 25: Indicates a strong trend, confirming the reliability of other trend signals.")
                elif adx_value > 20:
                    adx_strength = "developing"
                    details.append(f"ADX at {adx_value:.2f} suggests a developing trend.")
                    explanations.append("ADX between 20 and 25: Indicates a trend is developing, but not yet strong.")
                else:
                    adx_strength = "weak"
                    details.append(f"ADX at {adx_value:.2f} suggests a weak or non-trending market.")
                    explanations.append("ADX below 20: Indicates a weak or non-existent trend, so trend signals are less reliable.")
                    signals.append("neutral")

        # Determine overall trend
        bullish_count = signals.count("bullish")
        bearish_count = signals.count("bearish")

        overall_trend = "neutral"
        confidence = "low"

        if bullish_count > bearish_count:
            overall_trend = f"bullish ({adx_strength} strength)"
            confidence = "medium" if bullish_count >= len(signals) * 0.6 else "low"
            if adx_col and adx_value and adx_value > 25:
                confidence = "high"
        elif bearish_count > bullish_count:
            overall_trend = f"bearish ({adx_strength} strength)"
            confidence = "medium" if bearish_count >= len(signals) * 0.6 else "low"
            if adx_col and adx_value and adx_value > 25:
                confidence = "high"
        elif not signals:
            overall_trend = "undetermined"
            details.append("Not enough primary trend indicators available.")
            explanations.append("Not enough primary trend indicators available to determine trend.")
        else:
            overall_trend = "mixed"
            details.append("Indicators provide mixed trend signals.")
            explanations.append("Indicators are mixed, so no clear trend is present.")

        return {
            "overall_trend": overall_trend,
            "confidence": confidence,
            "signal_counts": {
                "bullish": bullish_count,
                "bearish": bearish_count,
                "mixed_neutral": signals.count("mixed") + signals.count("neutral"),
            },
            "details": details,
            "num_signals_considered": len(signals),
            "adx_strength": adx_strength,
            "explanation": "; ".join(explanations)
        }

    def _summarize_indicators(self) -> Dict[str, Any]:
        """
        Create a summary of key technical indicators and their signals, with confluence and educational explanations.
        """
        if self.data is None or self.data.empty:
            return {"error": "Data not available for indicator summary."}

        indicators_summary = {}
        default_indicators_config = self._get_default_indicators()
        confluence_signals = []
        for key, config in default_indicators_config.items():
            indicator_data = {"name": config.get("display_name", key.upper())}
            values = {}
            explanation = ""
            if key == "sma" and "lengths" in config.get("params", {}):
                for length in config["params"]["lengths"]:
                    col_name = f"SMA_{length}"
                    if col_name in self.data.columns:
                        ma_val = self.data[col_name].iloc[-1]
                        price = self.data["close"].iloc[-1]
                        values[f"SMA {length}"] = f"{ma_val:.2f}"
                        if price > ma_val:
                            values[f"Signal {length}"] = "Price above MA (bullish)"
                            confluence_signals.append("bullish")
                        else:
                            values[f"Signal {length}"] = "Price below MA (bearish)"
                            confluence_signals.append("bearish")
                explanation = "Price above a moving average is generally bullish, below is bearish. Multiple MAs in agreement strengthen the signal."
            elif key == "ema" and "lengths" in config.get("params", {}):
                for length in config["params"]["lengths"]:
                    col_name = f"EMA_{length}"
                    if col_name in self.data.columns:
                        ma_val = self.data[col_name].iloc[-1]
                        price = self.data["close"].iloc[-1]
                        values[f"EMA {length}"] = f"{ma_val:.2f}"
                        if price > ma_val:
                            values[f"Signal {length}"] = "Price above EMA (bullish)"
                            confluence_signals.append("bullish")
                        else:
                            values[f"Signal {length}"] = "Price below EMA (bearish)"
                            confluence_signals.append("bearish")
                explanation = "Price above an EMA is generally bullish, below is bearish. Multiple EMAs in agreement strengthen the signal."
            elif key == "rsi":
                col_name = "RSI_14"
                if col_name in self.data.columns:
                    rsi_val = self.data[col_name].iloc[-1]
                    values["RSI (14)"] = f"{rsi_val:.2f}"
                    if rsi_val > 70:
                        values["Signal"] = "Overbought"
                        confluence_signals.append("bearish")
                        explanation = "RSI above 70 is considered overbought, which can signal a potential reversal or pullback."
                    elif rsi_val < 30:
                        values["Signal"] = "Oversold"
                        confluence_signals.append("bullish")
                        explanation = "RSI below 30 is considered oversold, which can signal a potential upward reversal."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
                else:
                    values["Signal"] = "Neutral"
                    confluence_signals.append("neutral")
            elif key == "macd":
                macd_line_col = next(
                    (
                        c
                        for c in self.data.columns
                        if c.upper().startswith("MACD_")
                        and not c.upper().endswith("_H")
                        and not c.upper().endswith("_S")
                    ),
                    "MACD_12_26_9",
                )
                macd_signal_col = next(
                    (c for c in self.data.columns if c.upper().startswith("MACDS_")),
                    "MACDS_12_26_9",
                )
                macd_hist_col = next(
                    (c for c in self.data.columns if c.upper().startswith("MACDH_")),
                    "MACDH_12_26_9",
                )
                if (
                    macd_line_col in self.data.columns
                    and macd_signal_col in self.data.columns
                    and macd_hist_col in self.data.columns
                ):
                    macd_line = self.data[macd_line_col].iloc[-1]
                    macd_signal = self.data[macd_signal_col].iloc[-1]
                    macd_hist = self.data[macd_hist_col].iloc[-1]
                    values["MACD Line"] = f"{macd_line:.2f}"
                    values["Signal Line"] = f"{macd_signal:.2f}"
                    values["Histogram"] = f"{macd_hist:.2f}"
                    if macd_line > macd_signal and macd_hist > 0:
                        values["Signal"] = "Bullish Crossover"
                        confluence_signals.append("bullish")
                        explanation = "MACD line above the signal line with a positive histogram is bullish."
                    elif macd_line < macd_signal and macd_hist < 0:
                        values["Signal"] = "Bearish Crossover"
                        confluence_signals.append("bearish")
                        explanation = "MACD line below the signal line with a negative histogram is bearish."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
            elif key == "bbands":
                bbl_col = next(
                    (c for c in self.data.columns if c.upper().startswith("BBL_")),
                    "BBL_20_2.0",
                )
                bbm_col = next(
                    (c for c in self.data.columns if c.upper().startswith("BBM_")),
                    "BBM_20_2.0",
                )
                bbu_col = next(
                    (c for c in self.data.columns if c.upper().startswith("BBU_")),
                    "BBU_20_2.0",
                )
                if (
                    bbl_col in self.data.columns
                    and bbm_col in self.data.columns
                    and bbu_col in self.data.columns
                ):
                    lower = self.data[bbl_col].iloc[-1]
                    upper = self.data[bbu_col].iloc[-1]
                    close = self.data["close"].iloc[-1]
                    values["Lower Band"] = f"{lower:.2f}"
                    values["Upper Band"] = f"{upper:.2f}"
                    if close > upper:
                        values["Signal"] = "Price above Upper Band"
                        confluence_signals.append("bearish")
                        explanation = "Price above the upper Bollinger Band can indicate overbought conditions."
                    elif close < lower:
                        values["Signal"] = "Price below Lower Band"
                        confluence_signals.append("bullish")
                        explanation = "Price below the lower Bollinger Band can indicate oversold conditions."
                    else:
                        values["Signal"] = "Within Bands"
                        confluence_signals.append("neutral")
            elif key == "atr":
                atr_col = next(
                    (
                        c
                        for c in self.data.columns
                        if c.upper().startswith("ATRP_") or c.upper().startswith("ATR_")
                    ),
                    "ATRP_14",
                )
                if atr_col in self.data.columns:
                    atr_val = self.data[atr_col].iloc[-1]
                    unit = "%" if "P" in atr_col.upper() else ""
                    values[f"ATR ({'percent' if unit else 'value'})"] = (
                        f"{atr_val:.2f}{unit}"
                    )
                    explanation = "ATR measures average volatility. Higher ATR means more price movement."
            elif key == "psar":
                psar_col = next((c for c in self.data.columns if c.lower().startswith("psar")), None)
                if psar_col and psar_col in self.data.columns:
                    psar_val = self.data[psar_col].iloc[-1]
                    price = self.data["close"].iloc[-1]
                    values["PSAR"] = f"{psar_val:.2f}"
                    if price > psar_val:
                        values["Signal"] = "Bullish (price above PSAR)"
                        confluence_signals.append("bullish")
                        explanation = "Price above the Parabolic SAR suggests a bullish trend."
                    else:
                        values["Signal"] = "Bearish (price below PSAR)"
                        confluence_signals.append("bearish")
                        explanation = "Price below the Parabolic SAR suggests a bearish trend."
            elif key == "willr":
                col_name = "WILLR_14"
                if col_name in self.data.columns:
                    wr_val = self.data[col_name].iloc[-1]
                    values["Williams %R (14)"] = f"{wr_val:.2f}"
                    if wr_val > -20:
                        values["Signal"] = "Overbought"
                        confluence_signals.append("bearish")
                        explanation = "Williams %R above -20 is considered overbought, which can signal a potential reversal or pullback."
                    elif wr_val < -80:
                        values["Signal"] = "Oversold"
                        confluence_signals.append("bullish")
                        explanation = "Williams %R below -80 is considered oversold, which can signal a potential upward reversal."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
            elif key == "cmf":
                col_name = next((c for c in self.data.columns if c.lower().startswith("cmf")), None)
                if col_name and col_name in self.data.columns:
                    cmf_val = self.data[col_name].iloc[-1]
                    values["CMF (20)"] = f"{cmf_val:.3f}"
                    if cmf_val > 0.1:
                        values["Signal"] = "Bullish (strong buying pressure)"
                        confluence_signals.append("bullish")
                        explanation = "CMF above 0.1 indicates strong buying pressure and accumulation."
                    elif cmf_val < -0.1:
                        values["Signal"] = "Bearish (strong selling pressure)"
                        confluence_signals.append("bearish")
                        explanation = "CMF below -0.1 indicates strong selling pressure and distribution."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
            elif key == "vwap":
                vwap_col = next((c for c in self.data.columns if c.upper().startswith("VWAP")), None)
                if vwap_col and vwap_col in self.data.columns:
                    vwap_val = self.data[vwap_col].iloc[-1]
                    price = self.data["close"].iloc[-1]
                    values["VWAP"] = f"{vwap_val:.2f}"
                    if price > vwap_val:
                        values["Signal"] = "Bullish (price above VWAP)"
                        confluence_signals.append("bullish")
                        explanation = "Price above VWAP suggests bullish sentiment and institutional accumulation."
                    elif price < vwap_val:
                        values["Signal"] = "Bearish (price below VWAP)"
                        confluence_signals.append("bearish")
                        explanation = "Price below VWAP suggests bearish sentiment and potential distribution."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
            elif key == "heikinashi":
                ha_close_col = next((c for c in self.data.columns if c.lower().startswith("heikinashi_close")), None)
                ha_open_col = next((c for c in self.data.columns if c.lower().startswith("heikinashi_open")), None)
                if ha_close_col and ha_open_col and ha_close_col in self.data.columns and ha_open_col in self.data.columns:
                    ha_close = self.data[ha_close_col].iloc[-1]
                    ha_open = self.data[ha_open_col].iloc[-1]
                    values["Heikin Ashi Close"] = f"{ha_close:.2f}"
                    values["Heikin Ashi Open"] = f"{ha_open:.2f}"
                    if ha_close > ha_open:
                        values["Signal"] = "Bullish Heikin Ashi candle"
                        confluence_signals.append("bullish")
                        explanation = "Latest Heikin Ashi candle is bullish (close > open), suggesting upward momentum."
                    elif ha_close < ha_open:
                        values["Signal"] = "Bearish Heikin Ashi candle"
                        confluence_signals.append("bearish")
                        explanation = "Latest Heikin Ashi candle is bearish (close < open), suggesting downward momentum."
                    else:
                        values["Signal"] = "Neutral Heikin Ashi candle"
                        confluence_signals.append("neutral")
            elif key == "adx":
                adx_col = "ADX_14"
                plus_di_col = "DM+_14"
                minus_di_col = "DM-_14"
                if all(col in self.data.columns for col in [adx_col, plus_di_col, minus_di_col]):
                    adx = self.data[adx_col].iloc[-1]
                    plus_di = self.data[plus_di_col].iloc[-1]
                    minus_di = self.data[minus_di_col].iloc[-1]
                    values["ADX (14)"] = f"{adx:.2f}"
                    values["+DI (14)"] = f"{plus_di:.2f}"
                    values["-DI (14)"] = f"{minus_di:.2f}"
                    if adx > 20:
                        if plus_di > minus_di:
                            values["Signal"] = "Bullish (trend: +DI > -DI, ADX > 20)"
                            confluence_signals.append("bullish")
                            explanation = "+DI above -DI and ADX above 20 suggests a bullish trend."
                        elif minus_di > plus_di:
                            values["Signal"] = "Bearish (trend: -DI > +DI, ADX > 20)"
                            confluence_signals.append("bearish")
                            explanation = "-DI above +DI and ADX above 20 suggests a bearish trend."
                        else:
                            values["Signal"] = "Neutral (DI lines equal, ADX > 20)"
                            confluence_signals.append("neutral")
                            explanation = "DI lines are equal and ADX above 20, trend is unclear."
                    else:
                        values["Signal"] = "Neutral (ADX <= 20)"
                        confluence_signals.append("neutral")
                        explanation = "ADX below or equal to 20, trend is weak or ranging."
            elif key == "kc":
                kcl_col = next((c for c in self.data.columns if c.lower().startswith("kcl_")), None)
                kcm_col = next((c for c in self.data.columns if c.lower().startswith("kcm_")), None)
                kcu_col = next((c for c in self.data.columns if c.lower().startswith("kcu_")), None)
                if kcl_col and kcm_col and kcu_col:
                    lower = self.data[kcl_col].iloc[-1]
                    middle = self.data[kcm_col].iloc[-1]
                    upper = self.data[kcu_col].iloc[-1]
                    close = self.data["close"].iloc[-1]
                    values["KC Lower"] = f"{lower:.2f}"
                    values["KC Middle"] = f"{middle:.2f}"
                    values["KC Upper"] = f"{upper:.2f}"
                    if close > upper:
                        values["Signal"] = "Price above Upper KC (bullish breakout)"
                        confluence_signals.append("bullish")
                        explanation = "Price above the upper Keltner Channel can indicate a bullish breakout."
                    elif close < lower:
                        values["Signal"] = "Price below Lower KC (bearish breakdown)"
                        confluence_signals.append("bearish")
                        explanation = "Price below the lower Keltner Channel can indicate a bearish breakdown."
                    else:
                        values["Signal"] = "Within Keltner Channels"
                        confluence_signals.append("neutral")
                        explanation = "Price within Keltner Channels suggests no strong directional signal."
            elif key == "stoch":
                k_col = next((c for c in self.data.columns if c.lower().startswith("stochk_")), None)
                d_col = next((c for c in self.data.columns if c.lower().startswith("stochd_")), None)
                if k_col and d_col and k_col in self.data.columns and d_col in self.data.columns:
                    k_val = self.data[k_col].iloc[-1]
                    d_val = self.data[d_col].iloc[-1]
                    values["%K (14)"] = f"{k_val:.2f}"
                    values["%D (3)"] = f"{d_val:.2f}"
                    if k_val > 80:
                        values["Signal"] = "Overbought"
                        confluence_signals.append("bearish")
                        explanation = "Stochastic %K above 80 is considered overbought, which can signal a potential reversal or pullback."
                    elif k_val < 20:
                        values["Signal"] = "Oversold"
                        confluence_signals.append("bullish")
                        explanation = "Stochastic %K below 20 is considered oversold, which can signal a potential upward reversal."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
                        explanation = "Stochastic is in a neutral range."
            elif key == "cci":
                cci_col = next((c for c in self.data.columns if c.lower().startswith("cci_")), None)
                if cci_col and cci_col in self.data.columns:
                    cci_val = self.data[cci_col].iloc[-1]
                    values["CCI (20)"] = f"{cci_val:.2f}"
                    if cci_val > 100:
                        values["Signal"] = "Overbought"
                        confluence_signals.append("bearish")
                        explanation = "CCI above +100 is considered overbought, which can signal a potential reversal or pullback."
                    elif cci_val < -100:
                        values["Signal"] = "Oversold"
                        confluence_signals.append("bullish")
                        explanation = "CCI below -100 is considered oversold, which can signal a potential upward reversal."
                    else:
                        values["Signal"] = "Neutral"
                        confluence_signals.append("neutral")
            if values:
                indicator_data["values"] = values
                indicator_data["explanation"] = explanation
                # Always include the correct explanation_key for CLI/HTML output
                if "explanation_key" in config:
                    indicator_data["explanation_key"] = config["explanation_key"]
                if "explanation_key" in config:
                    try:
                        from src.cli.commands.analyzer_modules.education import get_indicator_explanation
                        indicator_data["education"] = get_indicator_explanation(
                            config["explanation_key"]
                        )
                    except ImportError:
                        indicator_data["education"] = "Explanation unavailable."
                indicators_summary[key] = indicator_data
        # Confluence summary
        bullish = confluence_signals.count("bullish")
        bearish = confluence_signals.count("bearish")
        neutral = confluence_signals.count("neutral")
        if bullish > bearish and bullish > 1:
            confluence_note = "Multiple indicators align bullishly (confluence)."
        elif bearish > bullish and bearish > 1:
            confluence_note = "Multiple indicators align bearishly (confluence)."
        else:
            confluence_note = "No strong confluence among indicators."
        indicators_summary["overall_indicator_confluence"] = {
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "note": confluence_note
        }
        self.analysis_results["indicators_summary"] = indicators_summary
        return indicators_summary

    def _get_default_indicators(self) -> Dict[str, Any]:
        """
        Returns a default configuration for indicators.
        This helps in summarizing and explaining them later.
        """
        # This should align with indicators added in market_data.add_technical_indicators
        return {
            "sma": {
                "display_name": "Simple Moving Averages",
                "params": {"lengths": [20, 50, 100]},
                "explanation_key": "sma",
            },
            "ema": {
                "display_name": "Exponential Moving Averages",
                "params": {"lengths": [12, 26]},
                "explanation_key": "ema",
            },
            "rsi": {
                "display_name": "Relative Strength Index",
                "params": {"length": 14},
                "explanation_key": "rsi",
            },
            "macd": {
                "display_name": "Moving Average Convergence Divergence",
                "params": {"fast": 12, "slow": 26, "signal": 9},
                "explanation_key": "macd",
            },
            "bbands": {
                "display_name": "Bollinger Bands",
                "params": {"length": 20, "std": 2},
                "explanation_key": "bollinger_bands",
            },
            "atr": {
                "display_name": "Average True Range",
                "params": {"length": 14},
                "explanation_key": "atr",
            },  # Assumes pandas-ta adds ATRP_14 or ATR_14
            # Add other common indicators like OBV, VWAP if they are standardly added and have explanations
            "obv": {
                "display_name": "On-Balance Volume",
                "params": {},
                "explanation_key": "obv",
            },
            "vwap": {
                "display_name": "Volume Weighted Average Price",
                "params": {},
                "explanation_key": "vwap",
            },
            "ichimoku": {
                "display_name": "Ichimoku Cloud",
                "params": {"tenkan": 9, "kijun": 26, "senkou": 52},
                "explanation_key": "ichimoku_cloud",
            },
            "psar": {
                "display_name": "Parabolic SAR",
                "params": {},
                "explanation_key": "psar",
            },
            "willr": {
                "display_name": "Williams %R",
                "params": {"length": 14},
                "explanation_key": "williamsr",
            },
            "cmf": {
                "display_name": "Chaikin Money Flow",
                "params": {"length": 20},
                "explanation_key": "cmf",
            },
            "heikinashi": {
                "display_name": "Heikin Ashi",
                "params": {},
                "explanation_key": "heikinashi",
            },
            "adx": {
                "display_name": "Directional Movement Index (DMI)",
                "params": {"length": 14},
                "explanation_key": "adx",
            },
            "kc": {
                "display_name": "Keltner Channels",
                "params": {"length": 20, "scalar": 2, "mamode": "ema"},
                "explanation_key": "kc",
            },
            "stoch": {
                "display_name": "Stochastic Oscillator",
                "params": {"k": 14, "d": 3},
                "explanation_key": "stoch",
            },
            "cci": {
                "display_name": "Commodity Channel Index",
                "params": {"length": 20},
                "explanation_key": "cci",
            },
        }

    def present_cases(self, num_cases: int = 3) -> Dict[str, Any]:
        """
        Presents a few potential market scenarios or cases based on the analysis, now including RSI and educational notes.
        """
        if self.analysis_results.get("error") or self.data is None or self.data.empty:
            return {
                "error": "Cannot generate market cases due to missing data or prior analysis error."
            }
        trend_summary = self.analysis_results.get(
            "trend_analysis", self._get_trend_summary()
        )
        overall_trend = trend_summary.get("overall_trend", "neutral").lower()
        sr_summary = self.analysis_results.get(
            "support_resistance", self._get_support_resistance_summary()
        )
        support = sr_summary.get("support", [])
        resistance = sr_summary.get("resistance", [])
        current_price_str = self.analysis_results.get("price_snapshot", {}).get(
            "current_price", "N/A"
        )
        indicators_summary = self.analysis_results.get("indicators_summary", self._summarize_indicators())
        rsi_signal = indicators_summary.get("rsi", {}).get("values", {}).get("Signal", "Neutral")
        rsi_explanation = indicators_summary.get("rsi", {}).get("explanation", "")
        # --- Open Interest Integration ---
        open_interest = self.analysis_results.get("open_interest")
        oi_factor = None
        if open_interest and isinstance(open_interest, dict):
            oi_val = open_interest.get("value")
            oi_prev = open_interest.get("prev_value")
            price_val = self.data["close"].iloc[-1] if "close" in self.data.columns else None
            price_prev = self.data["close"].iloc[-2] if "close" in self.data.columns and len(self.data) > 1 else None
            if oi_val is not None and oi_prev is not None and price_val is not None and price_prev is not None:
                oi_delta = oi_val - oi_prev
                price_delta = price_val - price_prev
                if oi_delta > 0 and price_delta > 0:
                    oi_factor = "Rising open interest with rising price confirms bullish scenario."
                elif oi_delta < 0 and price_delta > 0:
                    oi_factor = "Falling open interest with rising price suggests short covering, possible reversal."
                elif oi_delta > 0 and price_delta < 0:
                    oi_factor = "Rising open interest with falling price confirms bearish scenario."
                elif oi_delta < 0 and price_delta < 0:
                    oi_factor = "Falling open interest with falling price suggests long liquidation, possible bottom."
                elif oi_delta == 0:
                    oi_factor = "Flat open interest, conviction lacking."
        cases = {}
        if "bullish" in overall_trend:
            cases["bullish_continuation"] = {
                "scenario": "Bullish Trend Continuation",
                "description": (
                    f"The current {overall_trend} trend may continue. "
                    f"RSI status: {rsi_signal}. {rsi_explanation} "
                    "Look for buying opportunities on dips or breakouts above "
                    "near-term resistance."
                ),
                "confidence": trend_summary.get("confidence", "medium"),
                "key_levels": {
                    "resistance_targets": resistance,
                    "support_stops": support,
                },
                "potential_triggers": [
                    "Break above nearest resistance",
                    "Positive volume confirmation",
                ],
                "educational_note": "A bullish trend with RSI not overbought suggests more room for upside. If RSI is overbought, caution is warranted as a pullback may occur."
            }
        elif "bearish" in overall_trend:
            cases["bearish_continuation"] = {
                "scenario": "Bearish Trend Continuation",
                "description": (
                    f"The current {overall_trend} trend may persist. "
                    f"RSI status: {rsi_signal}. {rsi_explanation} "
                    "Consider shorting opportunities on rallies or breakdowns "
                    "below near-term support."
                ),
                "confidence": trend_summary.get("confidence", "medium"),
                "key_levels": {
                    "support_targets": support,
                    "resistance_stops": resistance,
                },
                "potential_triggers": [
                    "Break below nearest support",
                    "Negative volume confirmation",
                ],
                "educational_note": "A bearish trend with RSI not oversold suggests further downside is possible. If RSI is oversold, a bounce or reversal may be near."
            }
        else:
            cases["ranging_market"] = {
                "scenario": "Ranging or Indecisive Market",
                "description": (
                    "The market appears to be in a consolidation or neutral "
                    f"phase. RSI status: {rsi_signal}. {rsi_explanation} "
                    "Trade ranges if clearly defined, or wait for a "
                    "clear breakout. Support and resistance levels are key "
                    "to watch."
                ),
                "confidence": trend_summary.get("confidence", "low"),
                "key_levels": {"support": support, "resistance": resistance},
                "potential_triggers": [
                    "Breakout from range",
                    "Strong volume spike indicating new interest",
                ],
                "educational_note": "A neutral RSI in a ranging market suggests indecision. Watch for RSI to move out of neutral to signal a new trend."
            }
        if (
            trend_summary.get("confidence", "high") != "high"
            and overall_trend != "neutral"
        ):
            if "bullish" in overall_trend:
                cases["bearish_reversal"] = {
                    "scenario": "Potential Bearish Reversal",
                    "description": (
                        "If the current bullish momentum fades, a bearish "
                        "reversal could occur, especially if price fails at "
                        "key resistance or breaks below support. "
                        f"RSI status: {rsi_signal}. {rsi_explanation} "
                    ),
                    "confidence": "low",
                    "key_levels": {
                        "resistance_to_watch": resistance,
                        "support_breakdown": support,
                    },
                    "potential_triggers": [
                        "Failure at resistance",
                        "Breakdown of key support with volume",
                    ],
                    "educational_note": "If RSI is overbought during a bullish trend, the risk of reversal increases."
                }
            elif "bearish" in overall_trend:
                cases["bullish_reversal"] = {
                    "scenario": "Potential Bullish Reversal",
                    "description": (
                        "If the current bearish pressure eases, a bullish "
                        "reversal might be possible, particularly if price "
                        "reclaims key support or breaks resistance. "
                        f"RSI status: {rsi_signal}. {rsi_explanation} "
                    ),
                    "confidence": "low",
                    "key_levels": {
                        "support_to_hold": support,
                        "resistance_breakout": resistance,
                    },
                    "potential_triggers": [
                        "Strong bounce from support",
                        "Breakout above key resistance with volume",
                    ],
                    "educational_note": "If RSI is oversold during a bearish trend, the risk of reversal increases."
                }
        final_cases_list = list(cases.values())[:num_cases]
        # PATCH: Ensure every scenario has a non-empty supporting_factors list
        known_indicators = {
            'RSI', 'MACD', 'SMA', 'EMA', 'BBANDS', 'ATR', 'OBV', 'VWAP', 'ICHIMOKU',
            'PSAR', 'WILLR', 'CMF', 'STOCH', 'KC', 'CCI', 'ADX'
        }
        for case in final_cases_list:
            # Determine scenario direction
            scenario_name = case.get('scenario', '').lower()
            if 'bullish' in scenario_name:
                direction = 'bullish'
            elif 'bearish' in scenario_name:
                direction = 'bearish'
            elif 'neutral' in scenario_name or 'ranging' in scenario_name or 'sideways' in scenario_name:
                direction = 'neutral'
            else:
                direction = None

            # Build supporting_factors from indicator summary
            supporting_factors = []
            for ind_key, ind_data in indicators_summary.items():
                if not isinstance(ind_data, dict):
                    continue
                # Try to find the signal for this indicator
                values = ind_data.get('values', {})
                # Heuristic: look for a 'Signal' or 'Signal ...' key
                signal = None
                for k, v in values.items():
                    if k.lower().startswith('signal'):
                        signal = v.lower()
                        break
                if not signal:
                    continue
                # Match signal to scenario direction
                if direction == 'bullish' and 'bullish' in signal:
                    rationale = ind_data.get('explanation', 'Bullish signal from indicator.')
                    supporting_factors.append(f"{ind_key.upper()} rationale: {rationale}")
                elif direction == 'bearish' and 'bearish' in signal:
                    rationale = ind_data.get('explanation', 'Bearish signal from indicator.')
                    supporting_factors.append(f"{ind_key.upper()} rationale: {rationale}")
                elif direction == 'neutral' and ('neutral' in signal or 'mixed' in signal):
                    rationale = ind_data.get('explanation', 'Neutral/mixed signal from indicator.')
                    supporting_factors.append(f"{ind_key.upper()} rationale: {rationale}")
            # --- Add open interest factor if available ---
            if oi_factor:
                oi_factor_lower = oi_factor.lower()
                oi_val = open_interest.get("value") if open_interest else None
                oi_prev = open_interest.get("prev_value") if open_interest else None
                oi_value_str = f" (Current OI: {oi_val}, Previous OI: {oi_prev})" if oi_val is not None and oi_prev is not None else ""
                if direction == 'bullish' and 'bullish' in oi_factor_lower:
                    supporting_factors.append(f"Open Interest rationale: {oi_factor}{oi_value_str}")
                elif direction == 'bearish' and 'bearish' in oi_factor_lower:
                    supporting_factors.append(f"Open Interest rationale: {oi_factor}{oi_value_str}")
                elif direction == 'neutral' and ('neutral' in oi_factor_lower or 'conviction lacking' in oi_factor_lower):
                    supporting_factors.append(f"Open Interest rationale: {oi_factor}{oi_value_str}")
            # If none found, fall back to previous logic
            if not supporting_factors:
                rationale = case.get('educational_note') or case.get('description') or 'Scenario rationale not specified.'
                import re
                indicators_found = set()
                for indicator in known_indicators:
                    for text in [case.get('educational_note', ''), case.get('description', '')]:
                        if re.search(rf'\\b{indicator}\\b', text, re.IGNORECASE):
                            indicators_found.add(indicator)
                if indicators_found:
                    supporting_factors = [f"{ind} rationale: {rationale}" for ind in sorted(indicators_found)]
                else:
                    if 'rsi' in rationale.lower():
                        supporting_factors = [f"RSI rationale: {rationale}"]
                    else:
                        supporting_factors = [rationale]
            case['supporting_factors'] = supporting_factors
        self.analysis_results["market_cases"] = final_cases_list
        return {"cases": final_cases_list}

    def analyze(self, include_advanced: bool = False) -> Dict[str, Any]:
        """
        Run the full analysis pipeline and return a consolidated results dictionary.
        """
        logger.info(
            f"Starting full analysis pipeline for {self.symbol} (Advanced: "
            f"{include_advanced})..."
        )
        self.run_analysis()
        if self.analysis_results.get("error"):
            logger.error(
                f"Exiting full analysis early for {self.symbol} due to error: "
                f"{self.analysis_results['error']}"
            )
            return {"error": self.analysis_results["error"]}
        self.get_summary()
        self.present_cases()
        self.generate_visualizations()
        advanced_analytics_results = {}
        if include_advanced:
            if AdvancedMarketAnalyzer is not None and self.data is not None and not self.data.empty:
                try:
                    adv_analyzer = AdvancedMarketAnalyzer(
                        self.data, self.symbol, self.timeframe
                    )
                    advanced_analytics_results = (
                        adv_analyzer.run_all_advanced_analytics()
                    )
                    self.analysis_results["advanced_analytics"] = (
                        advanced_analytics_results
                    )
                except Exception as e:
                    logger.error(
                        f"Error running advanced analytics for {self.symbol}: {e}",
                        exc_info=True,
                    )
                    self.analysis_results["advanced_analytics"] = {"error": str(e)}
            elif AdvancedMarketAnalyzer is None:
                self.analysis_results["advanced_analytics"] = {
                    "status": "Advanced analyzer not available (import failed)."
                }
            else:
                self.analysis_results["advanced_analytics"] = {
                    "status": "Data not available for advanced analytics."
                }
        final_display_results = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "generated_at": datetime.now().isoformat(),
            # Ensure 'summary' is a dict with 'general_overview' key for CLI output compatibility
            "summary": {"general_overview": self.analysis_results.get("general_overview", "Summary not generated.")},
            "price_snapshot": self.analysis_results.get("price_snapshot", {}),
            "trend_analysis": self.analysis_results.get("trend_analysis", {}),
            "volatility_analysis": self.analysis_results.get("volatility_analysis", {}),
            "support_resistance": self.analysis_results.get("support_resistance", {}),
            "volume_analysis": self.analysis_results.get("volume_analysis", {}),
            "key_indicators": self.analysis_results.get("key_indicators", {}),
            "market_cases": self.analysis_results.get("market_cases", []),
            "advanced_analytics": self.analysis_results.get("advanced_analytics", {}),
            "visualizations": self.visualizations,
        }
        logger.info(f"Full analysis pipeline completed for {self.symbol}.")
        return final_display_results

    def _detect_candlestick_patterns(self) -> list:
        """
        Detect common candlestick patterns using TA-Lib and return a list of detected patterns.
        Each pattern is a dict with 'name', 'date', and 'signal' (bullish/bearish/neutral).
        """
        if self.data is None or self.data.empty:
            return []
        patterns = {
            'Doji': talib.CDLDOJI,
            'Engulfing': talib.CDLENGULFING,
            'Hammer': talib.CDLHAMMER,
            'Harami': talib.CDLHARAMI,
            'Morning Star': lambda o, h, l, c: talib.CDLMORNINGSTAR(o, h, l, c, penetration=0.3),
            'Shooting Star': talib.CDLSHOOTINGSTAR,
            'Hanging Man': talib.CDLHANGINGMAN,
            'Three White Soldiers': talib.CDL3WHITESOLDIERS,
            'Three Black Crows': talib.CDL3BLACKCROWS,
            'Dark Cloud Cover': talib.CDLDARKCLOUDCOVER,
            'Piercing Line': talib.CDLPIERCING,
            'Evening Star': lambda o, h, l, c: talib.CDLEVENINGSTAR(o, h, l, c, penetration=0.3),
            'Spinning Top': talib.CDLSPINNINGTOP,
            'Marubozu': talib.CDLMARUBOZU,
        }
        detected = []
        o = self.data['open'].astype(float).values
        h = self.data['high'].astype(float).values
        l = self.data['low'].astype(float).values
        c = self.data['close'].astype(float).values
        idx = self.data.index
        for name, func in patterns.items():
            try:
                result = func(o, h, l, c)
                for i, val in enumerate(result):
                    if val != 0:
                        signal = 'bullish' if val > 0 else 'bearish' if val < 0 else 'neutral'
                        detected.append({
                            'name': name,
                            'date': str(idx[i]),
                            'signal': signal
                        })
            except Exception as e:
                continue
        return detected

if __name__ == '__main__':
    from src.config.logging_config import setup_logging, set_log_level
    setup_logging()
    set_log_level("INFO")
    logger.info("Starting MarketAnalyzer example usage...")
