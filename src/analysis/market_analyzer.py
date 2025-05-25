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

from src.services.pattern_recognition import PatternRecognitionService
from src.analysis.indicator_calculations import summarize_indicators, get_default_indicators
from src.analysis.scenario_analysis import present_cases
from src.analysis.visualization import generate_visualizations_for_market

# Set up logging
logger = logging.getLogger(__name__)


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
        from src.analysis.data_fetching import fetch_historical_data
        
        self.data = fetch_historical_data(
            symbol=self.symbol,
            timeframe=self.timeframe,
            use_test_data=self.use_test_data
        )
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
        from src.analysis.data_fetching import fetch_open_interest_data
        self.analysis_results["open_interest"] = fetch_open_interest_data(self.symbol, exchange="okx")

        # Fetch funding rate and store in analysis_results
        from src.analysis.data_fetching import fetch_funding_rate_data
        self.analysis_results["funding_rate"] = fetch_funding_rate_data(self.symbol)
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
        self.visualizations = generate_visualizations_for_market(self.data, self.symbol, self.analysis_results)

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
        indicators_summary = summarize_indicators(self.data, self.performance)
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
        self.analysis_results["market_cases"] = present_cases(self.data, self.analysis_results, self.performance, 3)["cases"]
        self.generate_visualizations()
        advanced_analytics_results = {}
        # --- Pattern Recognition Integration ---
        pattern_recognition_results = {}
        if self.data is not None and not self.data.empty:
            try:
                pattern_service = PatternRecognitionService(self.data)
                harmonic_patterns = pattern_service.find_harmonic_patterns()
                elliott_waves = pattern_service.analyze_elliott_waves()
                pattern_recognition_results = {
                    "harmonic_patterns": [
                        {
                            "name": p.name,
                            "points": p.points,
                            "probability": p.probability,
                            "educational_notes": p.educational_notes
                        } for p in harmonic_patterns
                    ],
                    "elliott_wave_analysis": elliott_waves
                }
            except Exception as e:
                logger.error(f"Pattern recognition failed: {e}", exc_info=True)
                pattern_recognition_results = {"error": str(e)}
        self.analysis_results["pattern_recognition"] = pattern_recognition_results
        # --- End Pattern Recognition Integration ---
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
            "pattern_recognition": self.analysis_results.get("pattern_recognition", {}),
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
