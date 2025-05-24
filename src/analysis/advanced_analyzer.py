"""
Advanced market analytics module.

This module provides functions for performing advanced market analytics
including correlation analysis, volatility analysis, and forecasting.
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

# Set up logger
logger = logging.getLogger(__name__)


def _safe_std(series: pd.Series, min_periods: int = 1) -> float:
    """
    Safely calculate standard deviation, 
    returning 0 if series is too short or all NaNs.
    """
    if series.count() < min_periods:
        return 0.0
    std_val = series.std()
    return std_val if pd.notna(std_val) else 0.0


class AdvancedMarketAnalyzer:
    """
    Provides advanced market analysis functionalities.
    These are typically more computationally intensive or experimental.
    """

    def __init__(
        self,
        data_with_indicators: pd.DataFrame,
        symbol: str,
        timeframe: str
    ):
        """
        Initialize the AdvancedMarketAnalyzer.

        Args:
            data_with_indicators: DataFrame with OHLCV data and
                                  pre-calculated basic indicators.
            symbol: The trading symbol (e.g., BTC/USDT).
            timeframe: The timeframe string (e.g., '1h').
        """
        if data_with_indicators is None or data_with_indicators.empty:
            logger.error(
                "AdvancedMarketAnalyzer received None or empty DataFrame."
            )
            raise ValueError(
                "Data for advanced analysis cannot be None or empty."
            )
        self.data = data_with_indicators.copy()
        self.symbol = symbol
        self.timeframe = timeframe
        logger.info(
            f"AdvancedMarketAnalyzer initialized for {self.symbol} "
            f"({self.timeframe})"
        )

    def _calculate_rsi_divergence(self) -> Dict[str, Any]:
        """Calculate RSI divergence (simplified check)."""
        if ('RSI_14' not in self.data.columns or
                'close' not in self.data.columns):
            return {
                "rsi_divergence_signal": "unavailable",
                "rsi_divergence_details": "RSI_14 or close price missing"
            }

        n_periods = 14
        if len(self.data) < n_periods + 1:
            return {
                "rsi_divergence_signal": "unavailable",
                "rsi_divergence_details": (
                    f"Not enough data for {n_periods}-period RSI divergence"
                )
            }

        price_series = self.data['close'].tail(n_periods * 2)
        rsi_series = self.data['RSI_14'].tail(n_periods * 2)
        signal = "none"
        details = (
            f"No clear RSI divergence detected (simplified {n_periods}-period "
            "check)."
        )

        if len(price_series) < n_periods or len(rsi_series) < n_periods:
            return {
                "rsi_divergence_signal": "unavailable",
                "rsi_divergence_details": (
                    f"Not enough data in tail for {n_periods}-period RSI "
                    "divergence"
                )
            }

        # Simplified: Compare current with n_periods ago
        if (price_series.iloc[-1] > price_series.iloc[-n_periods] and
                rsi_series.iloc[-1] < rsi_series.iloc[-n_periods]):
            signal = "bearish_divergence_candidate"
            details = (
                f"Potential bearish RSI divergence: Price higher, RSI lower "
                f"over ~{n_periods} periods (simplified check)."
            )
        elif (price_series.iloc[-1] < price_series.iloc[-n_periods] and
                rsi_series.iloc[-1] > rsi_series.iloc[-n_periods]):
            signal = "bullish_divergence_candidate"
            details = (
                f"Potential bullish RSI divergence: Price lower, RSI higher "
                f"over ~{n_periods} periods (simplified check)."
            )

        return {
            "rsi_divergence_signal": signal,
            "rsi_divergence_details": details
        }

    def _calculate_macd_divergence(self) -> Dict[str, Any]:
        """Calculate MACD divergence (simplified check)."""
        macd_hist_col = next(
            (col for col in self.data.columns if 'MACDH' in col.upper()),
            None
        )
        if not macd_hist_col or 'close' not in self.data.columns:
            return {
                "macd_divergence_signal": "unavailable",
                "macd_divergence_details": (
                    "MACD histogram or close price missing"
                )
            }

        n_periods = 14
        if len(self.data) < n_periods + 1:
            return {
                "macd_divergence_signal": "unavailable",
                "macd_divergence_details": (
                    f"Not enough data for {n_periods}-period MACD divergence"
                )
            }

        price_series = self.data['close'].tail(n_periods * 2)
        macd_hist_series = self.data[macd_hist_col].tail(n_periods * 2)
        signal = "none"
        details = (
            "No clear MACD divergence detected (simplified "
            f"{n_periods}-period check)."
        )

        if len(price_series) < n_periods or len(macd_hist_series) < n_periods:
            return {
                "macd_divergence_signal": "unavailable",
                "macd_divergence_details": (
                    f"Not enough data in tail for {n_periods}-period MACD "
                    "divergence"
                )
            }

        if (price_series.iloc[-1] > price_series.iloc[-n_periods] and
                macd_hist_series.iloc[-1] < macd_hist_series.iloc[-n_periods]):
            signal = "bearish_divergence_candidate"
            details = (
                f"Potential bearish MACD divergence over ~{n_periods} "
                "periods (simplified check)."
            )
        elif (price_series.iloc[-1] < price_series.iloc[-n_periods] and
                macd_hist_series.iloc[-1] > macd_hist_series.iloc[-n_periods]):
            signal = "bullish_divergence_candidate"
            details = (
                f"Potential bullish MACD divergence over ~{n_periods} "
                "periods (simplified check)."
            )

        return {
            "macd_divergence_signal": signal,
            "macd_divergence_details": details
        }

    def _calculate_bollinger_extremes(self) -> Dict[str, Any]:
        """Identify Bollinger Bands extremes."""
        bbl_col = next(
            (col for col in self.data.columns if 'BBL' in col.upper()),
            None
        )
        bbu_col = next(
            (col for col in self.data.columns if 'BBU' in col.upper()),
            None
        )

        if not bbl_col or not bbu_col or 'close' not in self.data.columns:
            return {
                "bollinger_extremes_signal": "unavailable",
                "bollinger_extremes_details": (
                    "Bollinger Bands or close price missing"
                )
            }

        if len(self.data) < 1:
            return {
                "bollinger_extremes_signal": "unavailable",
                "bollinger_extremes_details": (
                    "No data to check Bollinger extremes."
                )
            }

        current_close = self.data['close'].iloc[-1]
        current_bbl = self.data[bbl_col].iloc[-1]
        current_bbu = self.data[bbu_col].iloc[-1]
        signal = "none"
        details = "Price is within Bollinger Bands."

        if (pd.isna(current_close) or pd.isna(current_bbl) or
                pd.isna(current_bbu)):
            return {
                "bollinger_extremes_signal": "unavailable",
                "bollinger_extremes_details": (
                    "NaN values in price or Bollinger Bands."
                )
            }

        if current_close < current_bbl:
            signal = "oversold_potential"
            details = (
                f"Price ({current_close:.2f}) is below the lower Bollinger "
                f"Band ({current_bbl:.2f}), potential oversold condition."
            )
        elif current_close > current_bbu:
            signal = "overbought_potential"
            details = (
                f"Price ({current_close:.2f}) is above the upper Bollinger "
                f"Band ({current_bbu:.2f}), potential overbought condition."
            )

        return {
            "bollinger_extremes_signal": signal,
            "bollinger_extremes_details": details
        }

    def _calculate_volume_analysis_adv(self) -> Dict[str, Any]:
        """Perform basic volume analysis (e.g., volume spikes)."""
        if 'volume' not in self.data.columns or len(self.data) < 1:
            return {
                "volume_signal": "unavailable",
                "volume_details": "Volume data missing or insufficient"
            }

        rolling_window = min(20, len(self.data))
        min_periods_vol = min(5, len(self.data))
        if min_periods_vol == 0 and rolling_window > 0:
            min_periods_vol = 1

        avg_volume = np.nan
        if len(self.data) >= min_periods_vol:
            avg_volume = self.data['volume'].rolling(
                window=rolling_window, min_periods=min_periods_vol
            ).mean().iloc[-1]

        current_volume = self.data['volume'].iloc[-1]
        signal = "normal_volume"
        details = (
            f"Current volume ({current_volume:,.0f}) is around the recent "
            "average."
        )

        if pd.isna(current_volume):
            return {
                "volume_signal": "unavailable",
                "volume_details": "Current volume is NaN."
            }
        if pd.isna(avg_volume) or avg_volume == 0:
            details = (
                f"Current volume is ({current_volume:,.0f}). Average volume "
                f"({rolling_window}-period) calculation unavailable or zero."
            )
        elif current_volume > avg_volume * 2:
            signal = "volume_spike"
            details = (
                f"Significant volume spike: Current volume "
                f"({current_volume:,.0f}) is much higher than the recent "
                f"{rolling_window}-period average ({avg_volume:,.0f})."
            )
        elif current_volume < avg_volume * 0.5:
            signal = "low_volume"
            details = (
                f"Low volume: Current volume ({current_volume:,.0f}) is much "
                f"lower than the recent {rolling_window}-period average "
                f"({avg_volume:,.0f})."
            )

        return {
            "volume_signal": signal,
            "volume_details": details,
            "current_volume": current_volume,
            f"average_volume_{rolling_window}p": avg_volume
        }

    def _calculate_trend_strength(self) -> Dict[str, Any]:
        """Assess trend strength (e.g., using ADX)."""
        adx_col = next(
            (col for col in self.data.columns if 'ADX' in col.upper()),
            None
        )
        if not adx_col or len(self.data) < 1:
            return {
                "trend_strength_signal": "unavailable",
                "trend_strength_details": (
                    f"{adx_col or 'ADX'} missing or no data"
                )
            }

        current_adx = self.data[adx_col].iloc[-1]
        signal = "weak_or_no_trend"
        details = (
            f"ADX ({current_adx:.2f}) suggests a weak or non-trending market."
        )

        if pd.isna(current_adx):
            details = "ADX value is not available."
            signal = "unavailable"
        elif current_adx > 25:
            signal = "strong_trend"
            details = f"ADX ({current_adx:.2f}) suggests a strong trend."
        elif current_adx > 20:
            signal = "developing_trend"
            details = f"ADX ({current_adx:.2f}) suggests a developing trend."

        return {
            "trend_strength_signal": signal,
            "trend_strength_details": details,
            "adx_value": current_adx
        }

    def _calculate_volatility_analysis_adv(self) -> Dict[str, Any]:
        """Analyze market volatility (e.g., using ATR as percentage)."""
        atr_col_name = None
        is_percent_atr = False
        # Try to find ATRP (percentage) first, then ATR (absolute),
        # then any ATR column
        if 'ATRP_14' in self.data.columns:
            atr_col_name = 'ATRP_14'
            is_percent_atr = True
        elif 'ATR_14' in self.data.columns:
            atr_col_name = 'ATR_14'
        else:
            atr_col_name = next(
                (col for col in self.data.columns if 'ATR' in col.upper()),
                None
            )
            if atr_col_name and 'P' in atr_col_name.upper():
                is_percent_atr = True

        if (not atr_col_name or 'close' not in self.data.columns or
                len(self.data) < 1):
            return {
                "atr_volatility_level": "unavailable",
                "atr_volatility_details": (
                    f"{atr_col_name or 'ATR'} or close price missing or no "
                    "data"
                )
            }

        current_atr_val = self.data[atr_col_name].iloc[-1]
        current_close_price = self.data['close'].iloc[-1]
        current_atr_percent = np.nan

        if (pd.notna(current_atr_val) and pd.notna(current_close_price) and
                current_close_price != 0):
            if is_percent_atr:
                current_atr_percent = current_atr_val # Assuming ATRP is already in percent, not fraction
            else:
                current_atr_percent = (current_atr_val / current_close_price) * 100


        rolling_window = min(20, len(self.data))
        min_periods_atr = min(5, len(self.data))
        if min_periods_atr == 0 and rolling_window > 0:
            min_periods_atr = 1

        avg_atr_percent = np.nan
        if len(self.data) >= min_periods_atr:
            if is_percent_atr:
                # ATRP is already a percentage
                avg_atr_percent = self.data[atr_col_name].rolling(
                    window=rolling_window, min_periods=min_periods_atr
                ).mean().iloc[-1]
            else:
                safe_close = self.data['close'].replace(0, np.nan)
                atr_as_percent_series = (
                    (self.data[atr_col_name] / safe_close) * 100
                ).replace([np.inf, -np.inf], np.nan)
                avg_atr_percent = atr_as_percent_series.rolling(
                    window=rolling_window, min_periods=min_periods_atr
                ).mean().iloc[-1]
        
        level = "normal"
        details = (
            f"Current ATR ({current_atr_percent:.2f}%) is around the recent "
            f"{rolling_window}-period average."
        )

        if pd.isna(current_atr_percent):
            level = "unavailable"
            details = "Current ATR percentage is not available."
        elif pd.isna(avg_atr_percent):
            details = (
                f"Current ATR ({current_atr_percent:.2f}%). Average ATR "
                f"({rolling_window}-period) is not available."
            )
        elif avg_atr_percent == 0: # Check if avg_atr_percent could be 0 to avoid division by zero
             details = (
                f"Current ATR ({current_atr_percent:.2f}%). Average ATR "
                f"({rolling_window}-period) is zero."
            )
        elif current_atr_percent > avg_atr_percent * 1.5:
            level = "high"
            details = (
                f"High ATR-based Volatility: Current ATR ({current_atr_percent:.2f}%) "
                f"is significantly above the recent {rolling_window}-period "
                f"average ({avg_atr_percent:.2f}%)."
            )
        elif current_atr_percent < avg_atr_percent * 0.7:
            level = "low"
            details = (
                f"Low ATR-based Volatility: Current ATR ({current_atr_percent:.2f}%) "
                f"is significantly below the recent {rolling_window}-period "
                f"average ({avg_atr_percent:.2f}%)."
            )

        return {
            "atr_volatility_level": level,
            "atr_volatility_details": details,
            "current_atr_percent": (
                current_atr_percent if pd.notna(current_atr_percent) else None
            ),
            f"average_atr_percent_{rolling_window}p": (
                avg_atr_percent if pd.notna(avg_atr_percent) else None
            )
        }

    def _analyze_volume_profile(self) -> Dict[str, Any]:
        logger.debug(
            "Volume Profile Analysis: Not implemented in this version."
        )
        return {
            "status": (
                "Volume Profile Analysis not implemented in this version."
            )
        }

    def _correlation_analysis(
        self, other_asset_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        if other_asset_data is None:
            logger.debug(
                "Correlation Analysis: No other asset data provided."
            )
            return {
                "status": "Correlation Analysis: No other asset data provided."
            }

        if ('close' not in self.data.columns or
                'close' not in other_asset_data.columns):
            logger.debug(
                "Correlation Analysis: Close price missing in one of the "
                "datasets."
            )
            return {"status": "Correlation Analysis: Close price missing."}

        aligned_data = self.data[['close']].join(
            other_asset_data[['close']],
            lsuffix='_main',
            rsuffix='_other',
            how='inner'
        )

        if len(aligned_data) < 20:
            logger.debug(
                "Correlation Analysis: Not enough overlapping data."
            )
            return {
                "status": (
                    "Correlation Analysis: Insufficient overlapping data."
                )
            }

        correlation = aligned_data['close_main'].corr(
            aligned_data['close_other']
        )
        logger.debug(
            f"Correlation Analysis: Calculated correlation = {correlation:.2f}"
        )
        return {
            "correlation_with_other": f"{correlation:.2f}",
            "status": "Correlation calculated (basic)."
        }

    def _seasonality_analysis(self) -> Dict[str, Any]:
        logger.debug("Seasonality Analysis: Not implemented in this version.")
        return {
            "status": "Seasonality Analysis not implemented in this version."
        }

    def _regime_detection(self) -> Dict[str, Any]:
        logger.debug(
            "Market Regime Detection: Not implemented in this version."
        )
        return {
            "status": (
                "Market Regime Detection not implemented in this version."
            )
        }

    def _analyze_risk_reward(self) -> Dict[str, Any]:
        logger.debug("Risk/Reward Analysis: Not implemented in this version.")
        return {
            "status": "Risk/Reward Analysis not implemented in this version."
        }

    def _generate_scenario_analysis(self) -> Dict[str, Any]:
        logger.debug("Scenario Analysis: Not implemented in this version.")
        return {
            "status": "Scenario Analysis not implemented in this version."
        }

    def run_all_advanced_analytics(self) -> Dict[str, Any]:
        if self.data is None or self.data.empty:
            logger.warning(
                f"Cannot run advanced analytics for {self.symbol} due to "
                "missing data."
            )
            return {"error": "Data not available for advanced analytics."}

        logger.info(f"Running all advanced analytics for {self.symbol}...")
        results: Dict[str, Any] = {}

        results['rsi_divergence'] = self._calculate_rsi_divergence()
        results['macd_divergence'] = self._calculate_macd_divergence()
        results['bollinger_extremes'] = self._calculate_bollinger_extremes()
        results['volume_analysis_adv'] = self._calculate_volume_analysis_adv()
        # Call to _calculate_trend_strength() is removed as its functionality
        # will be more deeply integrated into MarketAnalyzer._determine_trend().
        # results['trend_strength'] = self._calculate_trend_strength()
        results['atr_volatility_analysis'] = (
            self._calculate_volatility_analysis_adv() # Key already updated by previous edit if applied
        )

        # Removed calls to not-implemented or currently out-of-scope methods:
        # results['volume_profile'] = self._analyze_volume_profile()
        # results['correlation'] = self._correlation_analysis() # Requires other_asset_data
        # results['seasonality'] = self._seasonality_analysis()
        # results['regime_detection'] = self._regime_detection()
        # results['risk_reward'] = self._analyze_risk_reward()
        # results['scenario_analysis'] = self._generate_scenario_analysis()

        logger.info(f"Finished advanced analytics for {self.symbol}.")
        return results 