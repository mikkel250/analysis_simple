import logging
from typing import Dict, Any
import pandas as pd
import talib

# Helper function to convert confidence levels to numeric values for comparison
def confidence_level(confidence: str) -> int:
    """Convert confidence string to numeric value for comparison."""
    levels = {"high": 3, "medium": 2, "low": 1}
    return levels.get(confidence.lower(), 0)


def get_default_indicators() -> Dict[str, Any]:
    """
    Returns a default configuration for indicators.
    This helps in summarizing and explaining them later.
    """
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
        },
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


def summarize_indicators(data: pd.DataFrame, performance: dict = None) -> Dict[str, Any]:
    """
    Create a summary of key technical indicators and their signals, with confluence and educational explanations.
    This is a refactored version of MarketAnalyzer._summarize_indicators.
    """
    if data is None or data.empty:
        return {"error": "Data not available for indicator summary."}

    indicators_summary = {}
    default_indicators_config = get_default_indicators()
    confluence_signals = []
    for key, config in default_indicators_config.items():
        indicator_data = {"name": config.get("display_name", key.upper())}
        values = {}
        explanation = ""
        # --- Begin indicator logic (from MarketAnalyzer._summarize_indicators) ---
        if key == "sma" and "lengths" in config.get("params", {}):
            for length in config["params"]["lengths"]:
                col_name = f"SMA_{length}"
                if col_name in data.columns:
                    ma_val = data[col_name].iloc[-1]
                    price = data["close"].iloc[-1]
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
                if col_name in data.columns:
                    ma_val = data[col_name].iloc[-1]
                    price = data["close"].iloc[-1]
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
            if col_name in data.columns:
                rsi_val = data[col_name].iloc[-1]
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
                    for c in data.columns
                    if c.upper().startswith("MACD_")
                    and not c.upper().endswith("_H")
                    and not c.upper().endswith("_S")
                ),
                "MACD_12_26_9",
            )
            macd_signal_col = next(
                (c for c in data.columns if c.upper().startswith("MACDS_")),
                "MACDS_12_26_9",
            )
            macd_hist_col = next(
                (c for c in data.columns if c.upper().startswith("MACDH_")),
                "MACDH_12_26_9",
            )
            if (
                macd_line_col in data.columns
                and macd_signal_col in data.columns
                and macd_hist_col in data.columns
            ):
                macd_line = data[macd_line_col].iloc[-1]
                macd_signal = data[macd_signal_col].iloc[-1]
                macd_hist = data[macd_hist_col].iloc[-1]
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
                (c for c in data.columns if c.upper().startswith("BBL_")),
                "BBL_20_2.0",
            )
            bbm_col = next(
                (c for c in data.columns if c.upper().startswith("BBM_")),
                "BBM_20_2.0",
            )
            bbu_col = next(
                (c for c in data.columns if c.upper().startswith("BBU_")),
                "BBU_20_2.0",
            )
            if (
                bbl_col in data.columns
                and bbm_col in data.columns
                and bbu_col in data.columns
            ):
                lower = data[bbl_col].iloc[-1]
                upper = data[bbu_col].iloc[-1]
                close = data["close"].iloc[-1]
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
            # Robustly detect any ATR column variant
            atr_col = next(
                (
                    c
                    for c in data.columns
                    if c.upper().startswith("ATR")
                ),
                None,
            )
            if atr_col and atr_col in data.columns:
                atr_val = data[atr_col].iloc[-1]
                if pd.isna(atr_val):
                    values["ATR"] = "N/A"
                    explanation = (
                        "ATR value is not available for the latest period. This may be due to insufficient data, all prices being constant, or a calculation issue. "
                        "ATR requires a minimum number of periods (default 14) with valid high, low, and close prices."
                    )
                else:
                    unit = "%" if "P" in atr_col.upper() else ""
                    values[f"ATR ({'percent' if unit else 'value'})"] = (
                        f"{atr_val:.2f}{unit}"
                    )
                    explanation = f"ATR measures average volatility. Higher ATR means more price movement. (Column: {atr_col})"
            else:
                values["ATR"] = "N/A"
                explanation = (
                    "ATR column not found in the data. This may be due to missing or invalid price data, or a failure in indicator calculation. "
                    "Ensure the data includes enough valid high, low, and close prices."
                )
        elif key == "psar":
            psar_col = next((c for c in data.columns if c.lower().startswith("psar")), None)
            if psar_col and psar_col in data.columns:
                psar_val = data[psar_col].iloc[-1]
                price = data["close"].iloc[-1]
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
            if col_name in data.columns:
                wr_val = data[col_name].iloc[-1]
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
            col_name = next((c for c in data.columns if c.lower().startswith("cmf")), None)
            if col_name and col_name in data.columns:
                cmf_val = data[col_name].iloc[-1]
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
            vwap_col = next((c for c in data.columns if c.upper().startswith("VWAP")), None)
            if vwap_col and vwap_col in data.columns:
                vwap_val = data[vwap_col].iloc[-1]
                price = data["close"].iloc[-1]
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
            ha_close_col = next((c for c in data.columns if c.lower().startswith("heikinashi_close")), None)
            ha_open_col = next((c for c in data.columns if c.lower().startswith("heikinashi_open")), None)
            if ha_close_col and ha_open_col and ha_close_col in data.columns and ha_open_col in data.columns:
                ha_close = data[ha_close_col].iloc[-1]
                ha_open = data[ha_open_col].iloc[-1]
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
        elif key == "ichimoku":
            # Ichimoku Cloud components
            tenkan_col = "ITS_9"  # Tenkan-sen (Conversion Line)
            kijun_col = "IKS_26"  # Kijun-sen (Base Line)
            senkou_a_col = "ISA_9"  # Senkou Span A (Leading Span A)
            senkou_b_col = "ISB_26"  # Senkou Span B (Leading Span B)
            chikou_col = "ICS_26"  # Chikou Span (Lagging Span)
            
            if all(col in data.columns for col in [tenkan_col, kijun_col, senkou_a_col, senkou_b_col, chikou_col]):
                tenkan = data[tenkan_col].iloc[-1]
                kijun = data[kijun_col].iloc[-1]
                senkou_a = data[senkou_a_col].iloc[-1]
                senkou_b = data[senkou_b_col].iloc[-1]
                chikou = data[chikou_col].iloc[-1]
                current_price = data["close"].iloc[-1]
                
                values["Tenkan-sen (Conversion)"] = f"{tenkan:.2f}"
                values["Kijun-sen (Base)"] = f"{kijun:.2f}"
                values["Senkou Span A (Leading A)"] = f"{senkou_a:.2f}"
                values["Senkou Span B (Leading B)"] = f"{senkou_b:.2f}"
                values["Chikou Span (Lagging)"] = f"{chikou:.2f}"
                
                # Determine signal based on price position relative to cloud and line positions
                cloud_top = max(senkou_a, senkou_b)
                cloud_bottom = min(senkou_a, senkou_b)
                
                if current_price > cloud_top:
                    if tenkan > kijun:
                        values["Signal"] = "Bullish"
                        confluence_signals.append("bullish")
                        explanation = "Price above the Ichimoku cloud with Tenkan above Kijun suggests a strong bullish trend."
                    else:
                        values["Signal"] = "Bullish (weakening)"
                        confluence_signals.append("neutral")
                        explanation = "Price above the cloud but Tenkan below Kijun suggests weakening bullish momentum."
                elif current_price < cloud_bottom:
                    if tenkan < kijun:
                        values["Signal"] = "Bearish"
                        confluence_signals.append("bearish")
                        explanation = "Price below the Ichimoku cloud with Tenkan below Kijun suggests a strong bearish trend."
                    else:
                        values["Signal"] = "Bearish (weakening)"
                        confluence_signals.append("neutral")
                        explanation = "Price below the cloud but Tenkan above Kijun suggests weakening bearish momentum."
                else:
                    values["Signal"] = "Neutral (in cloud)"
                    confluence_signals.append("neutral")
                    explanation = "Price within the Ichimoku cloud suggests a ranging or consolidating market."
        elif key == "adx":
            adx_col = "ADX_14"
            plus_di_col = "DMP_14"
            minus_di_col = "DMN_14"
            if all(col in data.columns for col in [adx_col, plus_di_col, minus_di_col]):
                adx = data[adx_col].iloc[-1]
                plus_di = data[plus_di_col].iloc[-1]
                minus_di = data[minus_di_col].iloc[-1]
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
            kcl_col = next((c for c in data.columns if c.upper().startswith("KCLE_")), None)
            kcm_col = next((c for c in data.columns if c.upper().startswith("KCBE_")), None)
            kcu_col = next((c for c in data.columns if c.upper().startswith("KCUE_")), None)
            if kcl_col and kcm_col and kcu_col:
                lower = data[kcl_col].iloc[-1]
                middle = data[kcm_col].iloc[-1]
                upper = data[kcu_col].iloc[-1]
                close = data["close"].iloc[-1]
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
            else:
                values["KC Lower"] = values["KC Middle"] = values["KC Upper"] = "N/A"
                values["Signal"] = "Keltner Channel data not available. This may be due to missing data or a calculation error."
                explanation = "Keltner Channel columns were not found in the data. This may be due to missing OHLC data, insufficient data length, or a calculation issue with pandas_ta. Ensure your data includes open, high, low, and close columns and is of sufficient length."
        elif key == "stoch":
            k_col = next((c for c in data.columns if c.lower().startswith("stochk_")), None)
            d_col = next((c for c in data.columns if c.lower().startswith("stochd_")), None)
            if k_col and d_col and k_col in data.columns and d_col in data.columns:
                k_val = data[k_col].iloc[-1]
                d_val = data[d_col].iloc[-1]
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
            cci_col = next((c for c in data.columns if c.lower().startswith("cci_")), None)
            if cci_col and cci_col in data.columns:
                cci_val = data[cci_col].iloc[-1]
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
        elif key == "obv":
            obv_col = next((c for c in data.columns if c.lower() == "obv"), None)
            if obv_col and obv_col in data.columns:
                obv_val = data[obv_col].iloc[-1]
                if pd.isna(obv_val):
                    values["OBV"] = "N/A"
                    explanation = (
                        "OBV value is not available for the latest period. This may be due to insufficient data, missing volume, or a calculation issue. "
                        "OBV requires both price and volume data."
                    )
                else:
                    values["OBV"] = f"{obv_val:.2f}"
                    explanation = "On-Balance Volume (OBV) adds volume on up days and subtracts on down days. Rising OBV suggests positive volume pressure (buying); falling OBV suggests negative volume pressure (selling). Divergences between OBV and price may signal potential reversals."
            else:
                values["OBV"] = "N/A"
                explanation = (
                    "OBV column not found in the data. This may be due to missing or invalid price/volume data, or a failure in indicator calculation. "
                    "Ensure the data includes both valid close and volume columns."
                )
        if values:
            indicator_data["values"] = values
            indicator_data["explanation"] = explanation
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
    return indicators_summary 