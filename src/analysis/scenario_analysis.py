import logging
from typing import Dict, Any, List
import pandas as pd

def present_cases(
    data: pd.DataFrame,
    analysis_results: Dict[str, Any],
    performance: dict = None,
    num_cases: int = 3
) -> Dict[str, Any]:
    """
    Presents a few potential market scenarios or cases based on the analysis, now including RSI and educational notes.
    This is a refactored version of MarketAnalyzer.present_cases.
    """
    if analysis_results.get("error") or data is None or data.empty:
        return {
            "error": "Cannot generate market cases due to missing data or prior analysis error."
        }
    trend_summary = analysis_results.get(
        "trend_analysis", None
    )
    sr_summary = analysis_results.get(
        "support_resistance", None
    )
    indicators_summary = analysis_results.get("key_indicators", None)
    if trend_summary is None or sr_summary is None or indicators_summary is None:
        return {"error": "Required analysis results missing for scenario analysis."}
    overall_trend = trend_summary.get("overall_trend", "neutral").lower()
    support = sr_summary.get("support", [])
    resistance = sr_summary.get("resistance", [])
    rsi_signal = indicators_summary.get("rsi", {}).get("values", {}).get("Signal", "Neutral")
    rsi_explanation = indicators_summary.get("rsi", {}).get("explanation", "")
    open_interest = analysis_results.get("open_interest")
    oi_factor = None
    if open_interest and isinstance(open_interest, dict):
        oi_val = open_interest.get("value")
        oi_prev = open_interest.get("prev_value")
        price_val = data["close"].iloc[-1] if "close" in data.columns else None
        price_prev = data["close"].iloc[-2] if "close" in data.columns and len(data) > 1 else None
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
    funding_rate = analysis_results.get("funding_rate")
    fr_factor = None
    fr_val = None
    if funding_rate is None or (isinstance(funding_rate, dict) and (funding_rate.get("funding_rate") is None or "error" in funding_rate)):
        error_msg = funding_rate.get("error") if isinstance(funding_rate, dict) and "error" in funding_rate else None
        fr_factor = f"Funding rate data unavailable or fetch failed.{' Error: ' + error_msg if error_msg else ''}"
    elif isinstance(funding_rate, dict):
        fr_val = funding_rate.get("funding_rate")
        if fr_val is not None:
            try:
                fr_val_float = float(fr_val)
            except Exception:
                fr_val_float = None
            if fr_val_float is not None and abs(fr_val_float) > 0.0005:
                if fr_val_float > 0:
                    fr_factor = f"High positive funding rate ({fr_val_float:.4%}) may indicate crowded longs and risk of mean reversion."
                else:
                    fr_factor = f"High negative funding rate ({fr_val_float:.4%}) may indicate crowded shorts and risk of short squeeze."
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
        # --- Add funding rate factor if available ---
        if fr_factor:
            supporting_factors.append(f"Funding Rate rationale: {fr_factor}")
        # If none found, fall back to previous logic
        if not supporting_factors:
            rationale = case.get('educational_note') or case.get('description') or 'Scenario rationale not specified.'
            import re
            indicators_found = set()
            for indicator in known_indicators:
                for text in [case.get('educational_note', ''), case.get('description', '')]:
                    if re.search(rf'\b{indicator}\b', text, re.IGNORECASE):
                        indicators_found.add(indicator)
            if indicators_found:
                supporting_factors = [f"{ind} rationale: {rationale}" for ind in sorted(indicators_found)]
            else:
                if 'rsi' in rationale.lower():
                    supporting_factors = [f"RSI rationale: {rationale}"]
                else:
                    supporting_factors = [rationale]
        case['supporting_factors'] = supporting_factors
    return {"cases": final_cases_list} 