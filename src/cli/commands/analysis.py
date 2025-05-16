"""
Analysis Command Handler

Handles the 'analysis' command for generating comprehensive market analysis.
"""

import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime

import typer
import pandas as pd  # type: ignore
from tabulate import tabulate

from src.cli.display import (
    display_error, 
    display_success, 
    display_warning, 
    format_price,
    display_data_age,
    display_spinner
)
from src.services.data_fetcher import get_historical_data, get_current_price
from src.services.indicators import get_indicator
from src.cli.education import (
    get_category_explanation, 
    get_indicator_explanation, 
    get_summary_explanation, 
    category_header
)

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
analysis_app = typer.Typer()


def generate_analysis(
    df: pd.DataFrame,
    current_price_data: Dict[str, Any],
    symbol: str = 'BTC',
    timeframe: str = '1d',
    vs_currency: str = 'usd'
) -> Dict[str, Any]:
    """
    Generate comprehensive market analysis using multiple indicators.
    
    This function performs calculations on the provided historical data and current price
    to generate a structured analysis result that can be used for both CLI display
    and interactive visualization in Jupyter notebooks.
    
    Args:
        df: Historical price DataFrame with columns for OHLCV data
        current_price_data: Current price data dictionary from the API
        symbol: Symbol being analyzed (e.g., "BTC")
        timeframe: Timeframe of the data (e.g., "1d", "4h")
        vs_currency: Currency to calculate prices against (default: "usd")
        
    Returns:
        Dict[str, Any]: Structured analysis results with the following keys:
            - price_data: Current price information
            - trend_indicators: Trend indicator calculations
            - momentum_indicators: Momentum indicator calculations
            - volatility_indicators: Volatility indicator calculations
            - volume_indicators: Volume indicator calculations
            - summary: Market summary and signals
            - timestamp: Analysis generation timestamp
    """
    analysis_results = {
        "price_data": {},
        "trend_indicators": {},
        "momentum_indicators": {},
        "volatility_indicators": {},
        "volume_indicators": {},
        "summary": {},
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "symbol": symbol,
            "timeframe": timeframe,
            "vs_currency": vs_currency
        }
    }
    
    # Extract price data
    try:
        current_price = current_price_data.get("current_price", {}).get(vs_currency, 0)
        price_change_24h = current_price_data.get("price_change_24h", 0)
        price_change_pct_24h = current_price_data.get("price_change_percentage_24h", 0)
        
        analysis_results["price_data"] = {
            "current_price": current_price,
            "price_change_24h": price_change_24h,
            "price_change_percentage_24h": price_change_pct_24h,
            "last_updated": current_price_data.get("last_updated", "")
        }
    except Exception as e:
        logger.error(f"Error processing price data: {str(e)}")
    
    # Calculate trend indicators
    try:
        # SMA
        sma_result = get_indicator(df, "sma", {"length": 20}, symbol, timeframe)
        analysis_results["trend_indicators"]["sma"] = {
            "value": list(sma_result["values"].items())[-1][1] if sma_result["values"] else None,
            "params": sma_result["metadata"]["params"]
        }
        
        # EMA
        ema_result = get_indicator(df, "ema", {"length": 20}, symbol, timeframe)
        analysis_results["trend_indicators"]["ema"] = {
            "value": list(ema_result["values"].items())[-1][1] if ema_result["values"] else None,
            "params": ema_result["metadata"]["params"]
        }
        
        # MACD
        macd_result = get_indicator(df, "macd", {}, symbol, timeframe)
        if macd_result["values"]:
            macd_keys = list(macd_result["values"].keys())
            macd_line_key = next((k for k in macd_keys if k.startswith("MACD_")), None)
            signal_line_key = next((k for k in macd_keys if k.startswith("MACDs_")), None)
            histogram_key = next((k for k in macd_keys if k.startswith("MACDh_")), None)
            
            if macd_line_key and signal_line_key and histogram_key:
                macd_line = list(macd_result["values"][macd_line_key].items())[-1][1]
                signal_line = list(macd_result["values"][signal_line_key].items())[-1][1]
                histogram = list(macd_result["values"][histogram_key].items())[-1][1]
                
                analysis_results["trend_indicators"]["macd"] = {
                    "macd_line": macd_line,
                    "signal_line": signal_line,
                    "histogram": histogram,
                    "params": macd_result["metadata"]["params"]
                }
        
        # ADX
        adx_result = get_indicator(df, "adx", {}, symbol, timeframe)
        if adx_result["values"]:
            adx_keys = list(adx_result["values"].keys())
            adx_key = next((k for k in adx_keys if k.startswith("ADX_")), None)
            
            if adx_key:
                adx_value = list(adx_result["values"][adx_key].items())[-1][1]
                analysis_results["trend_indicators"]["adx"] = {
                    "value": adx_value,
                    "params": adx_result["metadata"]["params"]
                }
    except Exception as e:
        logger.error(f"Error calculating trend indicators: {str(e)}")
    
    # Calculate momentum indicators
    try:
        # RSI
        rsi_result = get_indicator(df, "rsi", {}, symbol, timeframe)
        analysis_results["momentum_indicators"]["rsi"] = {
            "value": list(rsi_result["values"].items())[-1][1] if rsi_result["values"] else None,
            "params": rsi_result["metadata"]["params"]
        }
        
        # Stochastic
        stoch_result = get_indicator(df, "stoch", {}, symbol, timeframe)
        if stoch_result["values"]:
            stoch_keys = list(stoch_result["values"].keys())
            k_line_key = next((k for k in stoch_keys if k.startswith("STOCHk_")), None)
            d_line_key = next((k for k in stoch_keys if k.startswith("STOCHd_")), None)
            
            if k_line_key and d_line_key:
                k_line = list(stoch_result["values"][k_line_key].items())[-1][1]
                d_line = list(stoch_result["values"][d_line_key].items())[-1][1]
                
                analysis_results["momentum_indicators"]["stoch"] = {
                    "k_line": k_line,
                    "d_line": d_line,
                    "params": stoch_result["metadata"]["params"]
                }
        
        # CCI
        cci_result = get_indicator(df, "cci", {}, symbol, timeframe)
        analysis_results["momentum_indicators"]["cci"] = {
            "value": list(cci_result["values"].items())[-1][1] if cci_result["values"] else None,
            "params": cci_result["metadata"]["params"]
        }
    except Exception as e:
        logger.error(f"Error calculating momentum indicators: {str(e)}")
    
    # Calculate volatility indicators
    try:
        # Bollinger Bands
        bbands_result = get_indicator(df, "bbands", {}, symbol, timeframe)
        if bbands_result["values"]:
            bb_keys = list(bbands_result["values"].keys())
            upper_key = next((k for k in bb_keys if k.startswith("BBU_")), None)
            middle_key = next((k for k in bb_keys if k.startswith("BBM_")), None)
            lower_key = next((k for k in bb_keys if k.startswith("BBL_")), None)
            
            if upper_key and middle_key and lower_key:
                upper = list(bbands_result["values"][upper_key].items())[-1][1]
                middle = list(bbands_result["values"][middle_key].items())[-1][1]
                lower = list(bbands_result["values"][lower_key].items())[-1][1]
                
                analysis_results["volatility_indicators"]["bbands"] = {
                    "upper": upper,
                    "middle": middle,
                    "lower": lower,
                    "params": bbands_result["metadata"]["params"]
                }
        
        # ATR
        atr_result = get_indicator(df, "atr", {}, symbol, timeframe)
        analysis_results["volatility_indicators"]["atr"] = {
            "value": list(atr_result["values"].items())[-1][1] if atr_result["values"] else None,
            "params": atr_result["metadata"]["params"]
        }
    except Exception as e:
        logger.error(f"Error calculating volatility indicators: {str(e)}")
    
    # Calculate volume indicators
    try:
        # OBV
        obv_result = get_indicator(df, "obv", {}, symbol, timeframe)
        analysis_results["volume_indicators"]["obv"] = {
            "value": list(obv_result["values"].items())[-1][1] if obv_result["values"] else None,
            "params": obv_result["metadata"]["params"]
        }
    except Exception as e:
        logger.error(f"Error calculating volume indicators: {str(e)}")
    
    # Generate summary
    analysis_results["summary"] = generate_summary(analysis_results, current_price)
    
    return analysis_results


def generate_summary(analysis_results: Dict[str, Any], current_price: float) -> Dict[str, Any]:
    """
    Generate a summary of market analysis.
    
    Args:
        analysis_results: Analysis results
        current_price: Current price
    
    Returns:
        Dict: Summary analysis
    """
    summary = {
        "trend": {
            "direction": "neutral",
            "strength": "neutral",
            "confidence": "medium", 
            "analysis": "Market conditions are neutral."
        },
        "signals": {
            "short_term": "neutral",
            "medium_term": "neutral",
            "long_term": "neutral",
            "action": "hold"
        }
    }
    
    # Extract indicator values
    try:
        # Trend indicators
        sma_20 = analysis_results["trend_indicators"].get("sma", {}).get("value")
        ema_20 = analysis_results["trend_indicators"].get("ema", {}).get("value")
        macd = analysis_results["trend_indicators"].get("macd", {})
        macd_line = macd.get("macd_line")
        signal_line = macd.get("signal_line")
        histogram = macd.get("histogram")
        adx = analysis_results["trend_indicators"].get("adx", {}).get("value")
        
        # Momentum indicators
        rsi = analysis_results["momentum_indicators"].get("rsi", {}).get("value")
        stoch = analysis_results["momentum_indicators"].get("stoch", {})
        stoch_k = stoch.get("k_line")
        stoch_d = stoch.get("d_line")
        cci = analysis_results["momentum_indicators"].get("cci", {}).get("value")
        
        # Volatility indicators
        bbands = analysis_results["volatility_indicators"].get("bbands", {})
        bb_upper = bbands.get("upper")
        bb_middle = bbands.get("middle")
        bb_lower = bbands.get("lower")
        atr = analysis_results["volatility_indicators"].get("atr", {}).get("value")
        
        # Determine trend direction
        trend_signals = []
        if current_price and sma_20:
            if current_price > sma_20:
                trend_signals.append(1)  # Bullish
            else:
                trend_signals.append(-1)  # Bearish
        
        if current_price and ema_20:
            if current_price > ema_20:
                trend_signals.append(1)  # Bullish
            else:
                trend_signals.append(-1)  # Bearish
        
        if macd_line and signal_line:
            if macd_line > signal_line:
                trend_signals.append(1)  # Bullish
            else:
                trend_signals.append(-1)  # Bearish
        
        # Calculate average trend signal
        if trend_signals:
            avg_trend = sum(trend_signals) / len(trend_signals)
            if avg_trend > 0.5:
                summary["trend"]["direction"] = "bullish"
            elif avg_trend < -0.5:
                summary["trend"]["direction"] = "bearish"
            else:
                summary["trend"]["direction"] = "neutral"
            
            # Trend strength based on ADX
            if adx:
                if adx > 25:
                    summary["trend"]["strength"] = "strong"
                elif adx > 15:
                    summary["trend"]["strength"] = "moderate"
                else:
                    summary["trend"]["strength"] = "weak"
                    
        # Determine signals
        signals = {"short": 0, "medium": 0, "long": 0}
        signal_count = 0
        
        # RSI signal
        if rsi:
            signal_count += 1
            if rsi > 70:
                signals["short"] -= 1  # Overbought
            elif rsi < 30:
                signals["short"] += 1  # Oversold
        
        # Stochastic signal
        if stoch_k and stoch_d:
            signal_count += 1
            if stoch_k > 80 and stoch_d > 80:
                signals["short"] -= 1  # Overbought
            elif stoch_k < 20 and stoch_d < 20:
                signals["short"] += 1  # Oversold
            
            if stoch_k > stoch_d:
                signals["medium"] += 0.5  # Bullish crossover
            else:
                signals["medium"] -= 0.5  # Bearish crossover
        
        # MACD signal
        if macd_line and signal_line and histogram:
            signal_count += 1
            if macd_line > signal_line:
                signals["medium"] += 1  # Bullish
            else:
                signals["medium"] -= 1  # Bearish
                
            # Histogram direction
            prev_histogram = 0  # Placeholder, in reality, would need historical data
            if histogram > prev_histogram:
                signals["short"] += 0.5  # Increasing momentum
            else:
                signals["short"] -= 0.5  # Decreasing momentum
        
        # Bollinger Bands signal
        if current_price and bb_upper and bb_lower:
            signal_count += 1
            if current_price > bb_upper:
                signals["short"] -= 1  # Overbought
            elif current_price < bb_lower:
                signals["short"] += 1  # Oversold
        
        # Normalize signals
        if signal_count > 0:
            for period in signals:
                signals[period] /= signal_count
            
            # Set signal directions
            if signals["short"] > 0.3:
                summary["signals"]["short_term"] = "bullish"
            elif signals["short"] < -0.3:
                summary["signals"]["short_term"] = "bearish"
            
            if signals["medium"] > 0.3:
                summary["signals"]["medium_term"] = "bullish"
            elif signals["medium"] < -0.3:
                summary["signals"]["medium_term"] = "bearish"
            
            # Long term based on price vs SMA
            if current_price and sma_20:
                if current_price > sma_20 * 1.05:  # 5% above SMA
                    summary["signals"]["long_term"] = "bullish"
                elif current_price < sma_20 * 0.95:  # 5% below SMA
                    summary["signals"]["long_term"] = "bearish"
            
            # Determine action
            short_score = 1 if summary["signals"]["short_term"] == "bullish" else (-1 if summary["signals"]["short_term"] == "bearish" else 0)
            medium_score = 1 if summary["signals"]["medium_term"] == "bullish" else (-1 if summary["signals"]["medium_term"] == "bearish" else 0)
            long_score = 1 if summary["signals"]["long_term"] == "bullish" else (-1 if summary["signals"]["long_term"] == "bearish" else 0)
            
            # Weight medium-term more
            action_score = short_score * 0.3 + medium_score * 0.5 + long_score * 0.2
            
            if action_score > 0.5:
                summary["signals"]["action"] = "buy"
            elif action_score < -0.5:
                summary["signals"]["action"] = "sell"
            else:
                summary["signals"]["action"] = "hold"
        
        # Generate analysis text
        trend_text = f"The market is showing a {summary['trend']['strength']} {summary['trend']['direction']} trend."
        signal_text = (f"Short-term signals are {summary['signals']['short_term']}, "
                      f"medium-term signals are {summary['signals']['medium_term']}, "
                      f"and long-term outlook is {summary['signals']['long_term']}.")
        action_text = f"The suggested action based on technical analysis is to {summary['signals']['action'].upper()}."
        
        summary["trend"]["analysis"] = f"{trend_text} {signal_text} {action_text}"
    
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        summary["trend"]["analysis"] = "Could not generate complete analysis due to insufficient data."
    
    return summary


def prepare_analysis_data(analysis_results: Dict[str, Any], explain: bool = False) -> List[List[Any]]:
    """
    Prepare analysis results data for display.
    
    Args:
        analysis_results: Analysis results dictionary
        explain: Whether to include educational explanations
        
    Returns:
        List[List[Any]]: Data ready for formatting
    """
    formatted_data = []
    
    # Get metadata
    metadata = analysis_results.get("metadata", {})
    symbol = metadata.get("symbol", "BTC")
    vs_currency = metadata.get("vs_currency", "usd").upper()
    
    # Add price information
    price_data = analysis_results.get("price_data", {})
    formatted_data.append([f"{symbol}/{vs_currency} Price", format_price(price_data.get("current_price", 0))])
    formatted_data.append([
        "24h Change", 
        format_price(price_data.get("price_change_24h", 0)) + 
        f" ({price_data.get('price_change_percentage_24h', 0):.2f}%)"
    ])
    
    # Add separator for trend indicators
    formatted_data.append(["", ""])
    formatted_data.append(["Trend Indicators", "Value"])
    
    # Add trend indicators
    trend_indicators = analysis_results.get("trend_indicators", {})
    for name, data in trend_indicators.items():
        if name == "sma":
            formatted_data.append([f"SMA ({data.get('params', {}).get('length', 20)})", format_price(data.get("value", 0))])
        elif name == "ema":
            formatted_data.append([f"EMA ({data.get('params', {}).get('length', 20)})", format_price(data.get("value", 0))])
        elif name == "macd":
            macd_line = data.get("macd_line", 0)
            signal_line = data.get("signal_line", 0)
            histogram = data.get("histogram", 0)
            formatted_data.append(["MACD Line", f"{macd_line:.2f}"])
            formatted_data.append(["MACD Signal", f"{signal_line:.2f}"])
            formatted_data.append(["MACD Histogram", f"{histogram:.2f}"])
        elif name == "adx":
            formatted_data.append([f"ADX ({data.get('params', {}).get('length', 14)})", f"{data.get('value', 0):.2f}"])
    
    # Add educational content for trend indicators if requested
    if explain:
        formatted_data.append(["", ""])
        formatted_data.append([category_header("Understanding Trend Indicators"), ""])
        formatted_data.append([get_category_explanation("trend"), ""])
        
        for name in trend_indicators.keys():
            if name in ["sma", "ema", "macd", "adx"]:
                formatted_data.append([get_indicator_explanation(name), ""])
    
    # Add separator for momentum indicators
    formatted_data.append(["", ""])
    formatted_data.append(["Momentum Indicators", "Value"])
    
    # Add momentum indicators
    momentum_indicators = analysis_results.get("momentum_indicators", {})
    for name, data in momentum_indicators.items():
        if name == "rsi":
            formatted_data.append([f"RSI ({data.get('params', {}).get('length', 14)})", f"{data.get('value', 0):.2f}"])
        elif name == "stoch":
            k_line = data.get("k_line", 0)
            d_line = data.get("d_line", 0)
            formatted_data.append([f"Stochastic %K", f"{k_line:.2f}"])
            formatted_data.append([f"Stochastic %D", f"{d_line:.2f}"])
        elif name == "cci":
            formatted_data.append([f"CCI ({data.get('params', {}).get('length', 20)})", f"{data.get('value', 0):.2f}"])
    
    # Add educational content for momentum indicators if requested
    if explain:
        formatted_data.append(["", ""])
        formatted_data.append([category_header("Understanding Momentum Indicators"), ""])
        formatted_data.append([get_category_explanation("momentum"), ""])
        
        for name in momentum_indicators.keys():
            if name in ["rsi", "stoch", "cci"]:
                formatted_data.append([get_indicator_explanation(name), ""])
    
    # Add separator for volatility indicators
    formatted_data.append(["", ""])
    formatted_data.append(["Volatility Indicators", "Value"])
    
    # Add volatility indicators
    volatility_indicators = analysis_results.get("volatility_indicators", {})
    for name, data in volatility_indicators.items():
        if name == "bbands":
            upper = data.get("upper", 0)
            middle = data.get("middle", 0)
            lower = data.get("lower", 0)
            formatted_data.append([f"Bollinger Upper ({data.get('params', {}).get('length', 20)})", format_price(upper)])
            formatted_data.append([f"Bollinger Middle", format_price(middle)])
            formatted_data.append([f"Bollinger Lower", format_price(lower)])
        elif name == "atr":
            formatted_data.append([f"ATR ({data.get('params', {}).get('length', 14)})", f"{data.get('value', 0):.2f}"])
    
    # Add educational content for volatility indicators if requested
    if explain:
        formatted_data.append(["", ""])
        formatted_data.append([category_header("Understanding Volatility Indicators"), ""])
        formatted_data.append([get_category_explanation("volatility"), ""])
        
        for name in volatility_indicators.keys():
            if name in ["bbands", "atr"]:
                formatted_data.append([get_indicator_explanation(name), ""])
    
    # Add separator for volume indicators
    formatted_data.append(["", ""])
    formatted_data.append(["Volume Indicators", "Value"])
    
    # Add volume indicators
    volume_indicators = analysis_results.get("volume_indicators", {})
    for name, data in volume_indicators.items():
        if name == "obv":
            formatted_data.append([f"OBV", f"{data.get('value', 0):.2f}"])
    
    # Add educational content for volume indicators if requested
    if explain:
        formatted_data.append(["", ""])
        formatted_data.append([category_header("Understanding Volume Indicators"), ""])
        formatted_data.append([get_category_explanation("volume"), ""])
        
        for name in volume_indicators.keys():
            if name == "obv":
                formatted_data.append([get_indicator_explanation(name), ""])
    
    # Add separator for summary
    formatted_data.append(["", ""])
    formatted_data.append(["Market Summary", ""])
    
    # Add summary information
    summary = analysis_results.get("summary", {})
    trend = summary.get("trend", {})
    signals = summary.get("signals", {})
    
    trend_direction = trend.get("direction", "neutral").upper()
    trend_strength = trend.get("strength", "neutral").upper()
    short_term = signals.get("short_term", "neutral").upper()
    medium_term = signals.get("medium_term", "neutral").upper()
    long_term = signals.get("long_term", "neutral").upper()
    action = signals.get("action", "hold").upper()
    
    formatted_data.append([f"Trend Direction", trend_direction])
    formatted_data.append([f"Trend Strength", trend_strength])
    formatted_data.append([f"Short-term Signal", short_term])
    formatted_data.append([f"Medium-term Signal", medium_term])
    formatted_data.append([f"Long-term Signal", long_term])
    formatted_data.append([f"Suggested Action", action])
    
    # Add educational content for market summary if requested
    if explain:
        formatted_data.append(["", ""])
        formatted_data.append([category_header("Understanding Market Summary"), ""])
        formatted_data.append([get_category_explanation("market_summary"), ""])
        formatted_data.append([f"Trend Direction: {get_summary_explanation('trend_direction', trend_direction)}", ""])
        formatted_data.append([f"Trend Strength: {get_summary_explanation('trend_strength', trend_strength)}", ""])
        formatted_data.append([f"Signal Timeframes: {get_summary_explanation('signal_terms', 'short_term')}", ""])
        formatted_data.append([f"Suggested Action: {get_summary_explanation('actions', action)}", ""])
    
    # Add separator for analysis text
    formatted_data.append(["", ""])
    formatted_data.append(["Analysis", trend.get("analysis", "")])
    
    return formatted_data


def format_analysis_result(analysis_results: Dict[str, Any], explain: bool = False) -> str:
    """
    Format analysis results as a string.
    
    Args:
        analysis_results: Analysis results dictionary
        explain: Whether to include educational explanations
        
    Returns:
        Formatted string
    """
    # Get summary section
    summary = analysis_results.get("summary", {})
    
    # Format the results as a table
    formatted_results = tabulate(
        prepare_analysis_data(analysis_results, explain),
        headers=["Section", "Data"],
        tablefmt="fancy_grid"
    )
    
    # Return the formatted results
    return formatted_results


@analysis_app.callback()
def callback():
    """Handle analysis commands."""
    pass


@analysis_app.command()
def run(
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to analyze"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe for data (e.g., 1d, 4h, 1h)"),
    days: int = typer.Option(100, "--days", "-d", help="Number of days of historical data to fetch"),
    vs_currency: str = typer.Option("usd", "--vs_currency", "-c", help="Currency to calculate prices against"),
    force_refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh data from API"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Include educational explanations for technical indicators"),
    debug: bool = typer.Option(False, "--debug", "-dbg", help="Enable debug logging for this command")
):
    """
    Perform a comprehensive market analysis for a given symbol and timeframe.
    Fetches historical data, calculates various technical indicators, and presents a summary.
    """
    if debug:
        # Ensure logging is configured to show debug messages for this run
        # This might need a more sophisticated setup if basicConfig was already called by root logger
        logging.basicConfig(level=logging.DEBUG, force=True) # force=True if basicConfig might have been called
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Debug logging enabled for analysis run: {symbol}")

    current_price_data = None
    df_historical = None

    try:
        with display_spinner(f"Fetching current price for {symbol}..."):
            current_price_data = get_current_price(symbol, vs_currency=vs_currency)
        
        with display_spinner(f"Fetching historical data for {symbol} ({days} days, {timeframe} timeframe)..."):
            df_historical = get_historical_data(
                symbol=symbol, 
                vs_currency=vs_currency, 
                days=days, 
                timeframe=timeframe,
                # force_refresh=force_refresh # Pass force_refresh if get_historical_data supports it for caching
            )

        if df_historical is None or df_historical.empty:
            display_error(f"No historical data returned for '{symbol}' with parameters: timeframe={timeframe}, days={days}, currency={vs_currency}. Cannot proceed with analysis.")
            raise typer.Exit(code=1)
        
        if current_price_data is None or not current_price_data.get("current_price", {}).get(vs_currency):
            display_warning(f"Could not fetch current price for '{symbol}'. Analysis may be incomplete or use last historical price.")
            # Allow to proceed but summary might be affected. generate_analysis should handle this.

    except ValueError as ve: # Catch specific ValueErrors, e.g. from underlying API calls
        error_msg = str(ve).lower()
        if "invalid symbol" in error_msg or "not found" in error_msg or "unknown currency" in error_msg:
            display_error(f"Could not fetch data for '{symbol}' (currency: {vs_currency}). The symbol or currency may be invalid or not supported. Please check your inputs.")
        else:
            display_error(f"A data-related error occurred while fetching data for '{symbol}': {ve}")
        logger.debug(f"Detailed data fetching error for {symbol} (analysis command):", exc_info=True)
        raise typer.Exit(code=1)
    except ConnectionError as ce: # Example for network issues if underlying libraries raise it
        display_error(f"A network connection error occurred while fetching data for '{symbol}': {ce}. Please check your internet connection.")
        logger.debug(f"Detailed connection error for {symbol} (analysis command):", exc_info=True)
        raise typer.Exit(code=1)
    except Exception as e: # General catch-all for data fetching phase
        display_error(f"An unexpected error occurred while fetching data for '{symbol}': {e}")
        logger.debug(f"Detailed data fetching error for {symbol} (analysis command):", exc_info=True)
        raise typer.Exit(code=1)

    # Proceed with analysis if data fetching was successful
    try:
        with display_spinner(f"Generating analysis for {symbol}..."):
            # Ensure current_price_data is at least an empty dict if None, to avoid errors in generate_analysis
            analysis_results = generate_analysis(
                df_historical, 
                current_price_data if current_price_data else {},
                symbol=symbol, 
                timeframe=timeframe, 
                vs_currency=vs_currency
            )
        
        # Display results
        formatted_output = format_analysis_result(analysis_results, explain=explain)
        print(formatted_output) # rich print should handle this well
        
        if current_price_data and current_price_data.get("last_updated"):
            display_data_age(current_price_data["last_updated"])
        else:
            display_warning("Could not determine data age for current price.")

        display_success(f"Analysis for {symbol} completed.")

    except typer.Exit: # To prevent re-catching if generate_analysis or format_analysis_result calls typer.Exit
        raise
    except Exception as e:
        logger.error(f"Error during analysis generation or display for {symbol}: {e}", exc_info=True)
        display_error(f"Failed to complete analysis for '{symbol}' after fetching data: {e}")
        raise typer.Exit(code=1) 