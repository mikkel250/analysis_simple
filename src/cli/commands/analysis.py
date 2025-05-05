"""
Analysis Command Handler

Handles the 'analysis' command for generating comprehensive market analysis.
"""

import logging
from typing import Dict, Any
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

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
analysis_app = typer.Typer()


def generate_analysis(
    df: pd.DataFrame,
    current_price_data: Dict[str, Any],
    symbol: str = 'BTC',
    timeframe: str = '1d'
) -> Dict[str, Any]:
    """
    Generate comprehensive market analysis using multiple indicators.
    
    Args:
        df: Historical price DataFrame
        current_price_data: Current price data
        symbol: Symbol being analyzed
        timeframe: Timeframe of the data
        
    Returns:
        Dict: Analysis results
    """
    analysis_results = {
        "price_data": {},
        "trend_indicators": {},
        "momentum_indicators": {},
        "volatility_indicators": {},
        "volume_indicators": {},
        "summary": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Extract price data
    try:
        current_price = current_price_data.get("current_price", {}).get("usd", 0)
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


def format_analysis_result(analysis_results: Dict[str, Any]) -> str:
    """
    Format analysis results for display.
    
    Args:
        analysis_results: Analysis results
        
    Returns:
        str: Formatted results for display
    """
    formatted_data = []
    
    # Add price information
    price_data = analysis_results.get("price_data", {})
    formatted_data.append(["BTC/USDT Price", format_price(price_data.get("current_price", 0))])
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
    
    # Add separator for volume indicators
    formatted_data.append(["", ""])
    formatted_data.append(["Volume Indicators", "Value"])
    
    # Add volume indicators
    volume_indicators = analysis_results.get("volume_indicators", {})
    for name, data in volume_indicators.items():
        if name == "obv":
            formatted_data.append([f"OBV", f"{data.get('value', 0):.2f}"])
    
    # Add separator for summary
    formatted_data.append(["", ""])
    formatted_data.append(["Market Summary", ""])
    
    # Add summary information
    summary = analysis_results.get("summary", {})
    trend = summary.get("trend", {})
    signals = summary.get("signals", {})
    
    formatted_data.append([f"Trend Direction", trend.get("direction", "neutral").upper()])
    formatted_data.append([f"Trend Strength", trend.get("strength", "neutral").upper()])
    formatted_data.append([f"Short-term Signal", signals.get("short_term", "neutral").upper()])
    formatted_data.append([f"Medium-term Signal", signals.get("medium_term", "neutral").upper()])
    formatted_data.append([f"Long-term Signal", signals.get("long_term", "neutral").upper()])
    formatted_data.append([f"Suggested Action", signals.get("action", "hold").upper()])
    
    # Add separator for analysis text
    formatted_data.append(["", ""])
    formatted_data.append(["Analysis", trend.get("analysis", "")])
    
    return tabulate(formatted_data, tablefmt="fancy_grid")


@analysis_app.callback()
def callback():
    """Handle analysis commands."""
    pass


@analysis_app.command()
def run(
    symbol: str = typer.Option("BTC", "--symbol", "-s", help="Symbol to analyze"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Timeframe for data (e.g., 1d, 4h, 1h)"),
    days: int = typer.Option(100, "--days", "-d", help="Number of days of historical data to fetch"),
    force_refresh: bool = typer.Option(False, "--refresh", "-r", help="Force refresh data from API"),
    forecast: bool = typer.Option(False, "--forecast", "-f", help="Include forecasting based on historical data"),
):
    """
    Generate comprehensive market analysis with multiple indicators.
    """
    try:
        # Get current price
        with display_spinner("Fetching current price..."):
            current_price_data = get_current_price(symbol, force_refresh)
        
        if not current_price_data:
            display_error(f"Failed to get current price for {symbol}")
            return
        
        # Get historical data
        with display_spinner(f"Fetching {timeframe} historical data..."):
            df = get_historical_data(symbol, timeframe, days, force_refresh)
        
        if df is None or df.empty:
            display_error(f"Failed to get historical data for {symbol}")
            return
        
        display_data_age(df.index[-1])
        
        # Generate analysis
        with display_spinner("Generating comprehensive analysis..."):
            analysis_results = generate_analysis(df, current_price_data, symbol, timeframe)
        
        # Generate forecast if requested
        if forecast:
            with display_spinner("Generating price forecast..."):
                try:
                    # Simple forecasting using exponential smoothing
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    
                    # Prepare data for forecasting
                    close_prices = df['close'].values
                    
                    # Configure and fit the model
                    model = ExponentialSmoothing(
                        close_prices, 
                        trend='add',  # Add trend component
                        seasonal=None,  # No seasonal component for simplicity
                        seasonal_periods=None
                    ).fit()
                    
                    # Generate forecast for the next 7 periods
                    forecast_periods = 7
                    forecast_values = model.forecast(forecast_periods)
                    
                    # Create a formatted table with forecasted values
                    forecast_data = [["Forecast Period", "Predicted Price", "Change"]]
                    
                    current_price = close_prices[-1]
                    for i, value in enumerate(forecast_values):
                        period = i + 1
                        change = ((value - current_price) / current_price) * 100
                        change_direction = "+" if change >= 0 else ""
                        forecast_data.append([
                            f"{period} {timeframe}" + ("s" if period > 1 else ""),
                            format_price(value),
                            f"{change_direction}{change:.2f}%"
                        ])
                    
                    print("\n" + tabulate(forecast_data, headers="firstrow", tablefmt="fancy_grid"))
                    display_warning("Forecast is based on historical patterns and should not be used as financial advice.")
                    
                except ImportError:
                    display_warning("Forecasting requires statsmodels package. Install with: pip install statsmodels")
                except Exception as e:
                    display_warning(f"Could not generate forecast: {str(e)}")
        
        # Format and display the analysis
        formatted_output = format_analysis_result(analysis_results)
        print(formatted_output)
        
        display_success("Market analysis completed successfully")
        
    except Exception as e:
        display_error(f"Error generating analysis: {str(e)}")
        logger.exception("Error in analysis command") 