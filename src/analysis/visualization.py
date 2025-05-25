import logging
from typing import Dict, Any
import pandas as pd
from src.plotting import charts

logger = logging.getLogger(__name__)

def generate_visualizations_for_market(data: pd.DataFrame, symbol: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate visualizations for the given market data and analysis results.
    Returns a dictionary of plotly figures keyed by visualization type.
    """
    if data is None or data.empty or "close" not in data.columns:
        logger.warning(
            f"Cannot generate visualizations for {symbol} due to missing or empty data."
        )
        return {}

    if analysis_results.get("error"):
        logger.warning(
            f"Skipping visualizations due to error in core analysis for {symbol}."
        )
        return {}

    logger.info(f"Generating visualizations for {symbol}")
    try:
        price_fig = charts.plot_price_history(data, symbol)
        technical_fig = charts.plot_technical_analysis(data, symbol)
        candlestick_fig = charts.plot_candlestick(data, symbol)

        visualizations = {
            "price_history": price_fig,
            "technical_indicators": technical_fig,
            "candlestick": candlestick_fig,
        }
        logger.info(f"Visualizations generated for {symbol}.")
        return visualizations
    except Exception as e:
        logger.error(
            f"Error generating visualizations for {symbol}: {e}", exc_info=True
        )
        return {} 