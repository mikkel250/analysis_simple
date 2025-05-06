"""
Section Formatters for Cryptocurrency Dashboard

This module provides functions for formatting different sections of the
HTML dashboard, such as price information, market trends, technical signals,
candlestick patterns, and recommended actions.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
from .style_utils import get_color_for_value, get_trend_color, get_trend_symbol, format_value_with_color


def format_price_section(price_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """
    Format the price section of the dashboard.
    
    Args:
        price_data: Dictionary containing price data
        metadata: Dictionary containing metadata like symbol, timeframe, etc.
        
    Returns:
        HTML string for the price section
    """
    current_price = price_data.get("close", 0)
    open_price = price_data.get("open", 0)
    high_price = price_data.get("high", 0)
    low_price = price_data.get("low", 0)
    volume = price_data.get("volume", 0)
    
    # Calculate price change and percentage
    price_change = current_price - open_price
    price_change_pct = (price_change / open_price) * 100 if open_price else 0
    
    # Get colors based on price change
    price_color = get_color_for_value(price_change)
    
    # Format the values with colors
    formatted_price_change = format_value_with_color(price_change, prefix='+' if price_change > 0 else '')
    formatted_price_change_pct = format_value_with_color(price_change_pct, prefix='+' if price_change_pct > 0 else '', suffix='%')
    
    # Build the HTML
    symbol = metadata.get("symbol", "UNKNOWN")
    timeframe = metadata.get("timeframe", "Unknown Timeframe")
    last_updated = metadata.get("last_updated", "Unknown")
    
    price_section_html = f"""
    <div class="price-section">
        <div class="price-header">
            <h2>{symbol} Price Analysis</h2>
            <div class="price-metadata">
                <span class="timeframe">{timeframe}</span>
                <span class="update-time">Last Updated: {last_updated}</span>
            </div>
        </div>
        
        <div class="current-price-container">
            <div class="current-price-value">
                <span class="price-label">Current Price:</span>
                <span class="price-value" style="color: {price_color}">${current_price:,.2f}</span>
            </div>
            <div class="price-change">
                <span class="change-label">Change:</span>
                <span class="change-value">{formatted_price_change} ({formatted_price_change_pct})</span>
            </div>
        </div>
        
        <div class="price-details">
            <div class="price-detail-item">
                <span class="detail-label">Open:</span>
                <span class="detail-value">${open_price:,.2f}</span>
            </div>
            <div class="price-detail-item">
                <span class="detail-label">High:</span>
                <span class="detail-value">${high_price:,.2f}</span>
            </div>
            <div class="price-detail-item">
                <span class="detail-label">Low:</span>
                <span class="detail-value">${low_price:,.2f}</span>
            </div>
            <div class="price-detail-item">
                <span class="detail-label">Volume:</span>
                <span class="detail-value">{volume:,.0f}</span>
            </div>
        </div>
    </div>
    """
    
    return price_section_html


def format_market_trend_analysis(analysis_results: Dict[str, Any]) -> str:
    """
    Format the market trend analysis section of the dashboard.
    
    Args:
        analysis_results: Dictionary containing market trend analysis data
        
    Returns:
        HTML string for the market trend analysis section
    """
    # Extract trend data from the analysis results
    trends = analysis_results.get("trends", {})
    
    # If trends data is missing, return an empty section
    if not trends:
        return """
        <div class="market-trend-section">
            <h3>Market Trend Analysis</h3>
            <p>No trend data available</p>
        </div>
        """
    
    # Process each timeframe
    trend_rows = ""
    
    for timeframe, trend in trends.items():
        # Format the timeframe for display
        display_timeframe = timeframe.replace("_", " ").title()
        
        # Get the trend signal
        signal = trend.get("signal", "neutral")
        
        # Get color and symbol for the trend
        color = get_trend_color(signal)
        symbol = get_trend_symbol(signal)
        
        # Create the row
        trend_rows += f"""
        <div class="trend-row">
            <div class="trend-timeframe">{display_timeframe}</div>
            <div class="trend-signal" style="color: {color}">
                {symbol} {signal.title()}
            </div>
        </div>
        """
    
    # Build the complete HTML for the section
    trend_html = f"""
    <div class="market-trend-section">
        <h3>Market Trend Analysis</h3>
        <div class="trend-table">
            {trend_rows}
        </div>
    </div>
    """
    
    return trend_html


def format_technical_signals(analysis_results: Dict[str, Any]) -> str:
    """
    Format the technical signals section of the dashboard.
    
    Args:
        analysis_results: Dictionary containing technical indicator data
        
    Returns:
        HTML string for the technical signals section
    """
    # Extract indicators data from the analysis results
    indicators = analysis_results.get("indicators", {})
    
    # If indicators data is missing, return an empty section
    if not indicators:
        return """
        <div class="technical-signals-section">
            <h3>Technical Signals</h3>
            <p>No indicator data available</p>
        </div>
        """
    
    # Keep track of the overall signal count
    signal_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    
    # Process each indicator
    indicator_rows = ""
    
    for indicator_name, indicator_data in indicators.items():
        # Format the indicator name for display
        display_name = indicator_name.replace("_", " ").upper()
        
        # Get the indicator signal
        signal = indicator_data.get("signal", "neutral")
        signal_counts[signal] += 1
        
        # Get value if available
        value = indicator_data.get("value", None)
        value_display = f"{value:.2f}" if value is not None else "N/A"
        
        # Get color and symbol for the signal
        color = get_trend_color(signal)
        symbol = get_trend_symbol(signal)
        
        # Create the row
        indicator_rows += f"""
        <div class="indicator-row">
            <div class="indicator-name">{display_name}</div>
            <div class="indicator-value">{value_display}</div>
            <div class="indicator-signal" style="color: {color}">
                {symbol} {signal.title()}
            </div>
        </div>
        """
    
    # Determine overall technical sentiment
    total_signals = sum(signal_counts.values())
    if total_signals > 0:
        bullish_percentage = (signal_counts["bullish"] / total_signals) * 100
        bearish_percentage = (signal_counts["bearish"] / total_signals) * 100
        
        if bullish_percentage > 60:
            overall_signal = "bullish"
        elif bearish_percentage > 60:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"
        
        overall_color = get_trend_color(overall_signal)
        overall_symbol = get_trend_symbol(overall_signal)
        
        overall_sentiment = f"""
        <div class="overall-sentiment">
            <div class="sentiment-label">Overall Sentiment:</div>
            <div class="sentiment-value" style="color: {overall_color}">
                {overall_symbol} {overall_signal.title()}
            </div>
            <div class="sentiment-distribution">
                <div class="bullish-percentage" style="width: {bullish_percentage}%; background-color: chartreuse"></div>
                <div class="neutral-percentage" style="width: {100 - bullish_percentage - bearish_percentage}%; background-color: orange"></div>
                <div class="bearish-percentage" style="width: {bearish_percentage}%; background-color: red"></div>
            </div>
            <div class="sentiment-legend">
                <span class="bullish-legend">Bullish: {bullish_percentage:.0f}%</span>
                <span class="neutral-legend">Neutral: {100 - bullish_percentage - bearish_percentage:.0f}%</span>
                <span class="bearish-legend">Bearish: {bearish_percentage:.0f}%</span>
            </div>
        </div>
        """
    else:
        overall_sentiment = """
        <div class="overall-sentiment">
            <div class="sentiment-label">Overall Sentiment:</div>
            <div class="sentiment-value">Not enough data</div>
        </div>
        """
    
    # Build the complete HTML for the section
    technical_html = f"""
    <div class="technical-signals-section">
        <h3>Technical Signals</h3>
        {overall_sentiment}
        <div class="indicator-table">
            {indicator_rows}
        </div>
    </div>
    """
    
    return technical_html


def format_candlestick_patterns(analysis_results: Dict[str, Any]) -> str:
    """
    Format the candlestick patterns section of the dashboard.
    
    Args:
        analysis_results: Dictionary containing candlestick pattern data
        
    Returns:
        HTML string for the candlestick patterns section
    """
    # Extract patterns data from the analysis results
    patterns = analysis_results.get("patterns", {})
    
    # If patterns data is missing, return an empty section
    if not patterns:
        return """
        <div class="candlestick-patterns-section">
            <h3>Candlestick Patterns</h3>
            <p>No pattern data available</p>
        </div>
        """
    
    # Process each pattern
    pattern_rows = ""
    
    # Sort patterns by reliability (if available) or alphabetically
    sorted_patterns = sorted(
        patterns.items(),
        key=lambda x: x[1].get("reliability", 0),
        reverse=True
    )
    
    for pattern_name, pattern_data in sorted_patterns:
        # Format the pattern name for display
        display_name = pattern_name.replace("_", " ").title()
        
        # Get pattern properties
        signal = pattern_data.get("signal", "neutral")
        reliability = pattern_data.get("reliability", 0)
        
        # Get color and symbol for the signal
        color = get_trend_color(signal)
        symbol = get_trend_symbol(signal)
        
        # Create a reliability display (e.g., stars)
        reliability_stars = "★" * min(int(reliability * 5), 5)
        
        # Create the row
        pattern_rows += f"""
        <div class="pattern-row">
            <div class="pattern-name">{display_name}</div>
            <div class="pattern-signal" style="color: {color}">
                {symbol} {signal.title()}
            </div>
            <div class="pattern-reliability">{reliability_stars}</div>
        </div>
        """
    
    # Build the complete HTML for the section
    patterns_html = f"""
    <div class="candlestick-patterns-section">
        <h3>Candlestick Patterns</h3>
        <div class="pattern-table">
            {pattern_rows}
        </div>
    </div>
    """
    
    return patterns_html


def format_recommended_action(analysis_results: Dict[str, Any]) -> str:
    """
    Format the recommended action section of the dashboard.
    
    Args:
        analysis_results: Dictionary containing recommendation data
        
    Returns:
        HTML string for the recommended action section
    """
    # Extract recommendation data
    recommendation = analysis_results.get("recommendation", {})
    
    # If recommendation data is missing, return an empty section
    if not recommendation:
        return """
        <div class="recommendation-section">
            <h3>Recommended Action</h3>
            <p>No recommendation available</p>
        </div>
        """
    
    # Extract recommendation properties
    action = recommendation.get("action", "hold")
    confidence = recommendation.get("confidence", 0.5)
    rationale = recommendation.get("rationale", "Insufficient data to provide a detailed rationale.")
    
    # Determine display values
    display_action = action.upper()
    confidence_percentage = f"{confidence * 100:.0f}%"
    
    # Get color for the action
    action_color = "chartreuse" if action == "buy" else "red" if action == "sell" else "orange"
    
    # Create a confidence bar
    confidence_bar = f"""
    <div class="confidence-bar-container">
        <div class="confidence-bar" style="width: {confidence * 100}%; background-color: {action_color}"></div>
    </div>
    """
    
    # Build the HTML
    recommendation_html = f"""
    <div class="recommendation-section">
        <h3>Recommended Action</h3>
        <div class="action-container">
            <div class="action-label">Action:</div>
            <div class="action-value" style="color: {action_color}">{display_action}</div>
        </div>
        <div class="confidence-container">
            <div class="confidence-label">Confidence:</div>
            <div class="confidence-value">{confidence_percentage}</div>
            {confidence_bar}
        </div>
        <div class="rationale-container">
            <div class="rationale-label">Rationale:</div>
            <div class="rationale-text">{rationale}</div>
        </div>
    </div>
    """
    
    return recommendation_html 