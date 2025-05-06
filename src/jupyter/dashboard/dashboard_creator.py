"""
Dashboard Creator for Cryptocurrency Analysis

This module provides the main function for creating the HTML dashboard by assembling
all the components from the other modules.
"""

from typing import Dict, Any, Union, List, Tuple, Optional
from IPython.display import HTML

from .css_styles import get_css_styles
from .section_formatters import (
    format_price_section,
    format_market_trend_analysis,
    format_technical_signals,
    format_candlestick_patterns,
    format_recommended_action
)
from .educational_content import (
    get_indicator_education,
    get_trend_education,
    format_methodology_section
)


def create_text_dashboard(analysis_results: Dict[str, Any]) -> HTML:
    """
    Create a detailed HTML dashboard for cryptocurrency analysis results.
    
    Args:
        analysis_results: Dictionary containing all analysis data, including price data,
                          trends, indicators, patterns, and recommendations.
        
    Returns:
        IPython.display.HTML object containing the complete dashboard
    """
    # Extract the metadata for display
    metadata = analysis_results.get("metadata", {})
    
    # Extract price data
    price_data = analysis_results.get("price_data", {})
    
    # Generate the CSS styles
    styles = get_css_styles()
    
    # Format each section of the dashboard
    price_section = format_price_section(price_data, metadata)
    trend_section = format_market_trend_analysis(analysis_results)
    technical_section = format_technical_signals(analysis_results)
    patterns_section = format_candlestick_patterns(analysis_results)
    recommendation_section = format_recommended_action(analysis_results)
    
    # Generate educational content if requested
    educational_content = ""
    if analysis_results.get("include_education", False):
        
        # Get the primary indicator for education
        indicators = analysis_results.get("indicators", {})
        primary_indicator = None
        if indicators:
            # Find the indicator with the strongest signal
            for name, data in indicators.items():
                signal = data.get("signal", "neutral")
                if signal != "neutral":
                    primary_indicator = name
                    break
            
            # If no strong signal, just use the first one
            if primary_indicator is None and indicators:
                primary_indicator = list(indicators.keys())[0]
        
        # Get the primary trend timeframe for education
        trends = analysis_results.get("trends", {})
        primary_timeframe = None
        if trends:
            # Find the medium-term trend if available
            if "medium_term" in trends:
                primary_timeframe = "medium_term"
            # Otherwise use the first available timeframe
            else:
                primary_timeframe = list(trends.keys())[0]
        
        # Generate the educational content
        edu_sections = []
        
        # Add indicator education if available
        if primary_indicator:
            indicator_data = indicators[primary_indicator]
            indicator_value = indicator_data.get("value")
            indicator_signal = indicator_data.get("signal", "neutral")
            
            indicator_edu = get_indicator_education(
                primary_indicator,
                current_value=indicator_value,
                signal=indicator_signal
            )
            edu_sections.append(indicator_edu)
        
        # Add trend education if available
        if primary_timeframe:
            trend_data = trends[primary_timeframe]
            trend_signal = trend_data.get("signal", "neutral")
            
            trend_edu = get_trend_education(
                primary_timeframe,
                signal=trend_signal
            )
            edu_sections.append(trend_edu)
        
        # Add methodology section
        methodology = format_methodology_section()
        edu_sections.append(methodology)
        
        # Combine all educational content
        educational_content = f"""
        <div class="dashboard-section educational-section">
            <h2>Educational Information</h2>
            {''.join(edu_sections)}
        </div>
        """
    
    # Assemble the complete dashboard HTML
    dashboard_html = f"""
    {styles}
    <div class="crypto-dashboard">
        <div class="dashboard-header">
            <h1>Cryptocurrency Analysis Dashboard</h1>
        </div>
        
        <div class="dashboard-section">
            {price_section}
        </div>
        
        <div class="dashboard-section">
            {trend_section}
        </div>
        
        <div class="dashboard-section">
            {technical_section}
        </div>
        
        <div class="dashboard-section">
            {patterns_section}
        </div>
        
        <div class="dashboard-section">
            {recommendation_section}
        </div>
        
        {educational_content}
    </div>
    """
    
    # Return the HTML object
    return HTML(dashboard_html) 