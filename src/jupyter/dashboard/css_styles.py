"""
CSS Styles for Cryptocurrency Dashboard

This module provides the CSS styling for the HTML dashboard.
"""

from typing import Dict, Any, Union, List, Tuple, Optional


def get_css_styles() -> str:
    """
    Get the CSS styles for the HTML dashboard.
    
    Returns:
        CSS styles as a string
    """
    styles = """
    <style>
        /* General Styles */
        .crypto-dashboard {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #1a1a1a;
            color: #e0e0e0;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
        }
        
        .dashboard-header {
            border-bottom: 1px solid #444;
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .dashboard-header h1 {
            margin: 0;
            color: #ffffff;
            font-size: 24px;
        }
        
        .dashboard-section {
            margin-bottom: 30px;
            padding: 15px;
            background-color: #252525;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        h2 {
            color: #ffffff;
            font-size: 20px;
            margin-top: 0;
            margin-bottom: 15px;
        }
        
        h3 {
            color: #ffffff;
            font-size: 18px;
            margin-top: 0;
            margin-bottom: 15px;
            border-bottom: 1px solid #444;
            padding-bottom: 5px;
        }
        
        h4 {
            color: #cccccc;
            font-size: 16px;
            margin-top: 15px;
            margin-bottom: 5px;
        }
        
        p {
            margin: 8px 0;
            line-height: 1.5;
        }
        
        /* Price Section Styles */
        .price-section {
            display: flex;
            flex-direction: column;
        }
        
        .price-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .price-metadata {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            font-size: 14px;
            color: #999;
        }
        
        .current-price-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        
        .current-price-value {
            display: flex;
            flex-direction: column;
        }
        
        .price-label {
            font-size: 14px;
            color: #999;
            margin-bottom: 5px;
        }
        
        .price-value {
            font-size: 28px;
            font-weight: bold;
        }
        
        .price-change {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
        }
        
        .change-label {
            font-size: 14px;
            color: #999;
            margin-bottom: 5px;
        }
        
        .change-value {
            font-size: 18px;
        }
        
        .price-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
        }
        
        .price-detail-item {
            padding: 10px;
            background-color: #2a2a2a;
            border-radius: 5px;
            display: flex;
            flex-direction: column;
        }
        
        .detail-label {
            font-size: 14px;
            color: #999;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 16px;
            font-weight: bold;
        }
        
        /* Market Trend Styles */
        .market-trend-section {
            display: flex;
            flex-direction: column;
        }
        
        .trend-table {
            display: flex;
            flex-direction: column;
        }
        
        .trend-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        
        .trend-row:last-child {
            border-bottom: none;
        }
        
        .trend-timeframe {
            font-weight: bold;
        }
        
        .trend-signal {
            font-weight: bold;
        }
        
        /* Technical Signals Styles */
        .technical-signals-section {
            display: flex;
            flex-direction: column;
        }
        
        .overall-sentiment {
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 5px;
            margin-bottom: 15px;
        }
        
        .sentiment-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .sentiment-value {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .sentiment-distribution {
            display: flex;
            height: 10px;
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 5px;
        }
        
        .sentiment-legend {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            color: #999;
        }
        
        .indicator-table {
            display: flex;
            flex-direction: column;
        }
        
        .indicator-row {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        
        .indicator-row:last-child {
            border-bottom: none;
        }
        
        .indicator-name {
            font-weight: bold;
        }
        
        .indicator-value {
            text-align: center;
        }
        
        .indicator-signal {
            text-align: right;
            font-weight: bold;
        }
        
        /* Candlestick Patterns Styles */
        .candlestick-patterns-section {
            display: flex;
            flex-direction: column;
        }
        
        .pattern-table {
            display: flex;
            flex-direction: column;
        }
        
        .pattern-row {
            display: grid;
            grid-template-columns: 3fr 1fr 1fr;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #333;
        }
        
        .pattern-row:last-child {
            border-bottom: none;
        }
        
        .pattern-name {
            font-weight: bold;
        }
        
        .pattern-signal {
            text-align: center;
            font-weight: bold;
        }
        
        .pattern-reliability {
            text-align: right;
            color: gold;
        }
        
        /* Recommendation Styles */
        .recommendation-section {
            display: flex;
            flex-direction: column;
        }
        
        .action-container {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .action-label {
            font-weight: bold;
            margin-right: 10px;
        }
        
        .action-value {
            font-size: 24px;
            font-weight: bold;
        }
        
        .confidence-container {
            margin-bottom: 15px;
        }
        
        .confidence-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .confidence-value {
            margin-bottom: 5px;
        }
        
        .confidence-bar-container {
            height: 10px;
            background-color: #444;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .confidence-bar {
            height: 100%;
        }
        
        .rationale-container {
            padding: 15px;
            background-color: #2a2a2a;
            border-radius: 5px;
        }
        
        .rationale-label {
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        /* Educational Content Styles */
        .educational-section {
            background-color: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        
        .indicator-education, .trend-education {
            margin-bottom: 20px;
        }
        
        .indicator-description, .trend-description,
        .indicator-calculation, .trend-timeframe,
        .indicator-value, .trend-indicators,
        .indicator-interpretation, .trend-interpretation {
            margin-bottom: 15px;
        }
        
        .methodology-section {
            margin-top: 20px;
        }
        
        .methodology-disclaimer {
            margin-top: 15px;
            padding: 10px;
            background-color: #333;
            border-radius: 5px;
            font-size: 14px;
        }
        
        /* Responsive Styles */
        @media (max-width: 768px) {
            .current-price-container {
                flex-direction: column;
                align-items: flex-start;
            }
            
            .price-change {
                align-items: flex-start;
                margin-top: 10px;
            }
            
            .indicator-row, .pattern-row {
                grid-template-columns: 1fr;
                gap: 5px;
            }
            
            .indicator-value, .indicator-signal, 
            .pattern-signal, .pattern-reliability {
                text-align: left;
            }
        }
    </style>
    """
    
    return styles 