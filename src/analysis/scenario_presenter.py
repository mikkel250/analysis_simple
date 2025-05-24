"""
Market scenario presentation module.

This module provides functions for analyzing possible market scenarios
and presenting potential trading/investment cases based on market data.
"""

from typing import Dict, Any, Union, List, Optional
import pandas as pd
import numpy as np
from src.config.logging_config import get_logger
from src.analysis.error_handling import (
    ValidationError, IndicatorError, validate_dataframe, safe_operation
)

# Set up logger
logger = get_logger(__name__)

def present_cases(data: pd.DataFrame, trend_data: Dict[str, Any], open_interest: dict = None) -> Dict[str, Any]:
    """
    Generate potential market scenarios and trading cases based on market data.
    
    Args:
        data: DataFrame with market data and technical indicators
        trend_data: Dict with trend analysis results
        open_interest: Dict with open interest data (optional)
        
    Returns:
        Dict with different market scenarios and probability assessments
        
    Raises:
        ValidationError: If the data doesn't meet requirements
        IndicatorError: If there's an error generating scenarios
    """
    logger.info("Generating market scenarios and trading cases")
    
    try:
        # Validate input data
        required_columns = ['close', 'high', 'low']
        validate_dataframe(data, required_columns=required_columns, min_rows=10)
        
        if not isinstance(trend_data, dict):
            raise ValidationError(f"Invalid trend data: expected dict, got {type(trend_data)}")
            
        if 'overall_trend' not in trend_data:
            raise ValidationError("Invalid trend data: missing 'overall_trend' key")
    
        # Log input data shape for debugging
        logger.debug(f"Data shape: {data.shape}")
        logger.debug(f"Using trend data: {trend_data.get('overall_trend', 'unknown')}")
        
        # Extract recent data for analysis
        recent_data = data.tail(20)
        current = recent_data.iloc[-1]
        logger.debug(f"Using most recent data point for scenario analysis: {current.name}")
        
        # Initialize the scenarios structure
        scenarios = {
            "bullish": {
                "probability": 0.0,
                "confidence": "low",
                "potential": "unknown",
                "supporting_factors": [],
                "risk_factors": [],
                "triggers": [],
                "targets": [],
                "scenario_summary": ""
            },
            "bearish": {
                "probability": 0.0,
                "confidence": "low",
                "potential": "unknown",
                "supporting_factors": [],
                "risk_factors": [],
                "triggers": [],
                "targets": [],
                "scenario_summary": ""
            },
            "sideways": {
                "probability": 0.0,
                "confidence": "low",
                "potential": "unknown",
                "supporting_factors": [],
                "risk_factors": [],
                "triggers": [],
                "targets": [],
                "scenario_summary": ""
            },
            "current_assessment": {
                "primary_scenario": "neutral",
                "primary_probability": 0.0,
                "secondary_scenario": "neutral",
                "secondary_probability": 0.0,
                "most_significant_factor": "",
                "confidence": "low"
            }
        }
        
        # Get overall trend from trend data
        overall_trend = trend_data.get('overall_trend', 'neutral')
        trend_confidence = trend_data.get('confidence', 'medium')
        
        logger.debug(f"Overall trend: {overall_trend}, confidence: {trend_confidence}")
        
        # Analyze bullish scenario
        bullish_scenario = safe_operation(
            lambda: _analyze_bullish_scenario(data, trend_data, open_interest=open_interest),
            fallback={
                "probability": 0.3,
                "confidence": "low",
                "potential": "unknown",
                "supporting_factors": [],
                "risk_factors": ["Error in analysis"],
                "triggers": [],
                "targets": [],
                "scenario_summary": "Unable to analyze bullish scenario"
            },
            error_msg="Error generating bullish scenario",
            logger_instance=logger
        )
        scenarios["bullish"] = bullish_scenario
        
        # Analyze bearish scenario
        bearish_scenario = safe_operation(
            lambda: _analyze_bearish_scenario(data, trend_data, open_interest=open_interest),
            fallback={
                "probability": 0.3,
                "confidence": "low",
                "potential": "unknown",
                "supporting_factors": [],
                "risk_factors": ["Error in analysis"],
                "triggers": [],
                "targets": [],
                "scenario_summary": "Unable to analyze bearish scenario"
            },
            error_msg="Error generating bearish scenario",
            logger_instance=logger
        )
        scenarios["bearish"] = bearish_scenario
        
        # Analyze sideways scenario
        sideways_scenario = safe_operation(
            lambda: _analyze_sideways_scenario(data, trend_data, open_interest=open_interest),
            fallback={
                "probability": 0.3,
                "confidence": "low",
                "potential": "unknown",
                "supporting_factors": [],
                "risk_factors": ["Error in analysis"],
                "triggers": [],
                "targets": [],
                "scenario_summary": "Unable to analyze sideways scenario"
            },
            error_msg="Error generating sideways scenario",
            logger_instance=logger
        )
        scenarios["sideways"] = sideways_scenario
        
        # Determine primary and secondary scenarios based on probabilities
        probs = {
            "bullish": bullish_scenario["probability"],
            "bearish": bearish_scenario["probability"],
            "sideways": sideways_scenario["probability"]
        }
        
        # Sort scenarios by probability
        sorted_scenarios = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primary = sorted_scenarios[0]
        secondary = sorted_scenarios[1] if len(sorted_scenarios) > 1 else None
        
        scenarios["current_assessment"]["primary_scenario"] = primary[0]
        scenarios["current_assessment"]["primary_probability"] = primary[1]
        
        if secondary:
            scenarios["current_assessment"]["secondary_scenario"] = secondary[0]
            scenarios["current_assessment"]["secondary_probability"] = secondary[1]
        
        # Determine confidence in the assessment
        primary_confidence = scenarios[primary[0]]["confidence"]
        scenarios["current_assessment"]["confidence"] = primary_confidence
        
        # Find most significant factor
        primary_supports = scenarios[primary[0]]["supporting_factors"]
        if primary_supports:
            scenarios["current_assessment"]["most_significant_factor"] = primary_supports[0]
        
        logger.info(f"Generated market scenarios - Primary: {primary[0]} ({primary[1]:.1%}), Confidence: {primary_confidence}")
        return scenarios
        
    except ValidationError as e:
        logger.error(f"Validation error in present_cases: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Error generating market scenarios: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise IndicatorError(error_msg)

def _analyze_bullish_scenario(data: pd.DataFrame, trend_data: Dict[str, Any], open_interest: dict = None) -> Dict[str, Any]:
    """
    Analyze the bullish market scenario.
    
    Args:
        data: DataFrame with market data and technical indicators
        trend_data: Dict with trend analysis results
        open_interest: Dict with open interest data (optional)
        
    Returns:
        Dict with bullish scenario analysis
        
    Raises:
        Exception: If analysis fails
    """
    logger.debug("Analyzing bullish scenario")
    
    # Validate inputs
    if data is None or data.empty or len(data) < 5:
        raise ValidationError("Insufficient data for bullish scenario analysis")
        
    if not isinstance(trend_data, dict):
        raise ValidationError("Invalid trend data format")
    
    # Initialize results
    scenario = {
        "probability": 0.3,  # Default middle probability
        "confidence": "low",
        "potential": "unknown",
        "supporting_factors": [],
        "risk_factors": [],
        "triggers": [],
        "targets": [],
        "scenario_summary": ""
    }
        
    # Extract the most recent data point
    current = data.iloc[-1]
    recent_data = data.tail(20)
    
    # Basic probability based on trend
    if trend_data.get('overall_trend') == 'bullish':
        if trend_data.get('confidence') == 'high':
            scenario["probability"] = 0.75
        elif trend_data.get('confidence') == 'medium':
            scenario["probability"] = 0.6
        else:
            scenario["probability"] = 0.45
    elif trend_data.get('overall_trend') == 'neutral':
        scenario["probability"] = 0.3
    else:  # bearish trend
        scenario["probability"] = 0.15
        
    # Check if price is above key moving averages
    ma_signals = []
    if 'close' in current and 'sma_20' in current and current['close'] > current['sma_20']:
        ma_signals.append(("Price above SMA(20)", 0.05))
        
    if 'close' in current and 'sma_50' in current and current['close'] > current['sma_50']:
        ma_signals.append(("Price above SMA(50)", 0.1))
        
    if 'close' in current and 'sma_200' in current and current['close'] > current['sma_200']:
        ma_signals.append(("Price above SMA(200)", 0.15))
    
    # Check RSI for bullish signals
    rsi_signals = []
    if 'rsi_14' in current:
        rsi = current['rsi_14']
        if pd.notna(rsi):  # Check for NaN values
            if 30 < rsi < 70:
                if rsi > 55:
                    rsi_signals.append((f"RSI(14) showing strength ({rsi:.1f})", 0.1))
                elif rsi > 45:
                    rsi_signals.append((f"RSI(14) neutral to positive ({rsi:.1f})", 0.05))
                elif rsi < 35:
                    rsi_signals.append((f"RSI(14) showing potential oversold reversal opportunity ({rsi:.1f})", 0.03))
    
    # Check MACD for bullish signals
    macd_signals = []
    if all(col in current for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
        macd = current['MACD_12_26_9']
        signal = current['MACDs_12_26_9']
        if pd.notna(macd) and pd.notna(signal):  # Check for NaN values
            if macd > signal:
                macd_signals.append(("MACD above signal line", 0.1))
            if macd > 0:
                macd_signals.append(("MACD above zero line", 0.05))
            elif macd_signals and macd < 0 and macd > signal:
                macd_signals.append(("MACD rising from below zero", 0.03))
    
    # Identify potential price targets
    close = current['close'] if 'close' in current and pd.notna(current['close']) else None
    if close is not None:
        # Simple percentage-based targets
        scenario["targets"] = [
            {
                "level": round(close * 1.05, 2),
                "description": "Short-term resistance (5% above current)",
                "type": "resistance",
                "strength": "weak"
            },
            {
                "level": round(close * 1.10, 2),
                "description": "Medium-term target (10% above current)",
                "type": "target",
                "strength": "medium"
            },
            {
                "level": round(close * 1.20, 2),
                "description": "Long-term target (20% above current)",
                "type": "target", 
                "strength": "strong"
            }
        ]
        
        # Find potential trigger points
        if 'sma_20' in current and pd.notna(current['sma_20']):
            scenario["triggers"].append({
                "level": round(current['sma_20'], 2),
                "description": "Break above/below SMA(20)",
                "significance": "minor"
            })
        
        if 'sma_50' in current and pd.notna(current['sma_50']):
            scenario["triggers"].append({
                "level": round(current['sma_50'], 2),
                "description": "Break above/below SMA(50)",
                "significance": "moderate"
            })
        
        # Calculate potential reward
        if scenario["targets"]:
            best_target = scenario["targets"][-1]["level"]
            scenario["potential"] = f"{((best_target/close - 1) * 100):.1f}%"
    
    # Collect all supporting factors and adjust probability
    for signal, prob_adjust in ma_signals + rsi_signals + macd_signals:
        scenario["supporting_factors"].append(signal)
        scenario["probability"] = min(0.95, scenario["probability"] + prob_adjust)
        
    # Collect risk factors
    if trend_data.get('overall_trend') == 'bearish':
        scenario["risk_factors"].append("Overall trend is bearish")
        
    if 'rsi_14' in current and pd.notna(current['rsi_14']) and current['rsi_14'] > 70:
        scenario["risk_factors"].append(f"RSI(14) overbought at {current['rsi_14']:.1f}")
        
    # Adjust confidence based on number of supporting vs risk factors
    supporting_count = len(scenario["supporting_factors"])
    risk_count = len(scenario["risk_factors"])
    
    if supporting_count > 3 and risk_count <= 1:
        scenario["confidence"] = "high"
    elif supporting_count > 1:
        scenario["confidence"] = "medium"
    else:
        scenario["confidence"] = "low"
        
    # Create scenario summary
    try:
        scenario["scenario_summary"] = _create_bullish_summary(scenario, trend_data)
    except Exception as e:
        logger.warning(f"Error creating bullish summary: {str(e)}")
        scenario["scenario_summary"] = "Bullish scenario with " + scenario["confidence"] + " confidence"
    
    logger.debug(f"Bullish scenario probability: {scenario['probability']:.1%}, confidence: {scenario['confidence']}")
    
    # Open interest logic
    if open_interest and 'value' in open_interest and 'prev_value' in open_interest:
        oi_delta = open_interest['value'] - open_interest['prev_value']
        price_delta = data['close'].iloc[-1] - data['close'].iloc[-2]
        if oi_delta > 0 and price_delta > 0:
            scenario["probability"] = 0.7
            scenario["confidence"] = "high"
            scenario["supporting_factors"].append("Rising open interest with rising price confirms bullish scenario.")
        elif oi_delta < 0 and price_delta > 0:
            scenario["probability"] = 0.5
            scenario["confidence"] = "medium"
            scenario["risk_factors"].append("Falling open interest with rising price suggests short covering, possible reversal.")
        elif oi_delta == 0:
            scenario["probability"] = 0.4
            scenario["confidence"] = "medium"
            scenario["risk_factors"].append("Flat open interest, conviction lacking.")
    
    return scenario

def _analyze_bearish_scenario(data: pd.DataFrame, trend_data: Dict[str, Any], open_interest: dict = None) -> Dict[str, Any]:
    """
    Analyze the bearish market scenario.
    
    Args:
        data: DataFrame with market data and technical indicators
        trend_data: Dict with trend analysis results
        open_interest: Dict with open interest data (optional)
        
    Returns:
        Dict with bearish scenario analysis
        
    Raises:
        Exception: If analysis fails
    """
    logger.debug("Analyzing bearish scenario")
    
    # Validate inputs
    if data is None or data.empty or len(data) < 5:
        raise ValidationError("Insufficient data for bearish scenario analysis")
        
    if not isinstance(trend_data, dict):
        raise ValidationError("Invalid trend data format")
    
    # Initialize results
    scenario = {
        "probability": 0.3,  # Default middle probability
        "confidence": "low",
        "potential": "unknown",
        "supporting_factors": [],
        "risk_factors": [],
        "triggers": [],
        "targets": [],
        "scenario_summary": ""
    }
    
    # Extract the most recent data point
    current = data.iloc[-1]
    recent_data = data.tail(20)
    
    # Basic probability based on trend
    if trend_data.get('overall_trend') == 'bearish':
        if trend_data.get('confidence') == 'high':
            scenario["probability"] = 0.75
        elif trend_data.get('confidence') == 'medium':
            scenario["probability"] = 0.6
        else:
            scenario["probability"] = 0.45
    elif trend_data.get('overall_trend') == 'neutral':
        scenario["probability"] = 0.3
    else:  # bullish trend
        scenario["probability"] = 0.15
        
    # Check if price is below key moving averages
    ma_signals = []
    if 'close' in current and 'sma_20' in current and pd.notna(current['close']) and pd.notna(current['sma_20']) and current['close'] < current['sma_20']:
        ma_signals.append(("Price below SMA(20)", 0.05))
        
    if 'close' in current and 'sma_50' in current and pd.notna(current['close']) and pd.notna(current['sma_50']) and current['close'] < current['sma_50']:
        ma_signals.append(("Price below SMA(50)", 0.1))
        
    if 'close' in current and 'sma_200' in current and pd.notna(current['close']) and pd.notna(current['sma_200']) and current['close'] < current['sma_200']:
        ma_signals.append(("Price below SMA(200)", 0.15))
    
    # Check RSI for bearish signals
    rsi_signals = []
    if 'rsi_14' in current and pd.notna(current['rsi_14']):
        rsi = current['rsi_14']
        if 30 < rsi < 70:
            if rsi < 45:
                rsi_signals.append((f"RSI(14) showing weakness ({rsi:.1f})", 0.1))
            elif rsi < 55:
                rsi_signals.append((f"RSI(14) neutral to negative ({rsi:.1f})", 0.05))
            elif rsi > 65:
                rsi_signals.append((f"RSI(14) showing potential overbought reversal opportunity ({rsi:.1f})", 0.03))
    
    # Check MACD for bearish signals
    macd_signals = []
    if all(col in current for col in ['MACD_12_26_9', 'MACDs_12_26_9']):
        macd = current['MACD_12_26_9']
        signal = current['MACDs_12_26_9']
        if pd.notna(macd) and pd.notna(signal):
            if macd < signal:
                macd_signals.append(("MACD below signal line", 0.1))
            if macd < 0:
                macd_signals.append(("MACD below zero line", 0.05))
            elif macd_signals and macd > 0 and macd < signal:
                macd_signals.append(("MACD falling from above zero", 0.03))
    
    # Identify potential price targets (support levels)
    close = current['close'] if 'close' in current and pd.notna(current['close']) else None
    if close is not None:
        # Simple percentage-based targets
        scenario["targets"] = [
            {
                "level": round(close * 0.95, 2),
                "description": "Short-term support (5% below current)",
                "type": "support",
                "strength": "weak"
            },
            {
                "level": round(close * 0.9, 2),
                "description": "Medium-term target (10% below current)",
                "type": "target",
                "strength": "medium"
            },
            {
                "level": round(close * 0.8, 2),
                "description": "Long-term target (20% below current)",
                "type": "target", 
                "strength": "strong"
            }
        ]
        
        # Find potential trigger points
        if 'sma_20' in current and pd.notna(current['sma_20']):
            scenario["triggers"].append({
                "level": round(current['sma_20'], 2),
                "description": "Break above/below SMA(20)",
                "significance": "minor"
            })
        
        if 'sma_50' in current and pd.notna(current['sma_50']):
            scenario["triggers"].append({
                "level": round(current['sma_50'], 2),
                "description": "Break above/below SMA(50)",
                "significance": "moderate"
            })
        
        # Calculate potential downside
        if scenario["targets"]:
            worst_target = scenario["targets"][-1]["level"]
            scenario["potential"] = f"-{((1 - worst_target/close) * 100):.1f}%"
    
    # Collect all supporting factors and adjust probability
    for signal, prob_adjust in ma_signals + rsi_signals + macd_signals:
        scenario["supporting_factors"].append(signal)
        scenario["probability"] = min(0.95, scenario["probability"] + prob_adjust)
        
    # Collect risk factors
    if trend_data.get('overall_trend') == 'bullish':
        scenario["risk_factors"].append("Overall trend is bullish")
        
    if 'rsi_14' in current and pd.notna(current['rsi_14']) and current['rsi_14'] < 30:
        scenario["risk_factors"].append(f"RSI(14) oversold at {current['rsi_14']:.1f}")
        
    # Adjust confidence based on number of supporting vs risk factors
    supporting_count = len(scenario["supporting_factors"])
    risk_count = len(scenario["risk_factors"])
    
    if supporting_count > 3 and risk_count <= 1:
        scenario["confidence"] = "high"
    elif supporting_count > 1:
        scenario["confidence"] = "medium"
    else:
        scenario["confidence"] = "low"
        
    # Create scenario summary
    try:
        scenario["scenario_summary"] = _create_bearish_summary(scenario, trend_data)
    except Exception as e:
        logger.warning(f"Error creating bearish summary: {str(e)}")
        scenario["scenario_summary"] = "Bearish scenario with " + scenario["confidence"] + " confidence"
    
    logger.debug(f"Bearish scenario probability: {scenario['probability']:.1%}, confidence: {scenario['confidence']}")
    
    # Open interest logic
    if open_interest and 'value' in open_interest and 'prev_value' in open_interest:
        oi_delta = open_interest['value'] - open_interest['prev_value']
        price_delta = data['close'].iloc[-1] - data['close'].iloc[-2]
        if oi_delta > 0 and price_delta < 0:
            scenario["probability"] = 0.7
            scenario["confidence"] = "high"
            scenario["supporting_factors"].append("Rising open interest with falling price confirms bearish scenario.")
        elif oi_delta < 0 and price_delta < 0:
            scenario["probability"] = 0.5
            scenario["confidence"] = "medium"
            scenario["risk_factors"].append("Falling open interest with falling price suggests long liquidation, possible reversal.")
        elif oi_delta == 0:
            scenario["probability"] = 0.4
            scenario["confidence"] = "medium"
            scenario["risk_factors"].append("Flat open interest, conviction lacking.")
    
    return scenario

def _analyze_sideways_scenario(data: pd.DataFrame, trend_data: Dict[str, Any], open_interest: dict = None) -> Dict[str, Any]:
    """
    Analyze the sideways/neutral market scenario.
    
    Args:
        data: DataFrame with market data and technical indicators
        trend_data: Dict with trend analysis results
        open_interest: Dict with open interest data (optional)
        
    Returns:
        Dict with sideways scenario analysis
        
    Raises:
        Exception: If analysis fails
    """
    logger.debug("Analyzing sideways/neutral scenario")
    
    # Validate inputs
    if data is None or data.empty or len(data) < 5:
        raise ValidationError("Insufficient data for sideways scenario analysis")
        
    if not isinstance(trend_data, dict):
        raise ValidationError("Invalid trend data format")
    
    # Initialize results
    scenario = {
        "probability": 0.3,  # Default middle probability
        "confidence": "low",
        "potential": "unknown",
        "supporting_factors": [],
        "risk_factors": [],
        "triggers": [],
        "targets": [],
        "scenario_summary": ""
    }
    
    # Extract the most recent data point
    current = data.iloc[-1]
    recent_data = data.tail(20)
    
    # Basic probability based on trend
    if trend_data.get('overall_trend') == 'neutral':
        if trend_data.get('confidence') == 'high':
            scenario["probability"] = 0.75
        elif trend_data.get('confidence') == 'medium':
            scenario["probability"] = 0.6
        else:
            scenario["probability"] = 0.45
    else:  # bullish or bearish trend
        scenario["probability"] = 0.25
    
    # Calculate volatility
    try:
        if len(recent_data) >= 5 and 'close' in recent_data.columns:
            # Check for missing values
            close_data = recent_data['close'].dropna()
            if len(close_data) >= 5:
                returns = close_data.pct_change().dropna()
                volatility = returns.std() * 100  # as percentage
                
                # Lower volatility supports sideways scenario
                if volatility < 1.0:  # Less than 1% daily volatility
                    scenario["supporting_factors"].append(f"Low volatility ({volatility:.2f}%)")
                    scenario["probability"] = min(0.95, scenario["probability"] + 0.15)
                elif volatility < 2.0:  # Less than 2% daily volatility
                    scenario["supporting_factors"].append(f"Moderate volatility ({volatility:.2f}%)")
                    scenario["probability"] = min(0.95, scenario["probability"] + 0.05)
                else:
                    scenario["risk_factors"].append(f"High volatility ({volatility:.2f}%)")
    except Exception as e:
        logger.warning(f"Error calculating volatility: {str(e)}")
    
    # Check price relative to moving averages
    try:
        if all(col in current for col in ['close', 'sma_20', 'sma_50']) and all(pd.notna(current[col]) for col in ['close', 'sma_20', 'sma_50']):
            price = current['close']
            sma20 = current['sma_20']
            sma50 = current['sma_50']
            
            # Price between MA's or close to them supports sideways
            if abs(price/sma20 - 1) < 0.02 and abs(price/sma50 - 1) < 0.05:
                scenario["supporting_factors"].append("Price close to key moving averages")
                scenario["probability"] = min(0.95, scenario["probability"] + 0.1)
                
            # Flat moving averages support sideways
            if len(data) > 20:
                sma20_slope = (data['sma_20'].iloc[-1] / data['sma_20'].iloc[-10] - 1) * 100
                if abs(sma20_slope) < 2:  # less than 2% change over 10 periods
                    scenario["supporting_factors"].append(f"Flat SMA(20) slope ({sma20_slope:.2f}%)")
                    scenario["probability"] = min(0.95, scenario["probability"] + 0.1)
    except Exception as e:
        logger.warning(f"Error analyzing moving averages for sideways scenario: {str(e)}")
    
    # Check RSI for neutral signals
    try:
        if 'rsi_14' in current and pd.notna(current['rsi_14']):
            rsi = current['rsi_14']
            if 45 < rsi < 55:
                scenario["supporting_factors"].append(f"RSI(14) neutral at {rsi:.1f}")
                scenario["probability"] = min(0.95, scenario["probability"] + 0.1)
    except Exception as e:
        logger.warning(f"Error analyzing RSI for sideways scenario: {str(e)}")
    
    # Check MACD for neutral signals
    try:
        if all(col in current for col in ['MACD_12_26_9', 'MACDs_12_26_9']) and pd.notna(current['MACD_12_26_9']) and pd.notna(current['MACDs_12_26_9']):
            macd = current['MACD_12_26_9']
            signal = current['MACDs_12_26_9']
            # MACD close to zero and signal line
            if abs(macd) < 0.1 * current['close'] and abs(macd - signal) < 0.05 * current['close']:
                scenario["supporting_factors"].append("MACD close to zero and signal line")
                scenario["probability"] = min(0.95, scenario["probability"] + 0.1)
    except Exception as e:
        logger.warning(f"Error analyzing MACD for sideways scenario: {str(e)}")
    
    # Set price range for the sideways scenario
    close = current['close'] if 'close' in current and pd.notna(current['close']) else None
    if close is not None:
        try:
            # Calculate a range based on recent volatility or fixed percentage
            range_percent = 0.05  # default 5% range
            
            if 'close' in recent_data.columns:
                close_data = recent_data['close'].dropna()
                if len(close_data) >= 5:
                    volatility = close_data.pct_change().std()
                    range_percent = max(0.03, min(0.1, volatility * 2.5))  # 2.5x daily volatility, min 3%, max 10%
            
            upper_bound = round(close * (1 + range_percent), 2)
            lower_bound = round(close * (1 - range_percent), 2)
            
            # Use these as targets
            scenario["targets"] = [
                {
                    "level": upper_bound,
                    "description": f"Upper range ({range_percent*100:.1f}% above current)",
                    "type": "resistance",
                    "strength": "moderate"
                },
                {
                    "level": lower_bound,
                    "description": f"Lower range ({range_percent*100:.1f}% below current)",
                    "type": "support",
                    "strength": "moderate"
                }
            ]
            
            # Calculate expected return for sideways
            scenario["potential"] = "±{:.1f}%".format(range_percent * 100)
            
            # Breakout/breakdown levels as triggers
            scenario["triggers"] = [
                {
                    "level": upper_bound,
                    "description": "Break above upper range",
                    "significance": "high",
                    "outcome": "bullish breakout"
                },
                {
                    "level": lower_bound,
                    "description": "Break below lower range",
                    "significance": "high",
                    "outcome": "bearish breakdown"
                }
            ]
        except Exception as e:
            logger.warning(f"Error calculating price targets for sideways scenario: {str(e)}")
    
    # Collect risk factors
    if trend_data.get('overall_trend') in ['bullish', 'bearish']:
        if trend_data.get('confidence') in ['high', 'medium']:
            scenario["risk_factors"].append(f"Strong {trend_data.get('overall_trend')} trend")
    
    # Adjust confidence based on number of supporting vs risk factors
    supporting_count = len(scenario["supporting_factors"])
    risk_count = len(scenario["risk_factors"])
    
    if supporting_count > 2 and risk_count == 0:
        scenario["confidence"] = "high"
    elif supporting_count > 1:
        scenario["confidence"] = "medium"
    else:
        scenario["confidence"] = "low"
        
    # Create scenario summary
    try:
        scenario["scenario_summary"] = _create_sideways_summary(scenario, trend_data)
    except Exception as e:
        logger.warning(f"Error creating sideways summary: {str(e)}")
        scenario["scenario_summary"] = "Sideways scenario with " + scenario["confidence"] + " confidence"
    
    logger.debug(f"Sideways scenario probability: {scenario['probability']:.1%}, confidence: {scenario['confidence']}")
    
    # Open interest logic
    if open_interest and 'value' in open_interest and 'prev_value' in open_interest:
        oi_delta = open_interest['value'] - open_interest['prev_value']
        if oi_delta == 0:
            scenario["probability"] = 0.7
            scenario["confidence"] = "high"
            scenario["supporting_factors"].append("Flat open interest supports neutral/sideways scenario.")
    
    return scenario

def _create_bullish_summary(scenario: Dict[str, Any], trend_data: Dict[str, Any]) -> str:
    """
    Create a text summary for the bullish scenario.
    
    Args:
        scenario: Dict with bullish scenario analysis
        trend_data: Dict with trend analysis results
        
    Returns:
        String with scenario summary
    """
    try:
        confidence = scenario["confidence"]
        factors = scenario["supporting_factors"]
        targets = scenario["targets"]
        potential = scenario["potential"]
        
        # Start with confidence level
        if confidence == "high":
            summary = "Strong bullish scenario"
        elif confidence == "medium":
            summary = "Moderate bullish scenario"
        else:
            summary = "Tentative bullish scenario"
            
        # Add some context from supporting factors
        if factors:
            summary += " supported by " + factors[0].lower()
            if len(factors) > 1:
                summary += f" and {len(factors)-1} other factors"
                
        # Add potential upside
        if potential != "unknown":
            summary += f". Potential upside of {potential}"
            
        # Add a target if available
        if targets:
            summary += f" with key target at {targets[0]['level']}"
            
        return summary
    except Exception as e:
        logger.warning(f"Error in _create_bullish_summary: {str(e)}")
        return "Bullish scenario"

def _create_bearish_summary(scenario: Dict[str, Any], trend_data: Dict[str, Any]) -> str:
    """
    Create a text summary for the bearish scenario.
    
    Args:
        scenario: Dict with bearish scenario analysis
        trend_data: Dict with trend analysis results
        
    Returns:
        String with scenario summary
    """
    try:
        confidence = scenario["confidence"]
        factors = scenario["supporting_factors"]
        targets = scenario["targets"]
        potential = scenario["potential"]
        
        # Start with confidence level
        if confidence == "high":
            summary = "Strong bearish scenario"
        elif confidence == "medium":
            summary = "Moderate bearish scenario"
        else:
            summary = "Tentative bearish scenario"
            
        # Add some context from supporting factors
        if factors:
            summary += " supported by " + factors[0].lower()
            if len(factors) > 1:
                summary += f" and {len(factors)-1} other factors"
                
        # Add potential upside
        if potential != "unknown":
            summary += f". Potential downside of {potential}"
            
        # Add a target if available
        if targets:
            summary += f" with key support at {targets[0]['level']}"
            
        return summary
    except Exception as e:
        logger.warning(f"Error in _create_bearish_summary: {str(e)}")
        return "Bearish scenario"

def _create_sideways_summary(scenario: Dict[str, Any], trend_data: Dict[str, Any]) -> str:
    """
    Create a text summary for the sideways scenario.
    
    Args:
        scenario: Dict with sideways scenario analysis
        trend_data: Dict with trend analysis results
        
    Returns:
        String with scenario summary
    """
    try:
        confidence = scenario["confidence"]
        factors = scenario["supporting_factors"]
        targets = scenario["targets"]
        potential = scenario["potential"]
        
        # Start with confidence level
        if confidence == "high":
            summary = "Strong consolidation scenario"
        elif confidence == "medium":
            summary = "Likely consolidation scenario"
        else:
            summary = "Possible consolidation scenario"
            
        # Add some context from supporting factors
        if factors:
            summary += " supported by " + factors[0].lower()
            if len(factors) > 1:
                summary += f" and {len(factors)-1} other factors"
                
        # Add expected range
        if potential != "unknown":
            summary += f". Expected trading range of {potential}"
            
        # Add range bounds if available
        if len(targets) >= 2:
            summary += f" between {targets[0]['level']} and {targets[1]['level']}"
            
        return summary
    except Exception as e:
        logger.warning(f"Error in _create_sideways_summary: {str(e)}")
        return "Sideways/consolidation scenario" 