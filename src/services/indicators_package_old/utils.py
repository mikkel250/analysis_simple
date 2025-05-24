"""
Utility functions for technical indicators calculations.

This module provides common utility functions that are used across
different indicator calculations.
"""

from typing import Dict, Any, Union

import pandas as pd
import numpy as np

# Fix for pandas-ta compatibility with newer numpy versions
# The module tries to import NaN from numpy but newer versions use np.nan
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Import the centralized logging configuration
from src.config.logging_config import get_logger

# Configure logger for this module
logger = get_logger(__name__)


def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate if the DataFrame has the required columns for technical analysis.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        bool: True if DataFrame is valid, False otherwise
        
    Raises:
        ValueError: If DataFrame is missing required columns
    """
    logger.debug("Validating DataFrame for technical analysis")
    
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"DataFrame validation failed: missing columns: {', '.join(missing_columns)}")
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
    
    if df.empty:
        logger.warning("Validated DataFrame is empty")
    else:
        logger.debug(f"DataFrame validated successfully: {len(df)} rows, columns: {list(df.columns)}")
    
    return True


def format_indicator_response(
    indicator_name: str,
    values: Union[pd.Series, pd.DataFrame],
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Format indicator values to match the expected API response structure.
    
    Args:
        indicator_name: Name of the indicator
        values: Calculated indicator values
        params: Parameters used for the calculation
        
    Returns:
        Dict: Formatted response with indicator values and metadata
    """
    logger.debug(f"Formatting response for indicator: {indicator_name}")
    
    # Convert to dictionary with timestamps as keys
    if isinstance(values, pd.Series):
        logger.debug(f"Formatting Series with {len(values)} values")
        result_dict = {
            str(idx): round(val, 6) if not pd.isna(val) else None 
            for idx, val in values.items()
        }
    elif isinstance(values, pd.DataFrame):
        # For indicators that return multiple series (like MACD)
        logger.debug(f"Formatting DataFrame with {len(values)} rows and {len(values.columns)} columns")
        result_dict = {}
        for col in values.columns:
            result_dict[col] = {
                str(idx): round(val, 6) if not pd.isna(val) else None 
                for idx, val in values[col].items()
            }
    else:
        logger.warning(f"Unknown values type for indicator {indicator_name}: {type(values)}")
        result_dict = {}
    
    # Create response structure
    response = {
        "indicator": indicator_name,
        "values": result_dict,
        "metadata": {
            "params": params or {},
            "count": len(values) if hasattr(values, "__len__") else 0,
            "indicator_name": indicator_name,
            "calculation_time": pd.Timestamp.now().isoformat(),
        }
    }
    
    logger.debug(f"Successfully formatted response for {indicator_name} with {response['metadata']['count']} values")
    return response 