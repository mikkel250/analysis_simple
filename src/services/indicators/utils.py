"""
Utility functions for technical indicators calculations.

This module provides common utility functions that are used across
different indicator calculations.
"""

import logging
from typing import Dict, Any, Union

import pandas as pd
import numpy as np

# Fix for pandas-ta compatibility with newer numpy versions
# The module tries to import NaN from numpy but newer versions use np.nan
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"DataFrame missing required columns: {', '.join(missing_columns)}")
    
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
    # Convert to dictionary with timestamps as keys
    if isinstance(values, pd.Series):
        result_dict = {
            str(idx): round(val, 6) if not pd.isna(val) else None 
            for idx, val in values.items()
        }
    elif isinstance(values, pd.DataFrame):
        # For indicators that return multiple series (like MACD)
        result_dict = {}
        for col in values.columns:
            result_dict[col] = {
                str(idx): round(val, 6) if not pd.isna(val) else None 
                for idx, val in values[col].items()
            }
    else:
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
    
    return response 