"""
Error handling for market analysis modules.

This module provides custom exceptions and utilities for validating inputs
and handling errors consistently across market analysis modules.
"""

from typing import List, Any, Optional, Union, Callable, Tuple
import pandas as pd
from src.config.logging_config import get_logger

# Set up logger
logger = get_logger(__name__)

# Base exception for all market analysis errors
class MarketAnalysisError(Exception):
    """Base exception for all market analysis errors."""
    pass

# For errors related to market data operations
class MarketDataError(MarketAnalysisError):
    """Base exception for market data errors."""
    pass

# For errors when fetching market data
class DataFetchError(MarketDataError):
    """Raised when there's an error fetching market data."""
    pass

# For errors with data format or structure
class DataFormatError(MarketDataError):
    """Raised when data doesn't conform to expected format."""
    pass

# For errors when calculating indicators
class IndicatorError(MarketAnalysisError):
    """Raised when there's an error calculating technical indicators."""
    pass

# For input validation errors
class ValidationError(MarketAnalysisError):
    """Raised when input validation fails."""
    pass 

def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None, min_rows: int = 1) -> bool:
    """
    Validate that a DataFrame meets minimum requirements.
    
    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        min_rows: Minimum number of rows required
        
    Raises:
        ValidationError: If validation fails
        
    Returns:
        bool: True if validation passes
    """
    if df is None:
        logger.error("DataFrame validation failed: DataFrame is None")
        raise ValidationError("DataFrame is None")
    
    if df.empty:
        logger.error("DataFrame validation failed: DataFrame is empty")
        raise ValidationError("DataFrame is empty")
        
    if len(df) < min_rows:
        logger.error(f"DataFrame validation failed: insufficient rows. Required: {min_rows}, Found: {len(df)}")
        raise ValidationError(f"DataFrame has insufficient rows. Required: {min_rows}, Found: {len(df)}")
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"DataFrame validation failed: missing columns: {', '.join(missing_columns)}")
            raise ValidationError(f"Missing required columns: {', '.join(missing_columns)}")
            
    logger.debug("DataFrame validation passed")
    return True

def validate_symbol(symbol: str) -> bool:
    """
    Validate that a symbol is properly formatted.
    
    Args:
        symbol: Trading symbol to validate
        
    Raises:
        ValidationError: If validation fails
        
    Returns:
        bool: True if validation passes
    """
    if not symbol:
        logger.error("Symbol validation failed: Symbol is empty")
        raise ValidationError("Symbol cannot be empty")
        
    if not isinstance(symbol, str):
        logger.error(f"Symbol validation failed: Symbol must be a string, got {type(symbol)}")
        raise ValidationError(f"Symbol must be a string, got {type(symbol)}")
    
    # Allow OKX/ccxt swap symbols (e.g., 'BTC/USD:BTC')
    if '/' in symbol and ':' in symbol:
        # Basic check: must be in the form BASE/USD:BASE (or similar)
        parts = symbol.split('/')
        if len(parts) == 2 and ':' in parts[1]:
            base, right = parts
            quote, base2 = right.split(':', 1)
            if base.isalnum() and quote.isalnum() and base2.isalnum():
                logger.debug(f"Symbol validation passed for OKX/ccxt swap symbol: {symbol}")
                return True
        logger.error(f"Symbol validation failed: Invalid OKX/ccxt swap symbol format: {symbol}")
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")
    
    # Basic symbol format validation - can be expanded later
    if not all(c.isalnum() or c in '.-_' for c in symbol):
        logger.error(f"Symbol validation failed: Symbol contains invalid characters: {symbol}")
        raise ValidationError(f"Symbol contains invalid characters: {symbol}")
        
    logger.debug(f"Symbol validation passed for {symbol}")
    return True

def validate_timeframe(timeframe: str, valid_timeframes: Tuple[str, ...] = ('short', 'medium', 'long')) -> bool:
    """
    Validate that a timeframe is one of the valid options.
    
    Args:
        timeframe: Timeframe to validate
        valid_timeframes: Tuple of valid timeframe options
        
    Raises:
        ValidationError: If validation fails
        
    Returns:
        bool: True if validation passes
    """
    if not timeframe:
        logger.error("Timeframe validation failed: Timeframe is empty")
        raise ValidationError("Timeframe cannot be empty")
        
    if not isinstance(timeframe, str):
        logger.error(f"Timeframe validation failed: Timeframe must be a string, got {type(timeframe)}")
        raise ValidationError(f"Timeframe must be a string, got {type(timeframe)}")
        
    if timeframe.lower() not in valid_timeframes:
        logger.error(f"Timeframe validation failed: Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
        raise ValidationError(f"Invalid timeframe: {timeframe}. Must be one of {valid_timeframes}")
        
    logger.debug(f"Timeframe validation passed for {timeframe}")
    return True

def validate_numeric_param(param: Any, name: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> bool:
    """
    Validate that a parameter is numeric and within specified range.
    
    Args:
        param: Parameter to validate
        name: Name of the parameter (for error messages)
        min_value: Minimum allowed value (optional)
        max_value: Maximum allowed value (optional)
        
    Raises:
        ValidationError: If validation fails
        
    Returns:
        bool: True if validation passes
    """
    try:
        param_value = float(param)
    except (ValueError, TypeError):
        logger.error(f"Parameter validation failed: Parameter '{name}' must be numeric, got {type(param)}")
        raise ValidationError(f"Parameter '{name}' must be numeric, got {type(param)}")
        
    if min_value is not None and param_value < min_value:
        logger.error(f"Parameter validation failed: Parameter '{name}' must be at least {min_value}, got {param_value}")
        raise ValidationError(f"Parameter '{name}' must be at least {min_value}, got {param_value}")
        
    if max_value is not None and param_value > max_value:
        logger.error(f"Parameter validation failed: Parameter '{name}' must be at most {max_value}, got {param_value}")
        raise ValidationError(f"Parameter '{name}' must be at most {max_value}, got {param_value}")
        
    logger.debug(f"Parameter validation passed for {name}={param_value}")
    return True

def safe_operation(operation: Callable, 
                  fallback: Any = None, 
                  exceptions: Tuple[Exception, ...] = (Exception,),
                  error_msg: Optional[str] = None, 
                  logger_instance = None) -> Any:
    """
    Execute an operation safely with fallback if it fails.
    
    Args:
        operation: Function to execute
        fallback: Value to return if operation fails
        exceptions: Tuple of exceptions to catch
        error_msg: Error message to log
        logger_instance: Logger instance (uses module logger if None)
        
    Returns:
        Result of operation or fallback value
    """
    log = logger_instance or logger
    try:
        return operation()
    except exceptions as e:
        log.error(error_msg or f"Operation failed: {str(e)}", exc_info=True)
        return fallback

def safe_dataframe_operation(df: pd.DataFrame, 
                            operation: Callable[[pd.DataFrame], Any],
                            required_columns: Optional[List[str]] = None,
                            fallback: Any = None,
                            exceptions: Tuple[Exception, ...] = (Exception,),
                            error_msg: Optional[str] = None,
                            logger_instance = None) -> Any:
    """
    Execute a DataFrame operation safely with validation and fallback.
    
    Args:
        df: DataFrame to operate on
        operation: Function that takes DataFrame as input
        required_columns: List of columns that must be present
        fallback: Value to return if operation fails
        exceptions: Tuple of exceptions to catch
        error_msg: Error message to log
        logger_instance: Logger instance (uses module logger if None)
        
    Returns:
        Result of operation or fallback value
    """
    log = logger_instance or logger
    try:
        # Validate DataFrame first
        validate_dataframe(df, required_columns=required_columns)
        
        # If validation passes, execute the operation
        return operation(df)
    except ValidationError as e:
        # Specific handling for validation errors
        log.error(f"Validation error before operation: {str(e)}")
        return fallback
    except exceptions as e:
        # Handling for other exceptions during operation
        log.error(error_msg or f"DataFrame operation failed: {str(e)}", exc_info=True)
        return fallback 