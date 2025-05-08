"""
API Configuration Module

This module handles the secure loading, storage, and retrieval of API credentials
for various exchanges and services used in the application.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "api_keys.json"
DEFAULT_EXCHANGE = "binance"

# Supported exchanges
SUPPORTED_EXCHANGES = [
    "binance",
    "bybit",
    "coinbase",
    "kraken",
    "kucoin",
    "ftx"
]

# Ensure config directory exists
CONFIG_DIR.mkdir(exist_ok=True)

def load_from_env(exchange: str = DEFAULT_EXCHANGE) -> Dict[str, str]:
    """
    Load API credentials from environment variables.
    
    Args:
        exchange: The exchange to load credentials for
        
    Returns:
        Dict containing API key and secret
    """
    # Try to load .env file if it exists
    dotenv_path = Path(".env")
    if dotenv_path.exists():
        dotenv.load_dotenv(dotenv_path)
    
    exchange = exchange.upper()
    api_key = os.environ.get(f"{exchange}_API_KEY")
    api_secret = os.environ.get(f"{exchange}_API_SECRET")
    
    if not api_key or not api_secret:
        logger.warning(f"Missing API credentials for {exchange} in environment variables")
        return {}
        
    return {
        "api_key": api_key,
        "api_secret": api_secret
    }

def load_from_file(exchange: str = DEFAULT_EXCHANGE) -> Dict[str, str]:
    """
    Load API credentials from config file.
    
    Args:
        exchange: The exchange to load credentials for
        
    Returns:
        Dict containing API key and secret
    """
    if not CONFIG_FILE.exists():
        logger.warning(f"Config file {CONFIG_FILE} does not exist")
        return {}
        
    try:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
            
        if exchange in config:
            return config[exchange]
        else:
            logger.warning(f"No configuration found for {exchange} in config file")
            return {}
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        return {}

def save_to_file(exchange: str, api_key: str, api_secret: str) -> bool:
    """
    Save API credentials to config file.
    
    Args:
        exchange: The exchange to save credentials for
        api_key: The API key
        api_secret: The API secret
        
    Returns:
        Whether the save was successful
    """
    # Ensure exchange is valid
    exchange = exchange.lower()
    if exchange not in SUPPORTED_EXCHANGES:
        logger.error(f"Unsupported exchange: {exchange}")
        return False
    
    # Load existing config or create new
    config = {}
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Error reading existing config: {e}")
            # Continue with empty config
    
    # Update config
    config[exchange] = {
        "api_key": api_key,
        "api_secret": api_secret
    }
    
    # Save config
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved API credentials for {exchange}")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

def get_api_credentials(exchange: str = DEFAULT_EXCHANGE) -> Dict[str, str]:
    """
    Get API credentials for the specified exchange, trying environment variables first,
    then falling back to config file.
    
    Args:
        exchange: The exchange to get credentials for
        
    Returns:
        Dict containing API key and secret
    """
    # Normalize exchange name
    exchange = exchange.lower()
    if exchange not in SUPPORTED_EXCHANGES:
        logger.warning(f"Unsupported exchange: {exchange}, falling back to {DEFAULT_EXCHANGE}")
        exchange = DEFAULT_EXCHANGE
    
    # Try environment variables first
    credentials = load_from_env(exchange)
    if credentials:
        logger.info(f"Using {exchange} API credentials from environment variables")
        return credentials
        
    # Fall back to config file
    credentials = load_from_file(exchange)
    if credentials:
        logger.info(f"Using {exchange} API credentials from config file")
        return credentials
        
    # No credentials found
    logger.warning(f"No API credentials found for {exchange}")
    return {}

def validate_credentials(credentials: Dict[str, str]) -> bool:
    """
    Check if credentials are valid (contain necessary fields).
    
    Args:
        credentials: The credentials to validate
        
    Returns:
        Whether the credentials are valid
    """
    return bool(credentials and "api_key" in credentials and "api_secret" in credentials)

def setup_cli_credentials(exchange: str, api_key: str, api_secret: str) -> bool:
    """
    Set up API credentials via CLI.
    
    Args:
        exchange: The exchange to set up credentials for
        api_key: The API key
        api_secret: The API secret
        
    Returns:
        Whether the setup was successful
    """
    if not exchange or not api_key or not api_secret:
        logger.error("Missing required parameters")
        return False
        
    return save_to_file(exchange, api_key, api_secret)

def mask_credentials(credentials: Dict[str, str]) -> Dict[str, str]:
    """
    Mask API credentials for logging purposes.
    
    Args:
        credentials: The credentials to mask
        
    Returns:
        Dict with masked values
    """
    if not credentials:
        return {}
        
    masked = {}
    for key, value in credentials.items():
        if key in ["api_key", "api_secret"] and value:
            # Show first 4 and last 4 characters
            if len(value) > 8:
                masked[key] = value[:4] + "*" * (len(value) - 8) + value[-4:]
            else:
                masked[key] = "****"
        else:
            masked[key] = value
            
    return masked 