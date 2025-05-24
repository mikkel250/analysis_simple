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
    "ftx",
    "okx",
    "coingecko"
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
    
    # Special handling for OKX which uses a different variable format
    if exchange.lower() == "okx":
        # Standard ccxt names for environment variables
        api_key_env = os.environ.get("OKX_API_KEY")
        api_secret_env = os.environ.get("OKX_SECRET_KEY")
        api_password_env = os.environ.get("OKX_PASSWORD") # Passphrase

        if api_key_env and api_secret_env:
            logger.info(f"Found OKX API credentials (OKX_API_KEY, OKX_SECRET_KEY) in environment variables.")
            creds_for_ccxt = {
                "apiKey": api_key_env, # ccxt expects apiKey
                "secret": api_secret_env, # ccxt expects secret
            }
            if api_password_env:
                creds_for_ccxt["password"] = api_password_env # ccxt expects password for passphrase
                logger.info(f"Found OKX_PASSWORD in environment variables.")
            return creds_for_ccxt
        else:
            # Fallback to old lowercase names if new ones are not found, for backward compatibility during transition
            # This part can be removed later once .env files are updated.
            logger.debug("OKX_API_KEY/SECRET_KEY not found, trying legacy 'apikey'/'secretkey'")
            api_key_legacy = os.environ.get("apikey")
            api_secret_legacy = os.environ.get("secretkey")
            if api_key_legacy and api_secret_legacy:
                logger.info(f"Found OKX API credentials (legacy 'apikey', 'secretkey') in environment variables.")
                # Strip quotes if present (e.g. apikey='key' -> key) - this was from old logic
                if api_key_legacy.startswith("'") and api_key_legacy.endswith("'"):
                    api_key_legacy = api_key_legacy[1:-1]
                if api_secret_legacy.startswith("'") and api_secret_legacy.endswith("'"):
                    api_secret_legacy = api_secret_legacy[1:-1]
                return {
                    "apiKey": api_key_legacy,
                    "secret": api_secret_legacy,
                    # No password handling for legacy names, assume it wasn't used with them
                }
            logger.warning("No OKX API credentials found in environment variables (tried OKX_API_KEY/SECRET & legacy apikey/secretkey).")
            return {}
    
    # Standard format for other exchanges
    exchange_upper = exchange.upper()
    api_key = os.environ.get(f"{exchange_upper}_API_KEY")
    api_secret = None
    if exchange.lower() != "coingecko":
        api_secret = os.environ.get(f"{exchange_upper}_API_SECRET")
    
    if not api_key or (exchange.lower() != "coingecko" and not api_secret):
        logger.warning(f"Missing API credentials for {exchange} in environment variables")
        return {}
    
    creds = {"api_key": api_key}
    if api_secret:
        creds["api_secret"] = api_secret
    return creds

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
            if exchange.lower() == "coingecko" and "api_key" in config[exchange]:
                return {"api_key": config[exchange]["api_key"]}
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
        Dict containing API key and secret (using 'apiKey', 'secret', 'password' for ccxt)
    """
    # Normalize exchange name
    exchange_lower = exchange.lower()
    # We don't need to check SUPPORTED_EXCHANGES here as load_from_env handles it
    # and ccxt will error out if the exchange name is truly invalid.

    # Try environment variables first
    # load_from_env is already designed to return the ccxt-expected format for OKX
    # and a more generic format for others (which might need adjustment if we standardize all to ccxt)
    credentials = load_from_env(exchange_lower) 

    # Check if credentials were successfully loaded from environment
    # For OKX (and ccxt in general), we expect 'apiKey' and 'secret'
    expected_key_field = 'apiKey' if exchange_lower == 'okx' else 'api_key'

    if credentials and credentials.get(expected_key_field):
        logger.info(f"Using {exchange_lower} API credentials from environment variables.")
        return credentials
        
    # Fall back to config file
    # load_from_file might also need adjustment if it doesn't return ccxt format for OKX
    logger.debug(f"Credentials not found or incomplete in environment for {exchange_lower}, trying config file.")
    credentials_file = load_from_file(exchange_lower)
    
    if credentials_file and credentials_file.get(expected_key_field): # Check using the same expected key field
        logger.info(f"Using {exchange_lower} API credentials from config file.")
        # If OKX and from file, ensure it's in ccxt format.
        # This part needs care if file format is different from env format.
        # For now, assuming load_from_file for OKX would also be ccxt-like if intended for ccxt.
        return credentials_file
        
    # No credentials found
    logger.warning(f"No API credentials found for {exchange_lower} in environment or config file.")
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
        if key in ["api_key", "api_secret"] and isinstance(value, str) and value:
            # Show first 4 and last 4 characters
            if len(value) > 8:
                masked[key] = value[:4] + "*" * (len(value) - 8) + value[-4:]
            else:
                masked[key] = "****"
        else:
            masked[key] = value
            
    return masked 