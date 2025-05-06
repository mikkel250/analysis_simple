"""Price validation utilities for cryptocurrency data"""

import logging
from typing import Dict, Optional, Union, Any
import math

# Configure logging
logger = logging.getLogger(__name__)

# Store last valid prices
_last_valid_prices: Dict[str, float] = {}

def validate_crypto_price(symbol: str, price: float) -> float:
    """
    Validate cryptocurrency price is within reasonable bounds for USD.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        price: Price value to validate
        
    Returns:
        Validated price (original if valid, fallback if invalid)
    """
    # Check for invalid input types
    if not isinstance(price, (int, float)):
        logger.error(f"Invalid price type for {symbol}: {type(price)}, defaulting to fallback")
        return get_last_valid_price(symbol) or 50000  # Default fallback for BTC
    
    # Check for NaN or infinite values
    if math.isnan(price) or math.isinf(price):
        logger.error(f"Invalid price value for {symbol}: {price}, defaulting to fallback")
        return get_last_valid_price(symbol) or 50000  # Default fallback for BTC
    
    # Different ranges for different cryptocurrencies
    price_ranges = {
        'BTC': (10000, 100000),  # Bitcoin: $10k-$100k
        'ETH': (1000, 10000),    # Ethereum: $1k-$10k
        'SOL': (20, 500),        # Solana: $20-$500
        'ADA': (0.1, 5),         # Cardano: $0.1-$5
        'DOT': (5, 100),         # Polkadot: $5-$100
        'AVAX': (5, 200),        # Avalanche: $5-$200
        'LINK': (5, 100),        # Chainlink: $5-$100
        'UNI': (1, 50),          # Uniswap: $1-$50
        'MATIC': (0.3, 5),       # Polygon: $0.3-$5
        'DOGE': (0.05, 1),       # Dogecoin: $0.05-$1
        'SHIB': (0.00001, 0.001) # Shiba Inu: very low price range
    }
    
    # Use default range if symbol not found
    min_price, max_price = price_ranges.get(symbol.upper(), (0, float('inf')))
    
    if price < min_price or price > max_price:
        logger.warning(f"{symbol} price ${price} outside valid range (${min_price}-${max_price})")
        # Return previous valid price or fallback
        return get_last_valid_price(symbol) or min_price * 5  # Fallback to middle of range
    
    # Store this as a valid price
    store_valid_price(symbol, price)
    return price

def store_valid_price(symbol: str, price: float) -> None:
    """
    Store a validated price for potential future fallback.
    
    Args:
        symbol: Cryptocurrency symbol
        price: Valid price to store
    """
    global _last_valid_prices
    _last_valid_prices[symbol.upper()] = price
    
def get_last_valid_price(symbol: str) -> Optional[float]:
    """
    Get the last known valid price for a symbol.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        Last valid price or None if not available
    """
    return _last_valid_prices.get(symbol.upper())

def clear_price_cache() -> None:
    """Clear the cached valid prices."""
    global _last_valid_prices
    _last_valid_prices.clear()
    logger.debug("Price validation cache cleared") 