"""
Data Fetcher Module

This module handles fetching cryptocurrency market data using CCXT for the OKX exchange
and converting it to pandas DataFrame format suitable for technical analysis.
"""

import time
import ccxt
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Optional, Union, Any, List

# Import config loader
from src.config.api_config import get_api_credentials, mask_credentials

# Import cache_service for caching price data
from .cache_service import (
    store_dataframe,
    get_cached_dataframe,
    store_json_data,
    get_cached_json_data,
)

import logging
logger = logging.getLogger(__name__)

# CCXT Timeframe mapping
CCXT_TIMEFRAMES = {
    '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '12h': '12h',
    '1d': '1d', '1w': '1w'
    # OKX also supports '1M', '3M', '6M', '1Y' - can be added if needed
}

class DataFetcher:
    """
    A class to fetch cryptocurrency data from OKX API via CCXT.
    """

    def __init__(self, exchange_name: str = "okx"):
        """Initialize the DataFetcher with CCXT targeting OKX."""
        logger.debug(f"Initializing DataFetcher for {exchange_name}...")
        self.exchange_name = exchange_name.lower()
        
        credentials = get_api_credentials(self.exchange_name)
        
        if not credentials.get('apiKey') or not credentials.get('secret'):
            logger.warning(
                f"API key or secret not found for {self.exchange_name}. "
                "Proceeding with unauthenticated access (rate limits will be stricter)."
            )
            # Initialize without authentication
            self.exchange = getattr(ccxt, self.exchange_name)()
        else:
            logger.info(f"Initializing {self.exchange_name} with API credentials.")
            logger.debug(f"Credentials for {self.exchange_name}: {mask_credentials(credentials)}")
            self.exchange = getattr(ccxt, self.exchange_name)(credentials)

        self.exchange.enableRateLimit = True  # CCXT handles rate limiting
        # OKX default rate limit: 20 requests/2 seconds for public, 60/2s for private V5 endpoints
        # fetchOHLCV and fetchTicker are typically public but ccxt might use private if authenticated
        logger.debug(
            f"CCXT rate limiting enabled for {self.exchange_name}. "
            f"Default sleep: {self.exchange.rateLimit / 1000}s (may vary by endpoint)"
        )

    def _normalize_symbol_for_ccxt(self, symbol: str) -> str:
        """Converts common symbol formats (e.g., BTC-USDT, btcusdt) to CCXT format (BTC/USDT)."""
        s = symbol.upper().replace('-', '/')
        if '/' not in s and len(s) > 3: # Attempt to split if no separator, e.g., BTCUSDT -> BTC/USDT
            common_bases = ['BTC', 'ETH', 'SOL', 'XRP'] # Add more if needed
            for base in common_bases:
                if s.startswith(base):
                    s = base + '/' + s[len(base):]
                    break
        logger.debug(f"Normalized symbol {symbol} to {s} for CCXT.")
        return s

    def _validate_timeframe(self, timeframe: str) -> str:
        """Validates and returns a CCXT-compatible timeframe string."""
        if timeframe in CCXT_TIMEFRAMES:
            return CCXT_TIMEFRAMES[timeframe]
        logger.warning(f"Invalid timeframe '{timeframe}'. Supported: {list(CCXT_TIMEFRAMES.keys())}. Defaulting to '1d'.")
        return '1d'

    def fetch_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100,
        since: Optional[int] = None, # timestamp in milliseconds
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a specified symbol and timeframe from OKX.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'BTC-USDT').
            timeframe: Timeframe string (e.g., '1d', '1h', '5m').
            limit: Number of candles to fetch (default: 100, OKX max varies, often 100-300 per call).
            since: Start time in milliseconds (optional).
            use_cache: Whether to use cached data if available (default: True).

        Returns:
            Pandas DataFrame with [timestamp, open, high, low, close, volume] and timestamp as index.
            Returns empty DataFrame on error or if no data.
        """
        normalized_symbol = self._normalize_symbol_for_ccxt(symbol)
        ccxt_timeframe = self._validate_timeframe(timeframe)
        
        # Adjust cache key to be more specific
        cache_key_parts = [
            "ohlcv", self.exchange_name, normalized_symbol.replace('/', '_'), 
            ccxt_timeframe, str(limit)
        ]
        if since:
            cache_key_parts.append(str(since))
        cache_key = "_".join(cache_key_parts)
        
        logger.info(
            f"Fetching historical OHLCV for {normalized_symbol}, timeframe: {ccxt_timeframe}, "
            f"limit: {limit}, since: {since}, use_cache: {use_cache}"
        )
        logger.debug(f"Cache key: {cache_key}")

        if use_cache:
            cached_df = get_cached_dataframe(cache_key)
            if cached_df is not None:
                logger.info(f"Cache hit for {cache_key}. Returning cached data.")
                return cached_df
            logger.debug(f"Cache miss for {cache_key}.")

        try:
            if not self.exchange.has['fetchOHLCV']:
                logger.error(f"{self.exchange_name} does not support fetchOHLCV via CCXT.")
                return pd.DataFrame()

            logger.debug(
                f"Calling CCXT: fetch_ohlcv for {normalized_symbol}, {ccxt_timeframe}, since={since}, limit={limit}"
            )
            
            params = {} # Placeholder for any exchange-specific params if needed in future
            ohlcv_data = self.exchange.fetch_ohlcv(
                normalized_symbol,
                timeframe=ccxt_timeframe,
                since=since,
                limit=limit,
                params=params
            )

            if not ohlcv_data:
                logger.warning(f"No OHLCV data returned from {self.exchange_name} for {normalized_symbol}, {ccxt_timeframe}.")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            df.set_index('timestamp', inplace=True)
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True) # Drop rows where conversion failed

            logger.debug(f"Processed DataFrame head for {normalized_symbol}:\n{df.head()}")

            if not df.empty and use_cache:
                metadata = {
                    'symbol': normalized_symbol,
                    'timeframe': ccxt_timeframe,
                    'limit': limit,
                    'since': since,
                    'source': self.exchange_name,
                    'endpoint': 'fetchOHLCV'
                }
                # TTL for historical data can be longer, e.g., 1 hour (3600s) or more
                # For frequently updated candles (e.g. 1m, 5m), TTL might be shorter
                # Default TTL in store_dataframe is 1 day if not specified by cache_service
                store_dataframe(cache_key, df, metadata=metadata, timeframe=ccxt_timeframe, ttl=3600) 
            return df

        except ccxt.NetworkError as e:
            logger.error(f"CCXT NetworkError fetching OHLCV for {normalized_symbol}: {str(e)}", exc_info=True)
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT ExchangeError fetching OHLCV for {normalized_symbol}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error fetching OHLCV for {normalized_symbol}: {str(e)}", exc_info=True)
        
        return pd.DataFrame() # Return empty DataFrame on error

    def get_current_price_data(
        self,
        symbol: str,
        use_cache: bool = True,
        cache_ttl_seconds: int = 300 # 5 minutes for current price
    ) -> Dict[str, Any]:
        """
        Fetch current price data for a symbol from OKX.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT', 'BTC-USDT').
            use_cache: Whether to use cached data.
            cache_ttl_seconds: TTL for the cache.

        Returns:
            Dictionary with current price information or empty dict on error.
            Example structure:
            {
                'symbol': 'BTC/USDT',
                'current_price': 60000.0,
                'price_change_percentage_24h': 1.5, (if available)
                'volume_24h': 10000.0, (if available)
                'last_updated': 'YYYY-MM-DDTHH:MM:SS.sssZ',
                'raw_ticker': {...} # Full ticker data from ccxt
            }
        """
        normalized_symbol = self._normalize_symbol_for_ccxt(symbol)
        cache_key = f"current_price_{self.exchange_name}_{normalized_symbol.replace('/', '_')}"
        
        logger.info(f"Fetching current price for {normalized_symbol}, use_cache: {use_cache}")
        logger.debug(f"Cache key for current price: {cache_key}")

        if use_cache:
            cached_data = get_cached_json_data(cache_key)
            if cached_data:
                logger.info(f"Cache hit for current price {cache_key}. Returning cached data.")
                return cached_data
            logger.debug(f"Cache miss for current price {cache_key}.")

        try:
            if not self.exchange.has['fetchTicker']:
                logger.error(f"{self.exchange_name} does not support fetchTicker via CCXT.")
                return {}

            logger.debug(f"Calling CCXT: fetch_ticker for {normalized_symbol}")
            ticker_data = self.exchange.fetch_ticker(normalized_symbol)

            if not ticker_data:
                logger.warning(f"No ticker data returned from {self.exchange_name} for {normalized_symbol}.")
                return {}

            # Adapt to a common structure
            price_data = {
                'symbol': normalized_symbol,
                'current_price': ticker_data.get('last'),
                'price_change_percentage_24h': ticker_data.get('percentage'), # CCXT often provides this
                'volume_24h': ticker_data.get('baseVolume'), # Volume in base currency
                'quote_volume_24h': ticker_data.get('quoteVolume'), # Volume in quote currency
                'last_updated': datetime.fromtimestamp(ticker_data['timestamp'] / 1000, tz=timezone.utc).isoformat() if ticker_data.get('timestamp') else datetime.now(timezone.utc).isoformat(),
                'raw_ticker': ticker_data
            }
            
            # Clean None values for cleaner output, but keep them in raw_ticker
            price_data_cleaned = {k: v for k, v in price_data.items() if v is not None or k == 'raw_ticker'}


            if use_cache:
                store_json_data(cache_key, price_data_cleaned, ttl=cache_ttl_seconds)
            
            logger.info(f"Current price for {normalized_symbol} fetched: {price_data_cleaned.get('current_price')}")
            return price_data_cleaned

        except ccxt.NetworkError as e:
            logger.error(f"CCXT NetworkError fetching ticker for {normalized_symbol}: {str(e)}", exc_info=True)
        except ccxt.ExchangeError as e:
            logger.error(f"CCXT ExchangeError fetching ticker for {normalized_symbol}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error fetching ticker for {normalized_symbol}: {str(e)}", exc_info=True)

        return {}

# Wrapper functions to maintain compatibility with existing calls if needed,
# or these can be refactored away if the CLI calls DataFetcher methods directly.

_data_fetcher_instance = None

def get_data_fetcher_instance() -> DataFetcher:
    """Returns a singleton instance of DataFetcher."""
    global _data_fetcher_instance
    if _data_fetcher_instance is None:
        _data_fetcher_instance = DataFetcher(exchange_name="okx")
    return _data_fetcher_instance

def get_historical_data(
    symbol: str = 'BTC/USDT', # Expects CCXT format now
    timeframe: str = '1d',
    limit: int = 100, # Number of candles
    use_cache: bool = True,
    # vs_currency is not directly used by CCXT as symbol contains quote
) -> pd.DataFrame:
    """
    High-level function to get historical OHLCV data using the DataFetcher.
    Symbol should be in CCXT format like 'BTC/USDT'.
    """
    fetcher = get_data_fetcher_instance()
    # 'since' parameter can be added here if complex pagination or specific start date is needed.
    # For simplicity, current implementation fetches `limit` most recent candles.
    return fetcher.fetch_historical_ohlcv(symbol, timeframe, limit, use_cache=use_cache)


def get_current_price(
    symbol: str = 'BTC/USDT', # Expects CCXT format
    force_refresh: bool = False,
    # vs_currency is not directly used by CCXT
) -> Dict[str, Any]:
    """
    High-level function to get current price data using the DataFetcher.
    Symbol should be in CCXT format like 'BTC/USDT'.
    """
    fetcher = get_data_fetcher_instance()
    use_cache = not force_refresh
    return fetcher.get_current_price_data(symbol, use_cache=use_cache)

def invalidate_price_cache(symbol: str = 'BTC/USDT', timeframe: str = '1d', limit: int = 100) -> bool:
    """
    Invalidates a specific cache entry.
    Note: This is a placeholder. Actual cache invalidation might need more specific
    cache key construction matching that in fetch_historical_ohlcv or get_current_price_data.
    The cache_service module itself might be better suited for direct invalidation calls.
    For now, it's illustrative.
    """
    # This needs to be more robust if used, constructing the exact cache key.
    # For current price:
    # cache_key = f"current_price_okx_{symbol.replace('/', '_')}"
    # For historical:
    # cache_key = f"ohlcv_okx_{symbol.replace('/', '_')}_{timeframe}_{limit}"
    # from .cache_service import invalidate_cache_item # Assuming such a function exists
    # return invalidate_cache_item(cache_key)
    logger.warning("invalidate_price_cache is a placeholder and not fully implemented for specific keys.")
    return False

# Example usage (for testing purposes)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test with .env file having OKX_API_KEY, OKX_SECRET_KEY, OKX_PASSWORD (optional)
    # Create a dummy .env for testing if you don't want to use real keys
    # with open(".env", "w") as f:
    #     f.write("OKX_API_KEY=\"your_test_key_if_public_endpoints_are_not_enough\"\n")
    #     f.write("OKX_SECRET_KEY=\"your_test_secret_if_public_endpoints_are_not_enough\"\n")
    #     f.write("OKX_PASSWORD=\"your_test_password_if_needed\"\n")

    test_symbol = 'BTC/USDT'
    test_timeframe = '1h'
    test_limit = 5

    print(f"--- Testing DataFetcher with {test_symbol} ---")

    # --- Test get_current_price ---
    print(f"\n--- Fetching current price for {test_symbol} ---")
    current_price_data = get_current_price(test_symbol, force_refresh=True)
    if current_price_data:
        print(f"Current Price for {test_symbol}: {current_price_data.get('current_price')}")
        print(f"Last Updated: {current_price_data.get('last_updated')}")
        print(f"Full raw ticker (first 200 chars): {str(current_price_data.get('raw_ticker'))[:200]}...")
    else:
        print(f"Could not fetch current price for {test_symbol}.")

    # --- Test get_historical_data ---
    print(f"\n--- Fetching historical OHLCV for {test_symbol} ({test_timeframe}, limit {test_limit}) ---")
    historical_df = get_historical_data(test_symbol, timeframe=test_timeframe, limit=test_limit, use_cache=False)
    if not historical_df.empty:
        print(f"Historical Data for {test_symbol}:")
        print(historical_df.head())
        print("...")
        print(historical_df.tail())
        print(f"Shape: {historical_df.shape}")
    else:
        print(f"Could not fetch historical data for {test_symbol}.")

    # --- Test with a different symbol or timeframe ---
    test_symbol_eth = 'ETH/USDT'
    test_timeframe_daily = '1d'
    test_limit_daily = 3
    print(f"\n--- Fetching historical OHLCV for {test_symbol_eth} ({test_timeframe_daily}, limit {test_limit_daily}) ---")
    historical_df_eth = get_historical_data(test_symbol_eth, timeframe=test_timeframe_daily, limit=test_limit_daily, use_cache=True) # Test cache
    if not historical_df_eth.empty:
        print(f"Historical Data for {test_symbol_eth}:")
        print(historical_df_eth)
    else:
        print(f"Could not fetch historical data for {test_symbol_eth}.")
    
    # Test fetching again to ensure cache works (if TTL is long enough and data was stored)
    print(f"\n--- Fetching historical OHLCV for {test_symbol_eth} ({test_timeframe_daily}, limit {test_limit_daily}) AGAIN (testing cache) ---")
    historical_df_eth_cached = get_historical_data(test_symbol_eth, timeframe=test_timeframe_daily, limit=test_limit_daily, use_cache=True)
    if not historical_df_eth_cached.empty:
        print(f"Historical Data for {test_symbol_eth} (cached):")
        print(historical_df_eth_cached)
    else:
        print(f"Could not fetch historical data for {test_symbol_eth} (cached).")

    # Clean up dummy .env if created for testing
    # import os
    # if os.path.exists(".env") and "your_test_key" in open(".env").read():
    #     os.remove(".env")
    #     print("\nRemoved dummy .env file.") 