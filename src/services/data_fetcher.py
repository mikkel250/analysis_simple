"""
Data Fetcher Module

This module handles fetching cryptocurrency market data from CoinGecko API
and converting it to pandas DataFrame format suitable for technical analysis.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
from pycoingecko import CoinGeckoAPI

# Import cache_service for caching price data
from .cache_service import (
    store_dataframe,
    get_cached_dataframe,
    store_json_data,
    get_cached_json_data,
    invalidate_cache,
    DEFAULT_TTL
)


class DataFetcher:
    """
    A class to fetch cryptocurrency data from CoinGecko API.
    """
    
    def __init__(self):
        """Initialize the DataFetcher with CoinGecko API client."""
        self.cg = CoinGeckoAPI()
        # Set rate limit sleep time (in seconds) to avoid hitting API limits
        self.rate_limit_sleep = 1.5
        
    def _handle_api_call(self, api_call_func, *args, **kwargs) -> Dict:
        """
        Helper method to handle API calls with rate limiting and error handling.
        
        Args:
            api_call_func: Function to call the CoinGecko API
            *args: Arguments to pass to the API call function
            **kwargs: Keyword arguments to pass to the API call function
            
        Returns:
            Dict: API response data
            
        Raises:
            Exception: If API call fails after retries
        """
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = api_call_func(*args, **kwargs)
                # Sleep to respect rate limits
                time.sleep(self.rate_limit_sleep)
                return result
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"Failed to fetch data from CoinGecko API after {max_retries} retries: {str(e)}")
                # Exponential backoff
                time.sleep(self.rate_limit_sleep * (2 ** retry_count))
        
        # This should never be reached due to the exception above
        raise Exception("Unexpected error in API call")
    
    def fetch_historical_ohlc(
        self, 
        coin_id: str = 'bitcoin', 
        vs_currency: str = 'usd',
        days: Union[int, str] = 30,
        interval: str = 'daily',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical OHLC data for a specified coin.
        
        Args:
            coin_id: CoinGecko coin ID (default: 'bitcoin')
            vs_currency: Quote currency (default: 'usd')
            days: Number of days to look back (default: 30)
            interval: Data interval - 'daily' or 'hourly' (default: 'daily')
            use_cache: Whether to use cached data if available (default: True)
            
        Returns:
            pandas.DataFrame: OHLC data with columns [timestamp, open, high, low, close, volume]
        """
        valid_intervals = ['daily', 'hourly']
        if interval not in valid_intervals:
            raise ValueError(f"Interval must be one of {valid_intervals}")
        
        # Generate cache key
        timeframe = '1d' if interval == 'daily' else '1h'
        cache_key = f"ohlc_{coin_id}_{vs_currency}_{timeframe}_{days}"
        
        # Try to get from cache first if use_cache is True
        if use_cache:
            cached_df = get_cached_dataframe(cache_key)
            if cached_df is not None:
                return cached_df
        
        # Get market chart data
        try:
            market_data = self._handle_api_call(
                self.cg.get_coin_ohlc_by_id,
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )
            
            # If no data is returned
            if not market_data or len(market_data) == 0:
                raise Exception(f"No data returned for {coin_id} in {vs_currency}")
                
            # Convert to DataFrame
            df = pd.DataFrame(
                market_data, 
                columns=['timestamp', 'open', 'high', 'low', 'close']
            )
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            
            # Add empty volume column (CoinGecko OHLC endpoint doesn't provide volume)
            df['volume'] = 0
            
            # Store in cache if not empty
            if not df.empty and use_cache:
                metadata = {
                    'coin_id': coin_id,
                    'vs_currency': vs_currency,
                    'days': days,
                    'interval': interval,
                    'source': 'coingecko',
                    'endpoint': 'ohlc'
                }
                store_dataframe(cache_key, df, metadata=metadata, timeframe=timeframe)
            
            return df
            
        except Exception as e:
            raise Exception(f"Error fetching OHLC data: {str(e)}")
    
    def fetch_historical_price_volume(
        self, 
        coin_id: str = 'bitcoin', 
        vs_currency: str = 'usd',
        days: Union[int, str] = 30,
        interval: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical price and volume data for a specified coin and convert to OHLCV format.
        This method uses market_chart endpoint which provides prices and volumes but not OHLC directly.
        
        Args:
            coin_id: CoinGecko coin ID (default: 'bitcoin')
            vs_currency: Quote currency (default: 'usd')
            days: Number of days to look back (default: 30)
            interval: Data interval - for days > 90, must be 'daily' (default: None)
            use_cache: Whether to use cached data if available (default: True)
            
        Returns:
            pandas.DataFrame: OHLCV data with columns [open, high, low, close, volume]
        """
        # Determine timeframe for caching
        timeframe = self._get_timeframe_from_days(days)
        
        # Generate cache key
        cache_key = f"market_chart_{coin_id}_{vs_currency}_{timeframe}_{days}"
        if interval:
            cache_key += f"_{interval}"
        
        # Try to get from cache first if use_cache is True
        if use_cache:
            cached_df = get_cached_dataframe(cache_key)
            if cached_df is not None:
                return cached_df
        
        try:
            # Get market chart data
            market_data = self._handle_api_call(
                self.cg.get_coin_market_chart_by_id,
                id=coin_id,
                vs_currency=vs_currency,
                days=days,
                interval=interval
            )
            
            # Extract price and volume data
            prices = market_data['prices']
            volumes = market_data['total_volumes']
            
            # Create DataFrames
            df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
            df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            
            # Convert timestamps from milliseconds to datetime
            df_prices['timestamp'] = pd.to_datetime(df_prices['timestamp'], unit='ms')
            df_volumes['timestamp'] = pd.to_datetime(df_volumes['timestamp'], unit='ms')
            
            # Resample to the desired interval to create OHLC
            df_prices.set_index('timestamp', inplace=True)
            
            # Determine appropriate resampling frequency based on input days
            if days == 1:
                freq = '5min'  # 5-minute intervals for 1 day
            elif days <= 7:
                freq = '1h'    # Hourly for 2-7 days
            elif days <= 30:
                freq = '4h'    # 4-hour intervals for 8-30 days
            else:
                freq = '1D'    # Daily for >30 days
            
            # Resample to get OHLC
            ohlc = df_prices['price'].resample(freq).ohlc()
            
            # Merge with volume data
            df_volumes.set_index('timestamp', inplace=True)
            volume_resampled = df_volumes['volume'].resample(freq).sum()
            
            # Combine OHLC and volume
            result = pd.concat([ohlc, volume_resampled], axis=1)
            
            # Store in cache if not empty
            if not result.empty and use_cache:
                # Update timeframe based on actual resampling frequency
                if freq == '5min':
                    cache_timeframe = '5m'
                elif freq == '1h':
                    cache_timeframe = '1h'
                elif freq == '4h':
                    cache_timeframe = '4h'
                elif freq == '1D':
                    cache_timeframe = '1d'
                else:
                    cache_timeframe = timeframe
                
                metadata = {
                    'coin_id': coin_id,
                    'vs_currency': vs_currency,
                    'days': days,
                    'interval': interval,
                    'resampled_freq': freq,
                    'source': 'coingecko',
                    'endpoint': 'market_chart'
                }
                store_dataframe(cache_key, result, metadata=metadata, timeframe=cache_timeframe)
            
            return result
        
        except Exception as e:
            raise Exception(f"Error fetching historical price and volume data: {str(e)}")
    
    def _get_timeframe_from_days(self, days: Union[int, str]) -> str:
        """
        Determine the appropriate timeframe string based on the number of days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            str: Timeframe string (1m, 5m, 1h, 1d, etc.)
        """
        if days == 1:
            return "5m"
        elif days <= 7:
            return "1h"
        elif days <= 30:
            return "4h"
        else:
            return "1d"
    
    def fetch_data_by_timeframe(
        self,
        timeframe: str = '1d',
        coin_id: str = 'bitcoin',
        vs_currency: str = 'usd',
        limit: int = 100,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical data for a specific timeframe.
        
        Args:
            timeframe: Time interval ('1h', '4h', '1d', etc.)
            coin_id: CoinGecko coin ID (default: 'bitcoin')
            vs_currency: Quote currency (default: 'usd')
            limit: Number of candles to fetch (default: 100)
            use_cache: Whether to use cached data if available (default: True)
            
        Returns:
            pandas.DataFrame: OHLCV data for the specified timeframe
            
        Raises:
            ValueError: If timeframe is not supported
        """
        # Map timeframes to days and resampling frequencies
        timeframe_map = {
            '5m': {'days': 1, 'freq': '5min'},
            '15m': {'days': 1, 'freq': '15min'},
            '30m': {'days': 1, 'freq': '30min'},
            '1h': {'days': 7, 'freq': '1h'},
            '2h': {'days': 14, 'freq': '2h'},
            '4h': {'days': 30, 'freq': '4h'},
            '6h': {'days': 30, 'freq': '6h'},
            '12h': {'days': 60, 'freq': '12h'},
            '1d': {'days': 365, 'freq': '1D'},
            '3d': {'days': 365, 'freq': '3D'},
            '1w': {'days': 365, 'freq': '1W'},
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Supported timeframes: {list(timeframe_map.keys())}")
        
        # Generate cache key
        cache_key = f"data_{coin_id}_{vs_currency}_{timeframe}_{limit}"
        
        # Try to get from cache first if use_cache is True
        if use_cache:
            cached_df = get_cached_dataframe(cache_key)
            if cached_df is not None:
                return cached_df
        
        # Get configuration for timeframe
        config = timeframe_map[timeframe]
        days = config['days']
        freq = config['freq']
        
        # For shorter timeframes, use market_chart endpoint
        if timeframe in ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h']:
            # Calculate days needed based on limit and frequency
            if isinstance(days, int) and days < (limit * int(freq[0]) / 24):
                days = max(days, int(limit * int(freq[0]) / 24) + 1)
            
            df = self.fetch_historical_price_volume(
                coin_id=coin_id,
                vs_currency=vs_currency,
                days=days,
                use_cache=use_cache
            )
            
            # Resample to the desired frequency
            if df.index.freq != freq:
                df = df['price'].resample(freq).ohlc()
                # Since we resampled, we need to get volume data again
                volume_data = self.fetch_historical_price_volume(
                    coin_id=coin_id,
                    vs_currency=vs_currency,
                    days=days,
                    use_cache=use_cache
                )
                volume_resampled = volume_data['volume'].resample(freq).sum()
                df = pd.concat([df, volume_resampled], axis=1)
        
        # For daily and longer timeframes, use OHLC endpoint
        else:
            # For daily timeframes, ensure we don't exceed API limits
            # CoinGecko API free tier allows max 365 days of historical data
            if days > 365:
                days = 365
                
            df = self.fetch_historical_ohlc(
                coin_id=coin_id,
                vs_currency=vs_currency,
                days=days,
                use_cache=use_cache
            )
            
            # Resample if needed
            if freq != '1D':
                df = df.resample(freq).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
        
        # Limit the number of rows and store in cache
        result_df = df.tail(limit)
        
        # Store in cache if not empty and not already cached
        if not result_df.empty and use_cache:
            metadata = {
                'coin_id': coin_id,
                'vs_currency': vs_currency,
                'timeframe': timeframe,
                'limit': limit,
                'source': 'coingecko'
            }
            store_dataframe(cache_key, result_df, metadata=metadata, timeframe=timeframe)
        
        # Limit the number of rows
        return result_df


# Convenience functions to use the DataFetcher class
def get_historical_data(
    symbol: str = 'BTC',
    timeframe: str = '1d',
    limit: int = 100,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Get historical OHLCV data for a cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (default: 'BTC')
        timeframe: Time interval ('1h', '4h', '1d', etc.) (default: '1d')
        limit: Number of candles to fetch (default: 100)
        use_cache: Whether to use cached data if available (default: True)
        
    Returns:
        pandas.DataFrame: OHLCV data with columns [open, high, low, close, volume]
    """
    # Map symbol to CoinGecko ID
    symbol_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'XRP': 'ripple',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash',
        'BNB': 'binancecoin',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'ADA': 'cardano',
        'SOL': 'solana',
    }
    
    coin_id = symbol_map.get(symbol, 'bitcoin')
    
    fetcher = DataFetcher()
    df = fetcher.fetch_data_by_timeframe(
        timeframe=timeframe,
        coin_id=coin_id,
        limit=limit,
        use_cache=use_cache
    )
    
    return df

def invalidate_price_cache(
    symbol: str = 'BTC',
    timeframe: str = '1d'
) -> bool:
    """
    Invalidate the cache for a specific symbol and timeframe.
    
    Args:
        symbol: Cryptocurrency symbol (default: 'BTC')
        timeframe: Time interval ('1h', '4h', '1d', etc.) (default: '1d')
        
    Returns:
        bool: True if cache was invalidated, False otherwise
    """
    # Map symbol to CoinGecko ID
    symbol_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'XRP': 'ripple',
        'LTC': 'litecoin',
        'BCH': 'bitcoin-cash',
        'BNB': 'binancecoin',
        'DOT': 'polkadot',
        'LINK': 'chainlink',
        'ADA': 'cardano',
        'SOL': 'solana',
    }
    
    coin_id = symbol_map.get(symbol, 'bitcoin')
    
    # Generate cache key pattern (partial match)
    cache_key_pattern = f"data_{coin_id}_usd_{timeframe}"
    
    # Actually we need multiple invalidations since we have different cache keys
    invalidated = invalidate_cache(cache_key_pattern)
    
    return invalidated

def get_current_price(
    symbol: str = 'BTC',
    force_refresh: bool = False
) -> Dict[str, Any]:
    """
    Get current price data for a specific cryptocurrency.
    
    Args:
        symbol: Cryptocurrency symbol (default: 'BTC')
        force_refresh: Whether to force refresh data from API instead of using cache
        
    Returns:
        Dict: Current price and related data
    """
    # Map symbols to CoinGecko IDs
    symbol_map = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'SOL': 'solana',
        'ADA': 'cardano',
        'DOT': 'polkadot',
        'AVAX': 'avalanche-2',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'AAVE': 'aave',
        'MATIC': 'polygon',
        'DOGE': 'dogecoin',
        'SHIB': 'shiba-inu'
    }
    
    # Default to 'bitcoin' if symbol not in map
    coin_id = symbol_map.get(symbol.upper(), 'bitcoin')
    
    # Generate cache key
    cache_key = f"price_{coin_id}_current"
    
    # Check cache first if not forcing refresh
    if not force_refresh:
        cached_data = get_cached_json_data(cache_key)
        if cached_data is not None:
            # Check if the cached data is less than 5 minutes old
            cached_time = cached_data.get('metadata', {}).get('timestamp', '')
            if cached_time:
                cache_time = datetime.fromisoformat(cached_time)
                if datetime.now() - cache_time < timedelta(minutes=5):
                    return cached_data.get('data', {})
    
    try:
        # Initialize DataFetcher and fetch data
        fetcher = DataFetcher()
        
        # Fetch price data from CoinGecko
        price_data = fetcher._handle_api_call(
            fetcher.cg.get_coin_by_id,
            id=coin_id,
            localization=False,
            tickers=False,
            market_data=True,
            community_data=False,
            developer_data=False,
            sparkline=False
        )
        
        # Extract only the market data
        if price_data and 'market_data' in price_data:
            market_data = price_data['market_data']
            
            # Store in cache
            metadata = {
                'coin_id': coin_id,
                'symbol': symbol,
                'source': 'coingecko',
                'endpoint': 'coin_by_id',
                'timestamp': datetime.now().isoformat()
            }
            store_json_data(cache_key, market_data, metadata=metadata)
            
            return market_data
        else:
            raise Exception(f"No market data available for {symbol}")
        
    except Exception as e:
        # If error occurs and we have cached data, return that even if it's old
        cached_data = get_cached_json_data(cache_key)
        if cached_data is not None:
            print(f"Warning: Error fetching current price, using cached data: {str(e)}")
            return cached_data.get('data', {})
        else:
            raise Exception(f"Error fetching current price and no cached data available: {str(e)}") 