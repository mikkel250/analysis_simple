"""
Cache Service

This module provides caching functionality for API responses and calculated indicators.
It implements persistent caching with TTL (Time To Live) based on data timeframe.

Typical usage example:
    # Store DataFrame in cache
    store_dataframe("btc_usdt_1h", df, ttl=3600)
    
    # Retrieve DataFrame from cache
    df = get_cached_dataframe("btc_usdt_1h")
    
    # Store indicator result in cache
    store_indicator("sma_20_btc_usdt_1h", indicator_result, ttl=3600)
    
    # Retrieve indicator result from cache
    indicator_result = get_cached_indicator("sma_20_btc_usdt_1h")
    
    # Store JSON data in cache
    store_json_data("btc_price", price_data, ttl=300)
    
    # Retrieve JSON data from cache
    price_data = get_cached_json_data("btc_price")
"""

import os
import json
import logging
import time
import fcntl
import shutil
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "cache")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Subdirectories for different types of cached data
DATAFRAME_CACHE_DIR = os.path.join(CACHE_DIR, "dataframes")
INDICATOR_CACHE_DIR = os.path.join(CACHE_DIR, "indicators")
JSON_CACHE_DIR = os.path.join(CACHE_DIR, "json")

# Ensure subdirectories exist
os.makedirs(DATAFRAME_CACHE_DIR, exist_ok=True)
os.makedirs(INDICATOR_CACHE_DIR, exist_ok=True)
os.makedirs(JSON_CACHE_DIR, exist_ok=True)

# Default TTL values (in seconds) based on timeframes
DEFAULT_TTL = {
    "1m": 5 * 60,         # 5 minutes
    "5m": 15 * 60,        # 15 minutes
    "15m": 30 * 60,       # 30 minutes
    "30m": 60 * 60,       # 1 hour
    "1h": 2 * 60 * 60,    # 2 hours
    "4h": 4 * 60 * 60,    # 4 hours
    "12h": 8 * 60 * 60,   # 8 hours
    "1d": 24 * 60 * 60,   # 1 day
    "3d": 3 * 24 * 60 * 60,  # 3 days
    "1w": 7 * 24 * 60 * 60   # 1 week
}

# Default TTL values (in seconds) for indicators based on their complexity
# More complex indicators require more computation, so they are cached longer
INDICATOR_TTL = {
    "simple": {  # Simple indicators like SMA, EMA, RSI
        "1m": 10 * 60,         # 10 minutes
        "5m": 30 * 60,         # 30 minutes
        "15m": 60 * 60,        # 1 hour
        "30m": 2 * 60 * 60,    # 2 hours
        "1h": 4 * 60 * 60,     # 4 hours
        "4h": 8 * 60 * 60,     # 8 hours
        "12h": 12 * 60 * 60,   # 12 hours
        "1d": 2 * 24 * 60 * 60,   # 2 days
        "3d": 4 * 24 * 60 * 60,   # 4 days
        "1w": 10 * 24 * 60 * 60   # 10 days
    },
    "complex": {  # Complex indicators like MACD, Bollinger Bands, Ichimoku
        "1m": 15 * 60,         # 15 minutes
        "5m": 45 * 60,         # 45 minutes
        "15m": 90 * 60,        # 1.5 hours
        "30m": 3 * 60 * 60,    # 3 hours
        "1h": 6 * 60 * 60,     # 6 hours
        "4h": 12 * 60 * 60,    # 12 hours
        "12h": 18 * 60 * 60,   # 18 hours
        "1d": 3 * 24 * 60 * 60,   # 3 days
        "3d": 6 * 24 * 60 * 60,   # 6 days
        "1w": 14 * 24 * 60 * 60   # 14 days
    }
}

# Indicator complexity categorization
INDICATOR_COMPLEXITY = {
    "simple": [
        "sma", "ema", "rsi", "adx", "atr", "cci", "obv"
    ],
    "complex": [
        "macd", "bbands", "stoch", "ichimoku"
    ]
}

# Current library versions for versioning cache entries
LIBRARY_VERSIONS = {
    "pandas_ta": "0.3.14b0",  # Should match the actual version used
    "pandas": pd.__version__,
    "numpy": np.__version__
}

# Statistics for cache operations
CACHE_STATS = {
    "dataframe": {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "errors": 0
    },
    "indicator": {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "errors": 0
    },
    "json": {
        "hits": 0,
        "misses": 0,
        "stores": 0,
        "errors": 0
    }
}


class CacheError(Exception):
    """Exception raised for cache-related errors."""
    pass


def _get_cache_path(key: str, cache_type: str = "dataframe") -> str:
    """
    Generate a file path for a cache key.
    
    Args:
        key: Unique identifier for the cached data
        cache_type: Type of cache ('dataframe', 'indicator', or 'json')
        
    Returns:
        str: Path to the cache file
    """
    # Sanitize key to ensure it's a valid filename
    sanitized_key = "".join(c if c.isalnum() or c in "_-." else "_" for c in key)
    
    # Select the appropriate cache directory
    if cache_type == "indicator":
        cache_dir = INDICATOR_CACHE_DIR
    elif cache_type == "json":
        cache_dir = JSON_CACHE_DIR
    else:
        cache_dir = DATAFRAME_CACHE_DIR
        
    return os.path.join(cache_dir, f"{sanitized_key}.json")


def _serialize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert a DataFrame to a serializable format.
    
    Args:
        df: DataFrame to serialize
        
    Returns:
        Dict: Serialized DataFrame
    """
    # Reset the index before serializing if it's a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        df_reset = df.reset_index()
        # Convert any datetime columns to ISO format strings
        for col in df_reset.columns:
            if pd.api.types.is_datetime64_any_dtype(df_reset[col]):
                df_reset[col] = df_reset[col].astype(str)
    else:
        df_reset = df.reset_index()
    
    # Convert the DataFrame to a dictionary
    data_dict = df_reset.to_dict(orient="records")
    
    # Include column and index information
    result = {
        "data": data_dict,
        "columns": df.columns.tolist(),
        "index_name": df.index.name,
        "index_type": str(type(df.index)),
        "has_datetime_index": isinstance(df.index, pd.DatetimeIndex)
    }
    
    return result


def _deserialize_dataframe(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Convert serialized data back to a DataFrame.
    
    Args:
        data: Serialized DataFrame
        
    Returns:
        pd.DataFrame: Deserialized DataFrame
    """
    # Create DataFrame from records
    df = pd.DataFrame(data["data"])
    
    # Restore the index
    if data.get("has_datetime_index", False):
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
    elif data.get("index_name") and data.get("index_name") in df.columns:
        df.set_index(data["index_name"], inplace=True)
    
    return df


def store_dataframe(
    key: str,
    df: pd.DataFrame,
    metadata: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None,
    timeframe: Optional[str] = None
) -> bool:
    """
    Store a DataFrame in the cache with metadata and TTL.
    
    Args:
        key: Unique identifier for the cached data
        df: DataFrame to cache
        metadata: Additional metadata to store with the DataFrame
        ttl: Time to live in seconds
        timeframe: Timeframe of the data (1m, 5m, 1h, 1d, etc.)
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        CacheError: If there's an error storing the data
    """
    if df is None or df.empty:
        logger.warning(f"Attempted to cache empty DataFrame with key: {key}")
        return False
    
    # Determine TTL
    if ttl is None and timeframe is not None:
        ttl = DEFAULT_TTL.get(timeframe, DEFAULT_TTL.get("1h"))
    elif ttl is None:
        ttl = DEFAULT_TTL.get("1h")  # Default to 1 hour
    
    # Create cache entry
    cache_entry = {
        "data": _serialize_dataframe(df),
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
        "ttl": ttl,
        "timeframe": timeframe,
        "version": LIBRARY_VERSIONS,
        "cache_type": "dataframe"
    }
    
    cache_path = _get_cache_path(key, cache_type="dataframe")
    temp_path = f"{cache_path}.tmp"
    
    try:
        # Write to a temporary file first
        with open(temp_path, 'w') as f:
            # Acquire a lock to ensure thread safety
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(cache_entry, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomically rename the temporary file
        shutil.move(temp_path, cache_path)
        
        CACHE_STATS["dataframe"]["stores"] += 1
        logger.debug(f"Stored DataFrame in cache with key: {key}, expires: {cache_entry['expires_at']}")
        return True
    
    except Exception as e:
        CACHE_STATS["dataframe"]["errors"] += 1
        logger.error(f"Error storing DataFrame in cache with key: {key}, error: {str(e)}")
        
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        raise CacheError(f"Failed to store DataFrame in cache: {str(e)}")


def get_cached_dataframe(
    key: str,
    default: Optional[pd.DataFrame] = None,
    allow_expired: bool = False
) -> Optional[pd.DataFrame]:
    """
    Retrieve a DataFrame from the cache if it exists and is not expired.
    
    Args:
        key: Unique identifier for the cached data
        default: Default value to return if cache miss
        allow_expired: If True, return expired data (with a warning log)
        
    Returns:
        pd.DataFrame: The cached DataFrame or default if not found/expired
        
    Raises:
        CacheError: If there's an error retrieving the data
    """
    cache_path = _get_cache_path(key, cache_type="dataframe")
    
    if not os.path.exists(cache_path):
        CACHE_STATS["dataframe"]["misses"] += 1
        logger.debug(f"Cache miss for key: {key}")
        return default
    
    try:
        with open(cache_path, 'r') as f:
            # Acquire a shared lock for reading
            fcntl.flock(f, fcntl.LOCK_SH)
            cache_entry = json.load(f)
        
        # Check if expired
        expires_at = datetime.fromisoformat(cache_entry["expires_at"])
        if datetime.now() > expires_at and not allow_expired:
            CACHE_STATS["dataframe"]["misses"] += 1
            logger.debug(f"Cache expired for key: {key}, expired at: {expires_at.isoformat()}")
            return default
        
        # Check library versions if needed for critical dependencies
        if "version" in cache_entry and isinstance(cache_entry["version"], dict):
            cache_version = cache_entry["version"]
            if "pandas_ta" in cache_version and cache_version["pandas_ta"] != LIBRARY_VERSIONS["pandas_ta"]:
                logger.warning(f"Cache entry was created with different pandas_ta version: {cache_version['pandas_ta']} vs current {LIBRARY_VERSIONS['pandas_ta']}")
                # We don't invalidate based on version difference for DataFrames
        
        # Return the DataFrame
        df = _deserialize_dataframe(cache_entry["data"])
        
        if datetime.now() > expires_at:
            logger.warning(f"Using expired cache data for key: {key}, expired at: {expires_at.isoformat()}")
        
        CACHE_STATS["dataframe"]["hits"] += 1
        logger.debug(f"Cache hit for key: {key}")
        return df
    
    except Exception as e:
        CACHE_STATS["dataframe"]["errors"] += 1
        logger.error(f"Error retrieving DataFrame from cache with key: {key}, error: {str(e)}")
        return default


def store_indicator(
    key: str,
    indicator_result: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None,
    timeframe: Optional[str] = None,
    indicator_type: Optional[str] = None
) -> bool:
    """
    Store an indicator result in the cache with metadata and TTL.
    
    Args:
        key: Unique identifier for the cached indicator
        indicator_result: Indicator calculation result to cache
        metadata: Additional metadata to store with the indicator result
        ttl: Time to live in seconds
        timeframe: Timeframe of the data (1m, 5m, 1h, 1d, etc.)
        indicator_type: Type of indicator ('simple' or 'complex')
        
    Returns:
        bool: True if successful, False otherwise
        
    Raises:
        CacheError: If there's an error storing the indicator result
    """
    if indicator_result is None or not indicator_result:
        logger.warning(f"Attempted to cache empty indicator result with key: {key}")
        return False
    
    # Get indicator name from result if available
    indicator_name = indicator_result.get("indicator", "")
    
    # Determine indicator complexity
    if indicator_type is None:
        if indicator_name in INDICATOR_COMPLEXITY["simple"]:
            indicator_type = "simple"
        elif indicator_name in INDICATOR_COMPLEXITY["complex"]:
            indicator_type = "complex"
        else:
            indicator_type = "simple"  # Default to simple
    
    # Determine TTL based on indicator complexity and timeframe
    if ttl is None and timeframe is not None:
        ttl_dict = INDICATOR_TTL.get(indicator_type, INDICATOR_TTL["simple"])
        ttl = ttl_dict.get(timeframe, ttl_dict.get("1h"))
    elif ttl is None:
        ttl = INDICATOR_TTL["simple"].get("1h")  # Default to 1 hour for simple indicators
    
    # Generate a hash of the indicator parameters for versioning
    params_hash = "unknown"
    if "metadata" in indicator_result and "params" in indicator_result["metadata"]:
        params_dict = indicator_result["metadata"]["params"]
        params_str = json.dumps(params_dict, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # Create cache entry
    cache_entry = {
        "data": indicator_result,
        "metadata": metadata or {},
        "created_at": datetime.now().isoformat(),
        "expires_at": (datetime.now() + timedelta(seconds=ttl)).isoformat(),
        "ttl": ttl,
        "timeframe": timeframe,
        "indicator_name": indicator_name,
        "indicator_type": indicator_type,
        "params_hash": params_hash,
        "version": LIBRARY_VERSIONS,
        "cache_type": "indicator"
    }
    
    cache_path = _get_cache_path(key, cache_type="indicator")
    temp_path = f"{cache_path}.tmp"
    
    try:
        # Write to a temporary file first
        with open(temp_path, 'w') as f:
            # Acquire a lock to ensure thread safety
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(cache_entry, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        
        # Atomically rename the temporary file
        shutil.move(temp_path, cache_path)
        
        CACHE_STATS["indicator"]["stores"] += 1
        logger.debug(f"Stored indicator result in cache with key: {key}, expires: {cache_entry['expires_at']}")
        return True
    
    except Exception as e:
        CACHE_STATS["indicator"]["errors"] += 1
        logger.error(f"Error storing indicator result in cache with key: {key}, error: {str(e)}")
        
        # Clean up temporary file if it exists
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        
        raise CacheError(f"Failed to store indicator result in cache: {str(e)}")


def get_cached_indicator(
    key: str,
    default: Optional[Dict[str, Any]] = None,
    allow_expired: bool = False,
    check_version: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Retrieve an indicator result from the cache if it exists and is not expired.
    
    Args:
        key: Unique identifier for the cached indicator
        default: Default value to return if cache miss
        allow_expired: If True, return expired data (with a warning log)
        check_version: If True, check library versions for compatibility
        
    Returns:
        Dict: The cached indicator result or default if not found/expired
        
    Raises:
        CacheError: If there's an error retrieving the indicator result
    """
    cache_path = _get_cache_path(key, cache_type="indicator")
    
    if not os.path.exists(cache_path):
        CACHE_STATS["indicator"]["misses"] += 1
        logger.debug(f"Cache miss for indicator key: {key}")
        return default
    
    try:
        with open(cache_path, 'r') as f:
            # Acquire a shared lock for reading
            fcntl.flock(f, fcntl.LOCK_SH)
            cache_entry = json.load(f)
        
        # Check if expired
        expires_at = datetime.fromisoformat(cache_entry["expires_at"])
        if datetime.now() > expires_at and not allow_expired:
            CACHE_STATS["indicator"]["misses"] += 1
            logger.debug(f"Cache expired for indicator key: {key}, expired at: {expires_at.isoformat()}")
            return default
        
        # Check library versions if needed
        if check_version and "version" in cache_entry and isinstance(cache_entry["version"], dict):
            cache_version = cache_entry["version"]
            # For indicators, we invalidate the cache if pandas_ta version is different
            if "pandas_ta" in cache_version and cache_version["pandas_ta"] != LIBRARY_VERSIONS["pandas_ta"]:
                logger.warning(f"Indicator cache entry was created with different pandas_ta version: {cache_version['pandas_ta']} vs current {LIBRARY_VERSIONS['pandas_ta']}. Invalidating.")
                CACHE_STATS["indicator"]["misses"] += 1
                return default
        
        # Return the indicator result
        indicator_result = cache_entry["data"]
        
        if datetime.now() > expires_at:
            logger.warning(f"Using expired indicator cache data for key: {key}, expired at: {expires_at.isoformat()}")
        
        CACHE_STATS["indicator"]["hits"] += 1
        logger.debug(f"Cache hit for indicator key: {key}")
        return indicator_result
    
    except Exception as e:
        CACHE_STATS["indicator"]["errors"] += 1
        logger.error(f"Error retrieving indicator from cache with key: {key}, error: {str(e)}")
        return default


def invalidate_cache(key: str, cache_type: Optional[str] = None) -> bool:
    """
    Explicitly invalidate a cache entry.
    
    Args:
        key: Unique identifier for the cached data
        cache_type: Type of cache ('dataframe', 'indicator', or None for all)
        
    Returns:
        bool: True if at least one cache entry was invalidated, False if none existed
    """
    invalidated = False
    
    if cache_type is None or cache_type == "dataframe":
        cache_path = _get_cache_path(key, cache_type="dataframe")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info(f"Invalidated dataframe cache entry: {key}")
                invalidated = True
            except Exception as e:
                logger.error(f"Error invalidating dataframe cache entry: {key}, error: {str(e)}")
    
    if cache_type is None or cache_type == "indicator":
        cache_path = _get_cache_path(key, cache_type="indicator")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                logger.info(f"Invalidated indicator cache entry: {key}")
                invalidated = True
            except Exception as e:
                logger.error(f"Error invalidating indicator cache entry: {key}, error: {str(e)}")
    
    return invalidated


def generate_indicator_cache_key(
    indicator_name: str,
    params: Dict[str, Any],
    symbol: str,
    timeframe: str
) -> str:
    """
    Generate a cache key for an indicator result.
    
    Args:
        indicator_name: Name of the indicator (e.g., 'sma', 'macd')
        params: Parameters used for the indicator calculation
        symbol: Symbol of the asset (e.g., 'BTC', 'ETH')
        timeframe: Timeframe of the data (e.g., '1h', '1d')
        
    Returns:
        str: Cache key for the indicator result
    """
    # Generate a hash of the parameters
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
    
    # Create a key with the indicator name, parameters hash, symbol, and timeframe
    key = f"{indicator_name}_{params_hash}_{symbol}_{timeframe}"
    
    return key


def clean_expired_cache(cache_type: Optional[str] = None) -> int:
    """
    Remove all expired cache entries.
    
    Args:
        cache_type: Type of cache to clean ('dataframe', 'indicator', or None for all)
        
    Returns:
        int: Number of expired entries removed
    """
    count = 0
    
    # Clean dataframe cache
    if cache_type is None or cache_type == "dataframe":
        for filename in os.listdir(DATAFRAME_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            cache_path = os.path.join(DATAFRAME_CACHE_DIR, filename)
            
            try:
                with open(cache_path, 'r') as f:
                    # Acquire a shared lock for reading
                    fcntl.flock(f, fcntl.LOCK_SH)
                    cache_entry = json.load(f)
                
                # Check if expired
                expires_at = datetime.fromisoformat(cache_entry["expires_at"])
                if datetime.now() > expires_at:
                    os.remove(cache_path)
                    count += 1
                    logger.debug(f"Removed expired dataframe cache entry: {filename}")
            
            except Exception as e:
                logger.error(f"Error checking/removing dataframe cache entry: {filename}, error: {str(e)}")
    
    # Clean indicator cache
    if cache_type is None or cache_type == "indicator":
        for filename in os.listdir(INDICATOR_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            cache_path = os.path.join(INDICATOR_CACHE_DIR, filename)
            
            try:
                with open(cache_path, 'r') as f:
                    # Acquire a shared lock for reading
                    fcntl.flock(f, fcntl.LOCK_SH)
                    cache_entry = json.load(f)
                
                # Check if expired
                expires_at = datetime.fromisoformat(cache_entry["expires_at"])
                if datetime.now() > expires_at:
                    os.remove(cache_path)
                    count += 1
                    logger.debug(f"Removed expired indicator cache entry: {filename}")
            
            except Exception as e:
                logger.error(f"Error checking/removing indicator cache entry: {filename}, error: {str(e)}")
    
    logger.info(f"Cleaned {count} expired cache entries")
    return count


def clear_all_cache(cache_type: Optional[str] = None) -> int:
    """
    Remove all cache entries.
    
    Args:
        cache_type: Type of cache to clear ('dataframe', 'indicator', or None for all)
        
    Returns:
        int: Number of entries removed
    """
    count = 0
    
    # Clear dataframe cache
    if cache_type is None or cache_type == "dataframe":
        for filename in os.listdir(DATAFRAME_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            cache_path = os.path.join(DATAFRAME_CACHE_DIR, filename)
            
            try:
                os.remove(cache_path)
                count += 1
            except Exception as e:
                logger.error(f"Error removing dataframe cache entry: {filename}, error: {str(e)}")
    
    # Clear indicator cache
    if cache_type is None or cache_type == "indicator":
        for filename in os.listdir(INDICATOR_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            cache_path = os.path.join(INDICATOR_CACHE_DIR, filename)
            
            try:
                os.remove(cache_path)
                count += 1
            except Exception as e:
                logger.error(f"Error removing indicator cache entry: {filename}, error: {str(e)}")
    
    logger.info(f"Cleared {count} cache entries")
    return count


def get_cache_stats(cache_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics about the cache.
    
    Args:
        cache_type: Type of cache to get statistics for ('dataframe', 'indicator', or None for all)
        
    Returns:
        Dict: Cache statistics
    """
    stats = {}
    
    # Get dataframe cache stats
    if cache_type is None or cache_type == "dataframe":
        file_count = 0
        total_size = 0
        expired_count = 0
        
        for filename in os.listdir(DATAFRAME_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            file_count += 1
            cache_path = os.path.join(DATAFRAME_CACHE_DIR, filename)
            total_size += os.path.getsize(cache_path)
            
            try:
                with open(cache_path, 'r') as f:
                    cache_entry = json.load(f)
                
                # Check if expired
                expires_at = datetime.fromisoformat(cache_entry["expires_at"])
                if datetime.now() > expires_at:
                    expired_count += 1
            
            except Exception:
                # Ignore errors in statistics collection
                pass
        
        # Add dataframe cache operation statistics
        dataframe_stats = {
            "file_count": file_count,
            "expired_count": expired_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0,
            "hits": CACHE_STATS["dataframe"]["hits"],
            "misses": CACHE_STATS["dataframe"]["misses"],
            "stores": CACHE_STATS["dataframe"]["stores"],
            "errors": CACHE_STATS["dataframe"]["errors"],
            "hit_ratio": round(CACHE_STATS["dataframe"]["hits"] / max(CACHE_STATS["dataframe"]["hits"] + CACHE_STATS["dataframe"]["misses"], 1), 2),
            "cache_dir": DATAFRAME_CACHE_DIR
        }
        
        stats["dataframe"] = dataframe_stats
    
    # Get indicator cache stats
    if cache_type is None or cache_type == "indicator":
        file_count = 0
        total_size = 0
        expired_count = 0
        
        for filename in os.listdir(INDICATOR_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            file_count += 1
            cache_path = os.path.join(INDICATOR_CACHE_DIR, filename)
            total_size += os.path.getsize(cache_path)
            
            try:
                with open(cache_path, 'r') as f:
                    cache_entry = json.load(f)
                
                # Check if expired
                expires_at = datetime.fromisoformat(cache_entry["expires_at"])
                if datetime.now() > expires_at:
                    expired_count += 1
            
            except Exception:
                # Ignore errors in statistics collection
                pass
        
        # Add indicator cache operation statistics
        indicator_stats = {
            "file_count": file_count,
            "expired_count": expired_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2) if total_size > 0 else 0,
            "hits": CACHE_STATS["indicator"]["hits"],
            "misses": CACHE_STATS["indicator"]["misses"],
            "stores": CACHE_STATS["indicator"]["stores"],
            "errors": CACHE_STATS["indicator"]["errors"],
            "hit_ratio": round(CACHE_STATS["indicator"]["hits"] / max(CACHE_STATS["indicator"]["hits"] + CACHE_STATS["indicator"]["misses"], 1), 2),
            "cache_dir": INDICATOR_CACHE_DIR
        }
        
        stats["indicator"] = indicator_stats
    
    # Calculate total stats if both types are requested
    if cache_type is None:
        df_stats = stats["dataframe"]
        ind_stats = stats["indicator"]
        
        total_stats = {
            "file_count": df_stats["file_count"] + ind_stats["file_count"],
            "expired_count": df_stats["expired_count"] + ind_stats["expired_count"],
            "total_size_bytes": df_stats["total_size_bytes"] + ind_stats["total_size_bytes"],
            "total_size_mb": df_stats["total_size_mb"] + ind_stats["total_size_mb"],
            "hits": df_stats["hits"] + ind_stats["hits"],
            "misses": df_stats["misses"] + ind_stats["misses"],
            "stores": df_stats["stores"] + ind_stats["stores"],
            "errors": df_stats["errors"] + ind_stats["errors"],
            "hit_ratio": round((df_stats["hits"] + ind_stats["hits"]) / max((df_stats["hits"] + ind_stats["hits"] + df_stats["misses"] + ind_stats["misses"]), 1), 2),
            "cache_dir": CACHE_DIR
        }
        
        stats["total"] = total_stats
    
    return stats


def get_cache_entries(cache_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get information about all cache entries.
    
    Args:
        cache_type: Type of cache to get entries for ('dataframe', 'indicator', or None for all)
        
    Returns:
        List[Dict]: List of cache entries with metadata
    """
    entries = []
    
    # Get dataframe cache entries
    if cache_type is None or cache_type == "dataframe":
        for filename in os.listdir(DATAFRAME_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            cache_path = os.path.join(DATAFRAME_CACHE_DIR, filename)
            key = filename[:-5]  # Remove .json extension
            
            try:
                with open(cache_path, 'r') as f:
                    cache_entry = json.load(f)
                
                # Get basic metadata without the actual data
                expires_at = datetime.fromisoformat(cache_entry["expires_at"])
                is_expired = datetime.now() > expires_at
                
                entry_info = {
                    "key": key,
                    "created_at": cache_entry["created_at"],
                    "expires_at": cache_entry["expires_at"],
                    "is_expired": is_expired,
                    "timeframe": cache_entry.get("timeframe"),
                    "ttl": cache_entry.get("ttl"),
                    "file_size_bytes": os.path.getsize(cache_path),
                    "metadata": cache_entry.get("metadata", {}),
                    "version": cache_entry.get("version", {}),
                    "cache_type": "dataframe"
                }
                
                entries.append(entry_info)
            
            except Exception as e:
                logger.error(f"Error reading dataframe cache entry: {filename}, error: {str(e)}")
    
    # Get indicator cache entries
    if cache_type is None or cache_type == "indicator":
        for filename in os.listdir(INDICATOR_CACHE_DIR):
            if not filename.endswith('.json'):
                continue
            
            cache_path = os.path.join(INDICATOR_CACHE_DIR, filename)
            key = filename[:-5]  # Remove .json extension
            
            try:
                with open(cache_path, 'r') as f:
                    cache_entry = json.load(f)
                
                # Get basic metadata without the actual data
                expires_at = datetime.fromisoformat(cache_entry["expires_at"])
                is_expired = datetime.now() > expires_at
                
                entry_info = {
                    "key": key,
                    "created_at": cache_entry["created_at"],
                    "expires_at": cache_entry["expires_at"],
                    "is_expired": is_expired,
                    "timeframe": cache_entry.get("timeframe"),
                    "ttl": cache_entry.get("ttl"),
                    "indicator_name": cache_entry.get("indicator_name"),
                    "indicator_type": cache_entry.get("indicator_type"),
                    "params_hash": cache_entry.get("params_hash"),
                    "file_size_bytes": os.path.getsize(cache_path),
                    "metadata": cache_entry.get("metadata", {}),
                    "version": cache_entry.get("version", {}),
                    "cache_type": "indicator"
                }
                
                entries.append(entry_info)
            
            except Exception as e:
                logger.error(f"Error reading indicator cache entry: {filename}, error: {str(e)}")
    
    # Sort by creation time (newest first)
    entries.sort(key=lambda x: x["created_at"], reverse=True)
    
    return entries


def store_json_data(
    key: str,
    data: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None,
    timeframe: Optional[str] = None
) -> bool:
    """
    Store JSON data in the cache.
    
    Args:
        key: Unique identifier for the cached data
        data: JSON-serializable data to store
        metadata: Additional information to store with the data
        ttl: Time to live in seconds (default: 5 minutes)
        timeframe: Timeframe of the data, used for determining TTL if not specified
        
    Returns:
        bool: True if successfully stored, False otherwise
        
    Raises:
        CacheError: If there's an error storing the data
    """
    if ttl is None:
        if timeframe and timeframe in DEFAULT_TTL:
            ttl = DEFAULT_TTL[timeframe]
        else:
            ttl = 300  # Default to 5 minutes for JSON data
    
    # Generate cache file path
    cache_path = _get_cache_path(key, cache_type="json")
    
    try:
        # Prepare data for storage
        cache_entry = {
            "data": data,
            "metadata": metadata or {},
            "expires_at": int(time.time() + ttl),
            "created_at": int(time.time()),
            "key": key,
            "library_versions": LIBRARY_VERSIONS
        }
        
        # Write to cache file
        with open(cache_path, 'w') as f:
            # Use file locking to prevent concurrent writes
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(cache_entry, f, indent=2)
            fcntl.flock(f, fcntl.LOCK_UN)
        
        # Update statistics
        CACHE_STATS.setdefault("json", {}).setdefault("stores", 0)
        CACHE_STATS["json"]["stores"] += 1
        
        logger.debug(f"Stored JSON data in cache: {key}")
        return True
        
    except Exception as e:
        # Update error statistics
        CACHE_STATS.setdefault("json", {}).setdefault("errors", 0)
        CACHE_STATS["json"]["errors"] += 1
        
        logger.error(f"Error storing JSON data in cache: {str(e)}")
        raise CacheError(f"Failed to store JSON data in cache: {str(e)}")


def get_cached_json_data(
    key: str,
    default: Optional[Dict[str, Any]] = None,
    allow_expired: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Retrieve JSON data from the cache.
    
    Args:
        key: Unique identifier for the cached data
        default: Value to return if the cache entry doesn't exist
        allow_expired: Whether to return expired cache entries
        
    Returns:
        Dict or None: Cached data, or default if not found
        
    Raises:
        CacheError: If there's an error reading the cache
    """
    # Generate cache file path
    cache_path = _get_cache_path(key, cache_type="json")
    
    # Check if cache file exists
    if not os.path.exists(cache_path):
        # Update miss statistics
        CACHE_STATS.setdefault("json", {}).setdefault("misses", 0)
        CACHE_STATS["json"]["misses"] += 1
        
        logger.debug(f"Cache miss for JSON data: {key}")
        return default
    
    try:
        # Read from cache file
        with open(cache_path, 'r') as f:
            # Use file locking to prevent concurrent reads during a write
            fcntl.flock(f, fcntl.LOCK_SH)
            cache_entry = json.load(f)
            fcntl.flock(f, fcntl.LOCK_UN)
        
        # Check if cache entry has expired
        if not allow_expired and cache_entry.get('expires_at', 0) < time.time():
            # Update miss statistics
            CACHE_STATS.setdefault("json", {}).setdefault("misses", 0)
            CACHE_STATS["json"]["misses"] += 1
            
            logger.debug(f"Cache expired for JSON data: {key}")
            return default
        
        # Update hit statistics
        CACHE_STATS.setdefault("json", {}).setdefault("hits", 0)
        CACHE_STATS["json"]["hits"] += 1
        
        logger.debug(f"Cache hit for JSON data: {key}")
        return cache_entry
        
    except Exception as e:
        # Update error statistics
        CACHE_STATS.setdefault("json", {}).setdefault("errors", 0)
        CACHE_STATS["json"]["errors"] += 1
        
        logger.error(f"Error reading JSON data from cache: {str(e)}")
        raise CacheError(f"Failed to read JSON data from cache: {str(e)}")


def get_cache_status() -> Dict[str, Any]:
    """
    Get comprehensive status information about the cache.
    
    Returns:
        Dict: Cache status information
    """
    try:
        # Get directory info
        cache_dir_info = {
            "cache_dir": CACHE_DIR,
            "dataframe_cache_dir": DATAFRAME_CACHE_DIR,
            "indicator_cache_dir": INDICATOR_CACHE_DIR,
            "json_cache_dir": JSON_CACHE_DIR
        }
        
        # Count files by type
        df_files = [f for f in os.listdir(DATAFRAME_CACHE_DIR) if f.endswith('.json')]
        indicator_files = [f for f in os.listdir(INDICATOR_CACHE_DIR) if f.endswith('.json')]
        json_files = [f for f in os.listdir(JSON_CACHE_DIR) if f.endswith('.json')]
        
        # Calculate total size
        total_size = sum(os.path.getsize(os.path.join(DATAFRAME_CACHE_DIR, f)) for f in df_files)
        total_size += sum(os.path.getsize(os.path.join(INDICATOR_CACHE_DIR, f)) for f in indicator_files)
        total_size += sum(os.path.getsize(os.path.join(JSON_CACHE_DIR, f)) for f in json_files)
        
        # Get files by symbol
        files_by_symbol = {}
        
        # Process dataframe files
        for f in df_files:
            file_path = os.path.join(DATAFRAME_CACHE_DIR, f)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    metadata = data.get("metadata", {})
                    symbol = metadata.get("coin_id", "unknown")
                    files_by_symbol[symbol] = files_by_symbol.get(symbol, 0) + 1
            except:
                pass
        
        # Process indicator files
        for f in indicator_files:
            file_path = os.path.join(INDICATOR_CACHE_DIR, f)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    metadata = data.get("metadata", {})
                    symbol = metadata.get("symbol", "unknown")
                    files_by_symbol[symbol] = files_by_symbol.get(symbol, 0) + 1
            except:
                pass
        
        # Process JSON files
        for f in json_files:
            file_path = os.path.join(JSON_CACHE_DIR, f)
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    metadata = data.get("metadata", {})
                    symbol = metadata.get("symbol", "unknown")
                    files_by_symbol[symbol] = files_by_symbol.get(symbol, 0) + 1
            except:
                pass
        
        # Get files by type
        files_by_type = {
            "dataframe": len(df_files),
            "indicator": len(indicator_files),
            "json": len(json_files)
        }
        
        # Get recent files
        all_files = []
        
        for f in df_files:
            file_path = os.path.join(DATAFRAME_CACHE_DIR, f)
            all_files.append({
                "path": file_path,
                "type": "dataframe",
                "mtime": os.path.getmtime(file_path)
            })
        
        for f in indicator_files:
            file_path = os.path.join(INDICATOR_CACHE_DIR, f)
            all_files.append({
                "path": file_path,
                "type": "indicator",
                "mtime": os.path.getmtime(file_path)
            })
        
        for f in json_files:
            file_path = os.path.join(JSON_CACHE_DIR, f)
            all_files.append({
                "path": file_path,
                "type": "json",
                "mtime": os.path.getmtime(file_path)
            })
        
        # Sort by modification time
        all_files.sort(key=lambda x: x["mtime"], reverse=True)
        recent_files = all_files[:10]  # Get 10 most recent files
        oldest_files = all_files[-10:] if len(all_files) >= 10 else all_files  # Get 10 oldest files (or all if less than 10)
        
        # Get cache statistics
        cache_stats = CACHE_STATS
        
        # Combine all the information
        status = {
            **cache_dir_info,
            "total_files": len(df_files) + len(indicator_files) + len(json_files),
            "total_size": total_size,
            "files_by_type": files_by_type,
            "files_by_symbol": files_by_symbol,
            "recent_files": recent_files,
            "oldest_files": oldest_files,
            "statistics": cache_stats
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        return {
            "error": str(e),
            "cache_dir": CACHE_DIR
        }


def clear_cache_by_age(days: int = 30) -> int:
    """
    Clear cache entries older than the specified number of days.
    
    Args:
        days: Number of days to keep cache entries
        
    Returns:
        int: Number of cache entries removed
    """
    try:
        # Calculate cutoff time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Get list of all cache files
        all_dirs = [DATAFRAME_CACHE_DIR, INDICATOR_CACHE_DIR, JSON_CACHE_DIR]
        removed_count = 0
        
        for cache_dir in all_dirs:
            for filename in os.listdir(cache_dir):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(cache_dir, filename)
                
                # Check modification time
                if os.path.getmtime(file_path) < cutoff_time:
                    try:
                        os.remove(file_path)
                        removed_count += 1
                    except Exception as e:
                        logger.error(f"Error removing cache file {file_path}: {str(e)}")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error clearing cache by age: {str(e)}")
        return 0


def clear_cache_by_type(cache_type: str) -> int:
    """
    Clear all cache entries of a specific type.
    
    Args:
        cache_type: Type of cache to clear ('dataframe', 'indicator', or 'json')
        
    Returns:
        int: Number of cache entries removed
    """
    try:
        # Determine cache directory
        if cache_type == "dataframe":
            cache_dir = DATAFRAME_CACHE_DIR
        elif cache_type == "indicator":
            cache_dir = INDICATOR_CACHE_DIR
        elif cache_type == "json":
            cache_dir = JSON_CACHE_DIR
        elif cache_type == "price":
            # Special case for price data (includes dataframes and current prices)
            return clear_cache_by_type("dataframe") + clear_cache_by_type("json")
        else:
            logger.error(f"Invalid cache type: {cache_type}")
            return 0
        
        removed_count = 0
        
        for filename in os.listdir(cache_dir):
            if not filename.endswith('.json'):
                continue
                
            file_path = os.path.join(cache_dir, filename)
            
            try:
                os.remove(file_path)
                removed_count += 1
            except Exception as e:
                logger.error(f"Error removing cache file {file_path}: {str(e)}")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error clearing cache by type: {str(e)}")
        return 0


def clear_cache_by_symbol(symbol: str) -> int:
    """
    Clear all cache entries for a specific symbol.
    
    Args:
        symbol: Symbol to clear cache for (e.g., 'BTC')
        
    Returns:
        int: Number of cache entries removed
    """
    try:
        # Convert symbol to standardized format
        symbol = symbol.upper()
        
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
        
        # Get CoinGecko ID
        coin_id = symbol_map.get(symbol, symbol.lower())
        
        # Get list of all cache files
        all_dirs = [DATAFRAME_CACHE_DIR, INDICATOR_CACHE_DIR, JSON_CACHE_DIR]
        removed_count = 0
        
        for cache_dir in all_dirs:
            for filename in os.listdir(cache_dir):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(cache_dir, filename)
                
                # Check if file contains the symbol or coin_id
                try:
                    with open(file_path, 'r') as f:
                        cache_entry = json.load(f)
                        
                    metadata = cache_entry.get("metadata", {})
                    file_symbol = metadata.get("symbol", "").upper()
                    file_coin_id = metadata.get("coin_id", "").lower()
                    
                    if file_symbol == symbol or file_coin_id == coin_id:
                        os.remove(file_path)
                        removed_count += 1
                except Exception as e:
                    logger.debug(f"Error checking cache file {file_path}: {str(e)}")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error clearing cache by symbol: {str(e)}")
        return 0


def clear_all_cache() -> int:
    """
    Clear all cache entries.
    
    Returns:
        int: Number of cache entries removed
    """
    try:
        # Get list of all cache files
        all_dirs = [DATAFRAME_CACHE_DIR, INDICATOR_CACHE_DIR, JSON_CACHE_DIR]
        removed_count = 0
        
        for cache_dir in all_dirs:
            for filename in os.listdir(cache_dir):
                if not filename.endswith('.json'):
                    continue
                    
                file_path = os.path.join(cache_dir, filename)
                
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.error(f"Error removing cache file {file_path}: {str(e)}")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error clearing all cache: {str(e)}")
        return 0 