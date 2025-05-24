from typing import List, Dict, Optional, Any
import ccxt
import logging

logger = logging.getLogger(__name__)


def fetch_funding_rates(symbol: str, api: Optional[Any] = None) -> List[Dict[str, Any]]:
    """
    Fetch funding rates for a given perpetual futures symbol.
    Args:
        symbol: The trading symbol (e.g., 'BTCUSDT')
        api: Optional API object for testability (must have get_funding_rates method)
    Returns:
        List of funding rate dicts with 'timestamp' and 'rate' keys
    Raises:
        Exception if API call fails or data is missing
    """
    if api is not None:
        return api.get_funding_rates(symbol)
    # Real API integration would go here (Binance, Bybit, etc.)
    raise NotImplementedError("Real API integration not implemented. Pass a mock API for testing.")


def analyze_funding_rates(rates: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Analyze funding rates: compute current, average, and history.
    Args:
        rates: List of funding rate dicts with 'timestamp' and 'rate'
    Returns:
        Dict with 'current', 'average', and 'history' (list of rates)
    """
    if not rates:
        return {'current': None, 'average': None, 'history': []}
    sorted_rates = sorted(rates, key=lambda r: r['timestamp'])
    history = [r['rate'] for r in sorted_rates]
    current = history[-1]
    average = sum(history) / len(history)
    return {'current': current, 'average': average, 'history': history}


def fetch_okx_funding_rate(symbol: str, credentials: Optional[dict] = None) -> Dict[str, Any]:
    """
    Fetch the current funding rate for a given symbol from OKX using ccxt.
    Args:
        symbol: Trading pair symbol (e.g., 'BTC-USDT' or 'BTC/USDT')
        credentials: Optional dict with 'apiKey' and 'secret' (not required for public endpoints)
    Returns:
        Dict with 'funding_rate', 'timestamp', and 'info' (raw ccxt response)
    """
    # Normalize symbol to CCXT format: 'BTC/USDT:USDT'
    try:
        if '-' in symbol:
            base, quote = symbol.split('-')
            ccxt_symbol = f"{base}/{quote}:USDT"
        elif '/' in symbol:
            base, quote = symbol.split('/')
            ccxt_symbol = f"{base}/{quote}:USDT"
        else:
            # Fallback: assume BTCUSDT -> BTC/USDT:USDT
            base, quote = symbol[:3], symbol[3:]
            ccxt_symbol = f"{base}/{quote}:USDT"
        # Initialize OKX exchange
        if credentials:
            exchange = ccxt.okx({
                'apiKey': credentials.get('apiKey', ''),
                'secret': credentials.get('secret', ''),
                'enableRateLimit': True
            })
        else:
            exchange = ccxt.okx({'enableRateLimit': True})
        # Fetch funding rate
        result = exchange.fetch_funding_rate(ccxt_symbol)
        return {
            'funding_rate': result.get('fundingRate'),
            'timestamp': result.get('timestamp'),
            'info': result
        }
    except Exception as e:
        logger.error(f"Error fetching OKX funding rate for {symbol}: {e}")
        return {'funding_rate': None, 'timestamp': None, 'info': {'error': str(e)}} 