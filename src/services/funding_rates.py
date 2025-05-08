from typing import List, Dict, Optional, Any


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