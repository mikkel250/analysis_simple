import ccxt
import traceback

def test_bybit_open_interest():
    try:
        exchange = ccxt.bybit()
        # Try fetching open interest for BTC
        result = exchange.fetch_open_interest('BTC/USDT:USDT', None)
        print("Successfully fetched open interest from Bybit:")
        print(result)
        return result
    except Exception as e:
        print(f"Error fetching from Bybit: {e}")
        traceback.print_exc()
        return None

def test_okx_open_interest():
    try:
        exchange = ccxt.okx()
        # Try fetching open interest for BTC
        result = exchange.fetch_open_interest('BTC/USDT:USDT', None)
        print("Successfully fetched open interest from OKX:")
        print(result)
        return result
    except Exception as e:
        print(f"Error fetching from OKX: {e}")
        traceback.print_exc()
        return None

def test_kucoin_open_interest():
    try:
        exchange = ccxt.kucoinfutures()
        # Try fetching open interest for BTC
        result = exchange.fetch_open_interest('BTC/USDT:USDT', None)
        print("Successfully fetched open interest from KuCoin Futures:")
        print(result)
        return result
    except Exception as e:
        print(f"Error fetching from KuCoin Futures: {e}")
        traceback.print_exc()
        return None

def test_open_interest_history():
    try:
        exchange = ccxt.bybit()
        # Try fetching open interest history for BTC
        result = exchange.fetch_open_interest_history('BTC/USDT:USDT', '1h', None, 10)
        print("Successfully fetched open interest history from Bybit:")
        print(result)
        return result
    except Exception as e:
        print(f"Error fetching open interest history: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Testing open interest data from different exchanges...")
    print("=" * 80)
    test_bybit_open_interest()
    print("=" * 80)
    test_okx_open_interest()
    print("=" * 80)
    test_kucoin_open_interest()
    print("=" * 80)
    test_open_interest_history() 