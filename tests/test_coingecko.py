"""
Test script to verify pycoingecko installation and basic functionality.
"""

try:
    # Import required libraries
    from pycoingecko import CoinGeckoAPI
    import time
    
    # Create a CoinGecko API client
    cg = CoinGeckoAPI()
    
    # Test basic API connection
    ping = cg.ping()
    print(f"\n✅ CoinGecko API connection test: {ping}")
    
    # Get Bitcoin price in USD
    bitcoin_price = cg.get_price(ids='bitcoin', vs_currencies='usd')
    print(f"\nBitcoin price: {bitcoin_price}")
    
    # Test getting historical market data for BTC
    print("\nFetching historical market data for Bitcoin (last 7 days)...")
    # Use UNIX timestamp for 7 days ago (in seconds)
    from_timestamp = int(time.time()) - 7 * 24 * 60 * 60
    
    # Get daily market data for the last 7 days
    historical_data = cg.get_coin_market_chart_range_by_id(
        id='bitcoin',
        vs_currency='usd',
        from_timestamp=from_timestamp,
        to_timestamp=int(time.time())
    )
    
    # Print the number of price data points received
    print(f"Received {len(historical_data['prices'])} price data points")
    print(f"First price point: {historical_data['prices'][0]}")
    print(f"Last price point: {historical_data['prices'][-1]}")
    
    # Print version information
    import pkg_resources
    pycoingecko_version = pkg_resources.get_distribution("pycoingecko").version
    print(f"\nPyCoinGecko version: {pycoingecko_version}")
    
    print("\n✅ CoinGecko API client test successful!")
    
except ImportError as e:
    print(f"❌ Installation test failed: {e}")
    print("Please install the required library with: pip install pycoingecko")
except Exception as e:
    print(f"❌ Test failed with error: {e}") 