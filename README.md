# BTC-USDT Market Analysis CLI

A command-line tool for cryptocurrency market analysis using CoinGecko API, focusing on BTC-USDT trading pair.

## Features

- Current price information with trend analysis
- Technical indicators (MA, MACD, RSI, Bollinger Bands, etc.)
- Human-readable market insights
- Efficient caching to minimize API calls
- Forecasting capabilities to extrapolate from cached data
- Educational content explaining technical indicators and their interpretation

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd analysis_simple
```

2. Install Python dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file with your CoinGecko API key:
```
API_KEY=your-api-key-here
```

## Usage

```bash
# Get current price with basic analysis
python src/main.py price get

# Get specific indicator
python src/main.py indicator calculate rsi

# List available indicators
python src/main.py indicator list

# Get full market analysis (uses cached data when possible)
python src/main.py analysis run

# Include educational content about technical indicators
python src/main.py analysis run --explain

# Include forecasting in analysis
python src/main.py analysis run --forecast

# Force refresh data (warns about API call limits)
python src/main.py analysis run --refresh

# Show data age and API call count
python src/main.py status cache
python src/main.py status api

# Clean cache
python src/main.py clean all
```

## Configuration

You can customize the tool behavior by editing the `.env` file:

```
# CoinGecko API key
API_KEY=your-api-key-here

# Default trading pair
DEFAULT_SYMBOL=BTC

# Default timeframe
DEFAULT_TIMEFRAME=1d
```

## Commands 

The tool has the following main commands:

1. **Indicator Commands**
   - `python src/main.py indicator calculate <name>` - Calculate and display a specific technical indicator
   - `python src/main.py indicator list` - List all available technical indicators
   - `python src/main.py indicator clear-cache <name>` - Clear the cache for a specific indicator

2. **Price Commands**
   - `python src/main.py price get` - Get current price and basic analysis for BTC-USDT

3. **Analysis Commands**
   - `python src/main.py analysis run` - Generate comprehensive market analysis with multiple indicators
   - `python src/main.py analysis run --explain` - Include educational content explaining each indicator
   - `python src/main.py analysis run --forecast` - Include price forecasting in analysis

4. **Status Commands**
   - `python src/main.py status cache` - Display information about the cache
   - `python src/main.py status api` - Display information about API usage

5. **Cache Cleaning Commands**
   - `python src/main.py clean all` - Clean all cached data
   - `python src/main.py clean by-age <days>` - Clean cached data older than specified days
   - `python src/main.py clean by-type <type>` - Clean cached data of a specific type
   - `python src/main.py clean by-symbol <symbol>` - Clean cached data for a specific symbol

Each command has various options that can be viewed by adding `--help` after the command.

## Common Options

- `--symbol`, `-s`: Specify cryptocurrency symbol (default: BTC)
- `--timeframe`, `-t`: Specify timeframe (e.g., 1d, 4h, 1h)
- `--days`, `-d`: Number of days of historical data to fetch
- `--refresh`, `-r`: Force refresh data from API
- `--explain`, `-e`: Include educational content about technical indicators

## Educational Content

The tool includes comprehensive educational content about technical indicators and market analysis. Enable this feature using the `--explain` or `-e` flag with any analysis command:

```bash
python src/main.py analysis run --explain
```

This will display:
- What each indicator measures and how it's calculated
- How to interpret different values (bullish/bearish thresholds)
- Cryptocurrency-specific considerations for each indicator
- Explanations of trend direction, strength, and signals

The educational content is designed to help users understand technical analysis concepts and make more informed trading decisions.