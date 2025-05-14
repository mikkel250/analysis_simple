# BTC-USDT Market Analysis CLI

A command-line tool for cryptocurrency market analysis using CoinGecko API, focusing on BTC-USDT trading pair.

## Features

- Current price information with trend analysis
- Technical indicators (MA, MACD, RSI, Bollinger Bands, etc.)
- Human-readable market insights
- Efficient caching to minimize API calls
- Forecasting capabilities to extrapolate from cached data
- Educational content explaining technical indicators and their interpretation
- Detailed numerical values display alongside indicator interpretations

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
   - `python src/main.py indicator calculate <n>` - Calculate and display a specific technical indicator
   - `python src/main.py indicator list` - List all available technical indicators
   - `python src/main.py indicator clear-cache <n>` - Clear the cache for a specific indicator

2. **Price Commands**
   - `python src/main.py price get` - Get current price and basic analysis for BTC-USDT

3. **Analysis Commands**
   - `python src/main.py analysis run` - Generate comprehensive market analysis with multiple indicators
   - `python src/main.py analysis run --explain` - Include educational content explaining each indicator (now displayed directly with each indicator)
   - `python src/main.py analysis run --forecast` - Include price forecasting in analysis

4. **Status Commands**
   - `python src/main.py status cache` - Display information about the cache
   - `python src/main.py status api` - Display information about API usage

5. **Cache Cleaning Commands**
   - `python src/main.py clean all` - Clean all cached data
   - `python src/main.py clean by-age <days>` - Clean cached data older than specified days
   - `python src/main.py clean by-type <type>` - Clean cached data of a specific type
   - `python src/main.py clean by-symbol <symbol>` - Clean cached data for a specific symbol

6. **Market Analyzer Commands**
   - `python src/main.py analyzer analyze` - Generate comprehensive market analysis for BTC with short timeframe (default)
   - `python src/main.py analyzer analyze --timeframe medium` - Generate analysis with medium timeframe
   - `python src/main.py analyzer analyze --output json` - Output analysis in JSON format
   - `python src/main.py analyzer analyze --output html` - Generate an HTML report with interactive charts
   - `python src/main.py analyzer analyze --save-charts` - Save visualization charts to files

*Note: Both 'analysis run' and 'analyzer analyze' commands now use the same underlying analysis engine for consistent results. The educational content (with --explain flag) now appears directly under each indicator for improved readability.*

Each command has various options that can be viewed by adding `--help` after the command.

## Common Options

- `--symbol`, `-s`: Specify cryptocurrency symbol (default: BTC)
- `--timeframe`, `-t`: Specify timeframe (e.g., 1d, 4h, 1h)
- `--days`, `-d`: Number of days of historical data to fetch
- `--refresh`, `-r`: Force refresh data from API
- `--explain`, `-e`: Include educational content about technical indicators

## Enhanced Indicator Display

The tool now provides detailed numerical values alongside indicator interpretations, offering more transparency into the analysis results:

- **RSI**: Shows the actual RSI value (0-100) together with its interpretation (Overbought/Oversold/Neutral)
- **MACD**: Displays MACD line, Signal line, and Histogram values alongside the interpretation
- **Bollinger Bands**: Shows Upper, Middle, and Lower bands, current price, and position (% from middle band)

Example output:
```
TECHNICAL INDICATORS:

Trend:
  - MACD: Bullish
    ├─ Line: 123.4567
    ├─ Signal: 100.2345
    └─ Histogram: 23.2222

Momentum:
  - RSI: Neutral (Value: 55.23)

Volatility:
  - BOLLINGER: Neutral
    ├─ Upper Band: 25456.78
    ├─ Middle Band: 24789.45
    ├─ Lower Band: 24122.12
    ├─ Price: 24800.00
    └─ Position: 0.43% from middle
```

This enhancement provides both a quick interpretation at a glance and the actual numerical data for traders who want to see the precise values behind the analysis.

## Educational Content

The tool includes comprehensive educational content about technical indicators and market analysis. Enable this feature using the `--explain` or `-e` flag with any analysis command:

```bash
python src/main.py analysis run --explain
python src/main.py analyzer analyze --explain
```

This will display:
- What each indicator measures and how it's calculated
- How to interpret different values (bullish/bearish thresholds)
- Cryptocurrency-specific considerations for each indicator
- Explanations of trend direction, strength, and signals

The educational content now appears directly under each indicator for improved context and readability. This integrated approach ensures that explanations are provided exactly where they're most relevant, making it easier to understand the significance of each indicator's current value.

The educational content is designed to help users understand technical analysis concepts and make more informed trading decisions.

## Market Analyzer

The tool includes a comprehensive market analyzer that can provide detailed analysis in various formats:

```bash
# Generate market analysis for BTC with short timeframe (default)
python src/main.py analyzer analyze

# Generate analysis with medium timeframe
python src/main.py analyzer analyze --timeframe medium

# Output analysis in JSON format
python src/main.py analyzer analyze --output json

# Generate an HTML report with interactive charts
python src/main.py analyzer analyze --output html

# Save visualization charts to files
python src/main.py analyzer analyze --save-charts
```

The market analyzer provides:
- Comprehensive technical analysis with multiple indicators
- Price trend analysis and volatility metrics
- Interactive visualizations (price history, technical, candlestick)
- Indicator interpretations and trading signals
- Exportable reports in various formats (text, JSON, HTML)