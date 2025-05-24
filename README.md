# BTC-USDT Market Analysis CLI

A command-line tool for cryptocurrency market analysis using the **OKX API (via the `ccxt` library)**, focusing on BTC-USDT trading pair. It fetches market data (prices, OHLCV) and open interest information from OKX.

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
cd analysis_simple # Or your project directory name
```

2. Install Python dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file with your **OKX API credentials**. The tool uses `ccxt` which generally requires `apiKey` and `secret`. You may also need a `password` (passphrase) if you set one for your OKX API key. Example:
```
# .env file
OKX_API_KEY="your-okx-api-key"
OKX_SECRET_KEY="your-okx-secret-key"
OKX_PASSWORD="your-okx-api-passphrase" # Optional, only if you set one
```
Ensure your `src/config/api_config.py` is set up to read these environment variables and provide them to `ccxt` in the format it expects (typically a dictionary with `'apiKey'`, `'secret'`, `'password'`).

## Usage

The primary command for detailed market analysis is `analyzer analyze`.

```bash
# Get full market analysis (default symbol BTC/USDT, default timeframe 1d)
python src/main.py analyzer analyze

# Analyze a different symbol
python src/main.py analyzer analyze ETH/USDT

# Specify a timeframe (e.g., 1h, 4h, 15m)
python src/main.py analyzer analyze --timeframe 4h

# Include detailed educational explanations for indicators
python src/main.py analyzer analyze --explain

# Output analysis in JSON format
python src/main.py analyzer analyze --output json

# Generate an HTML report with interactive charts
python src/main.py analyzer analyze --output html

# Force refresh data from API (use with caution due to rate limits)
python src/main.py analyzer analyze --refresh # Note: refresh applies to underlying data fetch

# --- Other Utility Commands ---
# Get current price for default symbol
python src/main.py price get

# Get specific indicator for default symbol and timeframe
python src/main.py indicator calculate rsi

# List available indicators
python src/main.py indicator list

# Show data age and API call count
python src/main.py status cache
python src/main.py status api

# Clean cache
python src/main.py clean all
```

## Configuration

You can customize the tool behavior by editing the `.env` file:

```
# OKX API Credentials
OKX_API_KEY="your-okx-api-key"
OKX_SECRET_KEY="your-okx-secret-key"
OKX_PASSWORD="your-okx-api-passphrase" # If you configured one for your API key

# Default trading pair (e.g., BTC/USDT, ETH/USDT)
DEFAULT_SYMBOL=BTC/USDT 

# Default timeframe for analysis (e.g., 1d, 4h, 1h, 15m)
DEFAULT_TIMEFRAME=1d
```

## Commands 

The tool has the following main commands:

1.  **Market Analyzer (Primary Analysis Command)**
    *   `python src/main.py analyzer analyze [SYMBOL]`
        *   Performs a comprehensive market analysis for the given symbol (default: `BTC/USDT`).
        *   `--timeframe <TIMEFRAME_STRING>`: Specify the analysis timeframe (e.g., `15m`, `1h`, `4h`, `1d`, `1w`). Default: `1d`.
        *   `--output <FORMAT>`: Specify output format (`text`, `json`, `html`, `jsf`). Default: `text`.
        *   `--explain` or `-e`: Include detailed educational explanations for each indicator and analysis component.
        *   `--refresh` or `-r`: Force refresh of underlying market data from the API.

2.  **Indicator Commands (Utilities)**
    *   `python src/main.py indicator calculate <INDICATOR_NAME>`: Calculate and display a specific technical indicator for the default symbol/timeframe.
    *   `python src/main.py indicator list`: List all available technical indicators.
    *   `python src/main.py indicator clear-cache <INDICATOR_NAME>`: Clear the cache for a specific indicator.

3.  **Price Commands (Utilities)**
    *   `python src/main.py price get`: Get current price and basic analysis for the default symbol.

4.  **Status Commands (Utilities)**
    *   `python src/main.py status cache`: Display information about the cache.
    *   `python src/main.py status api`: Display information about API usage.

5.  **Cache Cleaning Commands (Utilities)**
    *   `python src/main.py clean all`: Clean all cached data.
    *   `python src/main.py clean by-age <DAYS>`: Clean cached data older than specified days.
    *   `python src/main.py clean by-type <TYPE>`: Clean cached data of a specific type.
    *   `python src/main.py clean by-symbol <SYMBOL>`: Clean cached data for a specific symbol.

6.  **Configuration Commands (Utilities)**
    *   `python src/main.py config show`: Display the current configuration.
    *   `python src/main.py config set <KEY> <VALUE>`: Set a configuration value.

*Note: The `--explain` flag enhances the output by providing educational context directly alongside each relevant piece of information, making the analysis easier to understand.* 

Each command has various options that can be viewed by adding `--help` after the command.

## Common Options for `analyzer analyze`

- `SYMBOL`: (Positional Argument) Specify cryptocurrency symbol (e.g., `BTC/USDT`, `ETH/BTC`). Defaults to `DEFAULT_SYMBOL` from `.env` if not provided.
- `--timeframe`, `-t`: Specify timeframe string (e.g., `15m`, `1h`, `1d`, `1w`). Allowed values are `1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d, 1w`.
- `--output`, `-o`: Specify output format (`text`, `json`, `html`, `jsf`).
- `--explain`, `-e`: Include detailed educational content about technical indicators and analysis components.
- `--refresh`, `-r`: Force refresh data from API for the current analysis.
- `--api-key`: Optionally pass API key directly (less secure, prefer `.env`).

## Enhanced Indicator Display

The tool provides detailed numerical values alongside indicator interpretations, offering transparency:

- **RSI**: Shows the actual RSI value and its interpretation (Overbought/Oversold/Neutral).
- **MACD**: Displays MACD line, Signal line, and Histogram values with interpretation.
- **Bollinger Bands**: Shows Upper, Middle, and Lower bands, current price, and position relative to bands.

Example output from `analyzer analyze`:
```
[Technical Indicators]

RSI (14): Neutral
  └─ Value: 55.23

MACD (12,26,9): Bullish
  └─ Line: 123.45, Signal: 100.23, Histogram: 23.22

Bollinger Bands (20,2.0): Neutral
  └─ Upper: 25456.78, Middle: 24789.45, Lower: 24122.12, Close: 24800.00, Percent: 0.04
```

This provides both a quick interpretation and the precise numerical data.

## Educational Content with `--explain`

Using the `--explain` or `-e` flag with `analyzer analyze` integrates educational content directly with the analysis output:

```bash
python src/main.py analyzer analyze --explain
```

This will display:
- What each indicator measures and its typical calculation.
- How to interpret different values and signals (bullish/bearish thresholds).
- Cryptocurrency-specific considerations where applicable.
- Explanations of trend direction, strength, and derived signals.
- **Candlestick pattern explanations**: If candlestick patterns (e.g., Doji, Engulfing, Hammer, Harami, Morning Star) are detected in the analysis, their meaning and typical market implications are explained directly in the output.

The educational content is designed to help users understand technical analysis concepts and make more informed trading decisions. The explanations appear directly under each relevant indicator, pattern, or analysis section for improved context and readability.

## Market Analyzer Output Formats

The `analyzer analyze` command supports multiple output formats via the `--output` option:

- `text` (default): Richly formatted text output for the console.
- `json`: Machine-readable JSON output of the full analysis results.
- `html`: An HTML report, often including interactive charts (if visualizations are generated).
- `jsf` (JSON File): Saves the JSON output to a file named like `analysis_SYMBOL_TIMEFRAME.jsf`.

Example:
```bash
# Generate market analysis for ETH/USDT (1h) and save as HTML
python src/main.py analyzer analyze ETH/USDT --timeframe 1h --output html

# Get JSON output for BTC/USDT (15m)
python src/main.py analyzer analyze --timeframe 15m --output json
```