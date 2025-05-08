# API Configuration

This module provides a secure way to manage API credentials for various exchanges used in the application.

## Setup Options

### Option 1: Environment Variables

Set these environment variables in your shell or in a `.env` file in the project root:

```bash
# Binance API credentials
BINANCE_API_KEY="your_binance_api_key_here"
BINANCE_API_SECRET="your_binance_api_secret_here"

# Bybit API credentials (if needed)
BYBIT_API_KEY="your_bybit_api_key_here"
BYBIT_API_SECRET="your_bybit_api_secret_here"
```

The application will automatically load these variables if present.

### Option 2: CLI Command

Use the built-in CLI command to set up API credentials:

```bash
# View current configuration status
python -m src.main config api --list

# Set up Binance API credentials
python -m src.main config api --exchange binance

# Set up credentials for another exchange
python -m src.main config api --exchange bybit
```

The CLI will prompt for your API key and secret, and store them securely in the configuration file.

### Option 3: Configuration File

Manually create a file at `config/api_keys.json` with the following structure:

```json
{
  "binance": {
    "api_key": "your_binance_api_key_here",
    "api_secret": "your_binance_api_secret_here"
  },
  "bybit": {
    "api_key": "your_bybit_api_key_here",
    "api_secret": "your_bybit_api_secret_here"
  }
}
```

## Supported Exchanges

The configuration system currently supports the following exchanges:

- Binance
- Bybit
- Coinbase
- Kraken
- KuCoin
- FTX

## Security Notes

- API keys are stored in plaintext in the configuration file and environment variables. Use appropriate file permissions.
- Never commit your API keys to version control.
- Consider using environment variables for production deployments.
- The application will never log your full API keys, only masked versions for verification purposes. 