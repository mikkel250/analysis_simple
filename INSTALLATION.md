# Installation Guide

This document provides instructions for installing the required dependencies for the BTC-USDT market analysis CLI tool.

## Python Dependencies

The project requires Python 3.7+ and the following Python packages:

```bash
# Install from requirements.txt
pip install -r requirements.txt
```

This will install:
- pandas-ta - Technical analysis library for calculating indicators
- pandas - Data manipulation and analysis library
- numpy - Numerical computing library
- pycoingecko - CoinGecko API client
- typer - CLI interface library
- tabulate - Table formatting library
- rich - Terminal formatting library
- statsmodels - Time series forecasting library

## Verify Installation

You can verify the installation by running:

```bash
python src/main.py --help
```

If successful, you should see the available commands and options for the CLI tool.

## Troubleshooting

### pandas-ta Installation Issues

If you encounter issues installing pandas-ta:

1. Ensure you have the latest pip:
```bash
pip install --upgrade pip
```

2. Install pandas and numpy first:
```bash
pip install pandas numpy
```

3. Then install pandas-ta:
```bash
pip install pandas-ta
```

### Type Checking Issues

If you encounter type checking issues, install the required type stubs:

```bash
pip install types-tabulate
```

### Mac-specific Installation

On macOS, you might need to use:

```bash
pip3 install -r requirements.txt
```

Or if using Conda:

```bash
conda install -c conda-forge pandas-ta
``` 