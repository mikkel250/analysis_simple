# Financial Analysis Jupyter Kernel

A custom Jupyter kernel for financial market data analysis with built-in tools and automatic configuration.

## Features

- **Pre-loaded Financial Libraries**: Comes with pandas, numpy, matplotlib, plotly, yfinance, and pandas_ta pre-imported
- **Trading Style Modifiers**: Switch between different trading timeframes with magic commands (%short, %medium, %long)
- **Market Data Tools**: Built-in functions for fetching, processing, and visualizing financial data
- **Auto-Execution**: Automatically runs cells when a notebook is loaded
- **Project Path Integration**: Automatically adds your project path to the Python path

## Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Installation Steps

1. Clone this repository or download the source code
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the installation script:
   ```bash
   python -m src.jupyter.kernel.install --user
   ```

### Installation Options

The installation script supports several options:

```bash
python -m src.jupyter.kernel.install --help
```

- `--user`: Install for the current user only (default)
- `--sys`: Install system-wide (requires administrator privileges)
- `--prefix PATH`: Install to a specific prefix path
- `--uninstall`: Uninstall the kernel
- `-v, --verbose`: Enable verbose output

## Usage

### Starting a Notebook with the Kernel

1. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

2. Create a new notebook and select "Financial Analysis" as the kernel

### Pre-loaded Libraries

The following libraries are automatically imported and available in your notebook:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
import pandas_ta as ta
```

In addition, the following modules from the kernel are also imported:

```python
from src.jupyter.kernel import market_data
from src.jupyter.kernel import trading_styles
```

### Trading Style Modifiers

The kernel provides three magic commands to quickly switch between different trading timeframes:

- `%short`: For short-term trading (5m, 15m, 30m timeframes)
  ```python
  %short
  # or for more details
  %short --verbose
  ```

- `%medium`: For medium-term trading (1h, 4h, 1d timeframes)
  ```python
  %medium
  ```

- `%long`: For long-term trading (1d, 1wk, 1mo timeframes)
  ```python
  %long
  ```

### Market Data Functions

The kernel provides various functions for working with financial market data:

#### Data Fetching

```python
# Fetch data for a single stock
data = market_data.get_stock_data(symbol="AAPL", period="1y", interval="1d")

# Fetch data for multiple stocks
data = market_data.get_multiple_stocks(symbols=["AAPL", "MSFT", "GOOGL"], period="6mo")

# Fetch market index data
sp500 = market_data.get_market_index(index_symbol="^GSPC", period="1y")

# Fetch data based on current trading style settings
from src.jupyter.kernel.trading_styles import fetch_data_for_current_style
data = fetch_data_for_current_style(symbol="AAPL")
```

#### Data Processing

```python
# Calculate returns
data_with_returns = market_data.calculate_returns(data, return_type="log")

# Calculate rolling statistics (SMA, standard deviation)
data_with_stats = market_data.calculate_rolling_statistics(data, windows=[20, 50, 200])

# Add technical indicators (RSI, MACD, Bollinger Bands)
data_with_indicators = market_data.add_technical_indicators(data)

# Apply analysis based on current trading style
from src.jupyter.kernel.trading_styles import apply_analysis_for_current_style
data_analyzed = apply_analysis_for_current_style(data)
```

#### Visualization

```python
# Create a price history plot (matplotlib)
fig = market_data.plot_price_history(data, title="Stock Price History")
plt.show()

# Create a candlestick chart (plotly)
fig = market_data.plot_candlestick(data, include_volume=True)
fig.show()

# Create a technical analysis chart (plotly)
fig = market_data.plot_technical_analysis(data)
fig.show()

# Create a plot based on current trading style
from src.jupyter.kernel.trading_styles import plot_for_current_style
fig = plot_for_current_style(data_analyzed)
fig.show()

# Compare multiple stocks
data, fig = market_data.compare_stocks(
    symbols=["AAPL", "MSFT", "GOOGL"], 
    period="1y", 
    normalize=True
)
fig.show()
```

#### Performance Analysis

```python
# Get performance summary
performance = market_data.get_performance_summary(data)
print(performance)
```

### Auto-Execution

The kernel automatically executes cells when a notebook is opened. You can control this behavior by adding tags to cells:

- Cells with the `autorun` tag will automatically execute
- Cells with the `skip` or `norun` tag will be skipped during auto-execution

To add tags to a cell in Jupyter Notebook, click on the cell and then:
1. View → Cell Toolbar → Tags
2. Add the appropriate tag

In JupyterLab:
1. Click the gear icon on the right side of a cell
2. Add the appropriate tag

## Example Workflows

### Basic Stock Analysis

```python
# Set trading style to medium-term
%medium

# Fetch data for Apple
aapl_data = market_data.get_stock_data(symbol="AAPL", period="1y")

# Add technical indicators
aapl_data = market_data.add_technical_indicators(aapl_data)

# Create a technical analysis chart
fig = market_data.plot_technical_analysis(aapl_data)
fig.show()

# Get performance summary
performance = market_data.get_performance_summary(aapl_data)
print(performance)
```

### Comparing Multiple Stocks

```python
# Set trading style to long-term
%long

# Compare tech stocks
tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
data, fig = market_data.compare_stocks(
    symbols=tech_stocks, 
    period="5y", 
    normalize=True
)
fig.show()

# Get performance for each stock
for symbol in tech_stocks:
    stock_data = market_data.get_stock_data(symbol=symbol, period="5y")
    performance = market_data.get_performance_summary(stock_data)
    print(f"{symbol}: Total Return: {performance['total_return_pct']:.2f}%, "
          f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
```

### Scalping/Day Trading Analysis

```python
# Set trading style to short-term
%short

# Fetch intraday data for a stock
data = market_data.get_stock_data(
    symbol="TSLA", 
    interval="5m", 
    period="1d"
)

# Add technical indicators
data = market_data.add_technical_indicators(data)

# Create a candlestick chart
fig = market_data.plot_candlestick(data)
fig.show()
```

## Troubleshooting

### Common Issues

#### Kernel Not Available in Jupyter

If the Financial Analysis kernel doesn't appear in the list of available kernels:

1. Check if the kernel is installed:
   ```bash
   jupyter kernelspec list
   ```

2. If the kernel is not listed, reinstall it:
   ```bash
   python -m src.jupyter.kernel.install --user -v
   ```

3. Ensure your Jupyter installation is working correctly:
   ```bash
   jupyter --version
   ```

#### Import Errors

If you encounter import errors when using the kernel:

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Check if the project path is correctly set. You can verify this by running:
   ```python
   import sys
   print(sys.path)
   ```

#### Auto-Execution Not Working

If cells don't automatically execute when the notebook is opened:

1. Make sure cells have the `autorun` tag
2. Check the browser console for any JavaScript errors
3. Try refreshing the page or restarting the kernel

### Uninstalling the Kernel

To remove the kernel:

```bash
python -m src.jupyter.kernel.install --uninstall
```

## Development

### Project Structure

- `src/jupyter/kernel/`: Main package directory
  - `__init__.py`: Package initialization
  - `kernel.py`: Core kernel implementation
  - `market_data.py`: Market data functions
  - `trading_styles.py`: Trading style modifiers
  - `install.py`: Installation script
  - `kernelspec/`: Kernel specification files
    - `kernel.json`: Kernel configuration

### Creating a Custom Build

To modify the kernel for your own needs:

1. Clone the repository
2. Make your changes to the source files
3. Update the version in `__init__.py`
4. Reinstall the kernel with your changes:
   ```bash
   python -m src.jupyter.kernel.install --user
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 