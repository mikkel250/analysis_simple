# ATR Position Sizing CLI Tool

This command-line tool calculates position size and stop-loss percentage based on the Average True Range (ATR), entry price, risk amount, and an ATR multiplier. It helps traders manage risk by adjusting trade sizes according to market volatility.

## Background

The calculations and concepts used in this tool are based on the principles outlined in [../docs/ATR_stop_loss_and_position_size_calculations.md](../docs/ATR_stop_loss_and_position_size_calculations.md). 
*Note: The path to the documentation is relative to this README file located in the `src` directory.*

## Features

- Calculates position size based on user-defined risk per trade.
- Calculates stop-loss as a percentage from the entry price.
- Supports both long and short positions.
- Allows customization of risk amount and ATR multiplier (defaults to $100 and 1x respectively).

## Prerequisites

- Python 3.x

## Usage

The tool is run from the command line from the project's root directory using `python src/cli.py`.

```bash
python src/cli.py <entry_price> <atr> <position_type> [options]
```

### Arguments:

- `entry_price`: (Required, float) The entry price of the asset (e.g., `100.0`).
- `atr`: (Required, float) The Average True Range (ATR) value for the asset (e.g., `5.0`).
- `position_type`: (Required, string) The type of position: `long` or `short`.

### Options:

- `--risk_amount <amount>`: (Optional, float) The dollar amount you are willing to risk on this trade. Default: `100.0`.
- `--multiplier <value>`: (Optional, float) The ATR multiplier to determine the stop-loss distance. Default: `1.0` (meaning 1x ATR).
- `-h`, `--help`: Show the help message and exit.

### Examples:

1.  **Long position with default risk and multiplier:**
    Calculate parameters for a long position with an entry price of $100.00 and an ATR of $5.0, using the default risk of $100 and default ATR multiplier of 1.

    ```bash
    python src/cli.py 100.0 5.0 long
    ```

    Expected Output:
    ```
    --- ATR Trade Parameters ---
      Entry Price:          100.0
      ATR:                  5.0
      Position Type:        Long
      Risk Amount:          100.0
      ATR Multiplier:       1.0
      --------------------------
      Calculated Position Size: 20.00000000
      Stop-Loss Percentage:   5.00%
    --- End ---
    ```

2.  **Short position with specified risk and multiplier:**
    Calculate parameters for a short position with an entry price of $25000, an ATR of $300, risking $500 with a 1.5x ATR multiplier for the stop-loss.

    ```bash
    python src/cli.py 25000 300 short --risk_amount 500 --multiplier 1.5
    ```

    Expected Output:
    ```
    --- ATR Trade Parameters ---
      Entry Price:          25000.0
      ATR:                  300.0
      Position Type:        Short
      Risk Amount:          500.0
      ATR Multiplier:       1.5
      --------------------------
      Calculated Position Size: 1.11111111
      Stop-Loss Percentage:   1.80%
    --- End ---
    ```

## Running Tests

Unit tests are located in the `tests` directory and use Python's built-in `unittest` module. To run the tests, navigate to the project root directory and run:

```bash
PYTHONPATH=./src python -m unittest tests.test_atr_calculator
```

This command ensures that the `src` directory is included in the Python path so that the test runner can find the `atr_calculator` module.

## Formulas Used

- **Stop Loss Distance** = `ATR * Multiplier`
- **Position Size** = `Risk per Trade / Stop Loss Distance`
- **Stop Loss Percentage** = (`Stop Loss Distance / Entry Price`) * 100

For more details on these calculations, please refer to [../docs/ATR_stop_loss_and_position_size_calculations.md](../docs/ATR_stop_loss_and_position_size_calculations.md). 