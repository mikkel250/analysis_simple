# ATR Position Sizing CLI Tool

This command-line tool calculates ATR-based stop-loss and position size parameters. It can be used for:
1. Calculating an ideal new position based on ATR, entry price, and target risk.
2. Analyzing an existing position to determine its stop-loss based on a target risk, and comparing it to an ideal ATR-based position for scaling guidance.

## Background

The calculations and concepts used in this tool are based on the principles outlined in [../docs/ATR_stop_loss_and_position_size_calculations.md](../docs/ATR_stop_loss_and_position_size_calculations.md).
*Note: The path to the documentation is relative to this README file located in the `src` directory.*

## Features

- Calculates stop-loss price and percentage.
- Calculates ideal ATR-based position size (units and $ value).
- For existing positions: calculates stop-loss to meet a specific dollar risk and provides ATR-based scaling advice.
- Supports both long and short positions.
- Allows customization of risk amount and ATR multiplier.

## Prerequisites

- Python 3.x

## Usage

The tool is run from the command line from the project's root directory using `python src/cli.py`.

```bash
python src/cli.py <entry_price_or_avg_cost> <position_type> --atr <value>|--atr-from-symbol <symbol>|--atr-from-file <filepath> [OPTIONS]
```

### Main Arguments:

- `entry_price_or_avg_cost`: (Required, float)
    - If `--existing-units` is NOT used: This is the intended entry price for a new position.
    - If `--existing-units` IS used: This is the average entry price (cost basis) of your existing units.
- `position_type`: (Required, string) The type of position: `long` or `short`.

### ATR Specification (Choose ONE method):

1.  **Direct ATR Input:**
    - `--atr <value>`: (float) Directly provide the Average True Range (ATR) value (e.g., `0.0058`).

2.  **Calculate ATR from Symbol:**
    - `--atr-from-symbol <SYMBOL>`: (string) Symbol to fetch historical data for ATR calculation (e.g., `BTC`, `ETH-USD`).
    - `--atr-period <period>`: (Optional, int) Period for ATR calculation. Default: `14`. Used with `--atr-from-symbol` or when using `--csv-ohlc` with `--atr-from-file`.
    - `--atr-days <days>`: (Optional, int) Number of days of historical data to fetch. Default: `90`. Used with `--atr-from-symbol`.
    - `--atr-vs-currency <currency>`: (Optional, string) The quote currency for fetching historical data. Default: `usd`. Used with `--atr-from-symbol`.

3.  **Calculate/Use ATR from CSV File:**
    - `--atr-from-file <FILEPATH>`: (string) Path to a CSV file.
    - **EITHER** provide OHLC columns for calculation:
        - `--csv-ohlc <OPEN_COL> <HIGH_COL> <LOW_COL> <CLOSE_COL>`: (string, 4 values) Names of the Open, High, Low, and Close columns in your CSV file.
        - `--atr-period <period>`: (Optional, int) Period for ATR calculation if using `--csv-ohlc`. Default: `14`.
    - **OR** provide a pre-calculated ATR column:
        - `--csv-atr-column <COLUMN_NAME>`: (string) Name of the column in your CSV that already contains ATR values.
    - `--csv-date-column <DATE_COLUMN_NAME>`: (Optional, string) Name of the date/timestamp column in your CSV. Recommended if your CSV is a time series, as it helps ensure correct data parsing and ordering.

### Other Options:

- `--risk_amount <amount>`: (Optional, float) The total dollar amount you are willing to risk on the trade. Default: `100.0`.
- `--multiplier <value>`: (Optional, float) The ATR multiplier used for the ATR-based stop-loss distance calculation. Default: `1.0` (meaning 1x ATR).
- `--existing-units <units>`: (Optional, float) The number of asset units you currently hold. If provided, `entry_price_or_avg_cost` is treated as their average cost.
- `-h`, `--help`: Show the help message and exit.

### Examples:

**1. Calculate Parameters for a NEW Position (Direct ATR):**

Calculate for a new long position with an entry price of $0.10379, ATR of $0.0058, risking $100 with a 1x ATR multiplier.

```bash
python src/cli.py 0.10379 long --atr 0.0058 --risk_amount 100
```

Expected Output (similar to before):
```
--- ATR Position Sizing Tool ---
  Inputs:
    ATR:                                        0.00580000
    Position Type:                              Long
    Target Risk Amount:                         $100.00
    ATR Multiplier:                             1.00x
    Entry Price for New Position:             $0.10379000
  -----------------------------------------------------

  ATR-Based Calculation for NEW Position
    Position Size ($):                        $1789.48
    Position Size (Asset Units):            17241.37931034
    Stop-Loss Distance (ATR-based):         $0.00580000
    Stop-Loss Price:                        $0.09799000
    Stop-Loss Percentage:                   5.59%
  -----------------------------------------------------

--- End ---
```

**2. Calculate for a NEW Position (ATR from Symbol):**

Calculate for a new short position in BTC-USD with an entry price of $60000, risking $200, using a 1.5x ATR multiplier. ATR will be calculated from the last 90 days of daily data for BTC-USD using a 14-period ATR.

```bash
python src/cli.py 60000 short --atr-from-symbol BTC-USD --risk_amount 200 --multiplier 1.5
```
*(Output will vary based on live ATR calculation)*

**3. Calculate for a NEW Position (ATR from CSV - OHLC columns):**

Your `my_data.csv` has columns `Date`, `OpenPrice`, `HighPrice`, `LowPrice`, `ClosePrice`. Calculate ATR (10-period) for a long position with entry $50.

```bash
python src/cli.py 50 long --atr-from-file my_data.csv --csv-ohlc OpenPrice HighPrice LowPrice ClosePrice --csv-date-column Date --atr-period 10
```

**4. Calculate for a NEW Position (ATR from CSV - Pre-calculated ATR column):**

Your `my_data_with_atr.csv` has a column `CalculatedATR` and `Timestamp`.

```bash
python src/cli.py 50 long --atr-from-file my_data_with_atr.csv --csv-atr-column CalculatedATR --csv-date-column Timestamp
```

**5. Analyze an EXISTING Position (Direct ATR):**

You hold 8630 units of an asset with an average entry price of $0.10379. The current ATR is $0.0058. You want to risk a total of $100 on this position and are using a 1x ATR multiplier for reference.

```bash
python src/cli.py 0.10379 long --atr 0.0058 --risk_amount 100 --existing-units 8630
```

Expected Output (similar to before):
```
--- ATR Position Sizing Tool ---
  Inputs:
    ATR:                                        0.00580000
    Position Type:                              Long
    Target Risk Amount:                         $100.00
    ATR Multiplier:                             1.00x
    Existing Units:                             8630.00000000
    Average Entry Price (of existing units):    $0.10379000
  -----------------------------------------------------

  Part 1: Analysis of EXISTING Position (to meet Target Risk Amount)
    Target Risk:                              $100.00
    Calculated SL Distance per Unit:          $0.01158749
    Stop-Loss Price for Existing Position:    $0.09220251
    Stop-Loss Percentage for Existing Position: 11.16%
    Value of Existing Position:               $895.71
  -----------------------------------------------------

  Part 2: Ideal ATR-Based Position (for Target Risk Amount, using existing avg entry as ref price)
    ATR-based SL Distance per Unit:           $0.00580000
    Ideal Position Size (Units, ATR-based):   17241.37931034
    Ideal Position Value ($, ATR-based):      $1789.48
    Stop-Loss Price (ATR-based):              $0.09799000
    Stop-Loss Percentage (ATR-based):         5.59%
    Units to ADD to reach ATR-Ideal:        8611.37931034 (at current market or target price)
  -----------------------------------------------------

--- End ---
```

## Running Tests

Unit tests for the core calculation logic are located in the `tests` directory and use Python's built-in `unittest` module. To run the tests, navigate to the project root directory and run:

```bash
PYTHONPATH=./src python -m unittest tests.test_atr_calculator
```

This command ensures that the `src` directory is included in the Python path so that the test runner can find the `atr_calculator` module.

## Formulas Used

- **For Ideal ATR-Based Position:**
    - Stop Loss Distance (ATR-based) = `ATR * Multiplier`
    - Position Size (Units) = `Target Risk Amount / Stop Loss Distance (ATR-based)`
- **For Existing Position Analysis (to meet Target Risk Amount):**
    - Stop Loss Distance (Calculated) = `Target Risk Amount / Existing Units`
- **General:**
    - Stop Loss Percentage = (`Relevant Stop Loss Distance / Reference Entry Price`) * 100
    - Stop Loss Price (Long) = `Reference Entry Price - Relevant Stop Loss Distance`
    - Stop Loss Price (Short) = `Reference Entry Price + Relevant Stop Loss Distance`

For more details on these calculations, please refer to [../docs/ATR_stop_loss_and_position_size_calculations.md](../docs/ATR_stop_loss_and_position_size_calculations.md). 