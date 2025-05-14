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
python src/cli.py <entry_price_or_avg_cost> <atr> <position_type> [--risk_amount RISK] [--multiplier MULT] [--existing-units UNITS]
```

### Arguments:

- `entry_price_or_avg_cost`: (Required, float)
    - If `--existing-units` is NOT used: This is the intended entry price for a new position.
    - If `--existing-units` IS used: This is the average entry price (cost basis) of your existing units.
- `atr`: (Required, float) The Average True Range (ATR) value for the asset (e.g., `0.0058`).
- `position_type`: (Required, string) The type of position: `long` or `short`.

### Options:

- `--risk_amount <amount>`: (Optional, float) The total dollar amount you are willing to risk on the trade. Default: `100.0`.
- `--multiplier <value>`: (Optional, float) The ATR multiplier used for the ATR-based stop-loss distance calculation. Default: `1.0` (meaning 1x ATR).
- `--existing-units <units>`: (Optional, float) The number of asset units you currently hold. If provided, the tool will analyze this existing position.
- `-h`, `--help`: Show the help message and exit.

### Examples:

**1. Calculate Parameters for a NEW Position:**

Calculate for a new long position with an entry price of $0.10379, ATR of $0.0058, risking $100 with a 1x ATR multiplier.

```bash
python src/cli.py 0.10379 0.0058 long --risk_amount 100
```

Expected Output:
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

**2. Analyze an EXISTING Position and Get Scaling Advice:**

You hold 8630 units of an asset with an average entry price of $0.10379. The current ATR is $0.0058. You want to risk a total of $100 on this position and are using a 1x ATR multiplier for reference.

```bash
python src/cli.py 0.10379 0.0058 long --risk_amount 100 --existing-units 8630
```

Expected Output:
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