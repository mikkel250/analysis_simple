import argparse
import sys
import pandas as pd
import logging

from src.atr_calculator import calculate_atr_trade_parameters
from src.services.data_fetcher import get_historical_data
from src.services.indicators import get_indicator

logging.disable(logging.CRITICAL)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate ATR-based position size and stop-loss parameters. "
            "Can also analyze existing positions."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\\n"
            "  # Long position, new calculation with direct ATR:\\n"
            "  python src/cli.py 100.0 long --atr 5.0\\n"
            "\\n"
            "  # Short position, calculate ATR from symbol:\\n"
            "  python src/cli.py 25000 short --atr-from-symbol BTC-USD \\n"
            "    --risk_amount 500 --multiplier 1.5\\n"
            "\\n"
            "  # Long position, calculate ATR from CSV with OHLC columns:\\n"
            "  python src/cli.py 0.5 long --atr-from-file data.csv \\n"
            "    --csv-ohlc Open High Low Close --csv-date-column Date\\n"
            "\\n"
            "  # Long position, use pre-calculated ATR from CSV:\\n"
            "  python src/cli.py 0.5 long --atr-from-file data_with_atr.csv \\n"
            "    --csv-atr-column MyATR --csv-date-column Date"
        )
    )

    parser.add_argument(
        "entry_price",
        type=float,
        help=(
            "Entry price for a new position, OR average entry price of an "
            "existing position (if --existing-units is used)."
        )
    )
    parser.add_argument(
        "position_type",
        type=str.lower,
        choices=['long', 'short'],
        help="Position type: 'long' or 'short'"
    )

    atr_source_group = parser.add_mutually_exclusive_group(required=True)
    atr_source_group.add_argument(
        "--atr",
        type=float,
        help="Directly provide the Average True Range (ATR) value."
    )
    atr_source_group.add_argument(
        "--atr-from-symbol",
        type=str,
        metavar='SYMBOL',
        help=(
            "Calculate ATR from historical data for the given symbol "
            "(e.g., BTC, ETH-USD)."
        )
    )
    atr_source_group.add_argument(
        "--atr-from-file",
        type=str,
        metavar='FILEPATH',
        help=(
            "Calculate ATR from a CSV file. "
            "Requires --csv-ohlc or --csv-atr-column."
        )
    )

    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help=(
            "Period for ATR calculation (default: 14). Used with "
            "--atr-from-symbol or with --atr-from-file and --csv-ohlc."
        )
    )
    parser.add_argument(
        "--atr-days",
        type=int,
        default=90,
        help=(
            "Number of days of historical data to fetch for ATR calculation "
            "(default: 90). Used with --atr-from-symbol."
        )
    )
    parser.add_argument(
        "--atr-vs-currency",
        type=str,
        default="usd",
        help=(
            "The currency to get historical data against (default: usd). "
            "Used with --atr-from-symbol."
        )
    )

    csv_options_group = parser.add_argument_group(
        title="CSV ATR Options (used with --atr-from-file)"
    )
    csv_options_group.add_argument(
        "--csv-ohlc",
        nargs=4,
        metavar=('OPEN_COL', 'HIGH_COL', 'LOW_COL', 'CLOSE_COL'),
        help=(
            "Names of Open, High, Low, Close columns in CSV for "
            "ATR calculation from OHLC."
        )
    )
    csv_options_group.add_argument(
        "--csv-atr-column",
        type=str,
        metavar='COLUMN_NAME',
        help="Name of a pre-calculated ATR column in the CSV file."
    )
    csv_options_group.add_argument(
        "--csv-date-column",
        type=str,
        metavar='DATE_COLUMN_NAME',
        help=(
            "Name of the date/timestamp column in the CSV "
            "(recommended for timeseries data)."
        )
    )

    parser.add_argument(
        "--risk_amount",
        type=float,
        default=100.0,
        help="Amount to risk on the trade (default: 100.0)"
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="ATR multiplier for stop-loss (default: 1.0)"
    )
    parser.add_argument(
        "--existing-units",
        type=float,
        default=None,
        help=(
            "Number of asset units currently held (optional). If provided, "
            "'entry_price' is treated as their average cost."
        )
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Validate conditional requirement for existing_units
    if args.existing_units is not None and args.existing_units < 0:
        parser.error("--existing-units cannot be negative.")

    # Determine ATR value
    atr_value = None

    if args.atr is not None:
        atr_value = args.atr
        logger.info(f"Using directly provided ATR: {atr_value}")
    elif args.atr_from_symbol:
        logger.info(
            f"Calculating ATR for symbol: {args.atr_from_symbol}, "
            f"days: {args.atr_days}, period: {args.atr_period}, "
            f"vs_currency: {args.atr_vs_currency}"
        )
        try:
            # Fetch historical data
            # For daily ATR, timeframe='1d' is appropriate.
            # Limit ensures enough data for ATR period.
            # get_historical_data needs symbol, timeframe, limit, vs_currency
            # A 14-period ATR needs at least 14 periods of data.
            # To be safe, fetch more.
            # atr_days should be sufficient if it's e.g. 90 for a 14-period ATR.
            # The 'limit' in get_historical_data is number of candles.
            df = get_historical_data(
                symbol=args.atr_from_symbol,
                timeframe='1d',  # Daily data for robust ATR
                limit=args.atr_days,  # Number of days
                vs_currency=args.atr_vs_currency
            )
            if df.empty or len(df) < args.atr_period:
                parser.error(
                    f"Not enough historical data fetched for {args.atr_from_symbol} "
                    f"to calculate ATR with period {args.atr_period}. "
                    f"Fetched {len(df)} days."
                )

            # Calculate ATR
            # Ensure columns are lowercase 'open', 'high', 'low', 'close'
            # as expected by get_indicator
            df.columns = [col.lower() for col in df.columns]
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                parser.error(
                    f"Fetched data for {args.atr_from_symbol} is missing one or "
                    f"more required OHLC columns: {required_cols}"
                )

            indicator_output = get_indicator(
                df, indicator='atr', params={'length': args.atr_period}
            )
            
            if indicator_output and 'values' in indicator_output and \
               indicator_output['values']:
                # Values from get_indicator is a dict {timestamp: value} or a Series
                atr_series = pd.Series(indicator_output['values'])
                if not atr_series.empty:
                    last_atr = atr_series.dropna().iloc[-1] \
                        if not atr_series.dropna().empty else None
                    if last_atr is not None:
                        atr_value = float(last_atr)
                        logger.info(
                            f"Calculated ATR from symbol {args.atr_from_symbol}: "
                            f"{atr_value:.8f}"
                        )
                    else:
                        parser.error(
                            f"Could not extract a valid ATR value for "
                            f"{args.atr_from_symbol} after calculation (all NaNs?)."
                        )
                else:
                    parser.error(
                        f"ATR calculation for {args.atr_from_symbol} "
                        "returned no values."
                    )
            else:
                parser.error(
                    f"Failed to calculate ATR for symbol {args.atr_from_symbol}. "
                    f"Output: {indicator_output}"
                )
        except Exception as e:
            parser.error(
                f"Error calculating ATR from symbol {args.atr_from_symbol}: {e}"
            )
    
    elif args.atr_from_file:
        logger.info(f"Calculating ATR from file: {args.atr_from_file}")
        if not (args.csv_ohlc or args.csv_atr_column):
            parser.error(
                "--atr-from-file requires either --csv-ohlc or "
                "--csv-atr-column to be specified."
            )
        if args.csv_ohlc and args.csv_atr_column:
            parser.error(
                "Cannot use both --csv-ohlc and --csv-atr-column "
                "with --atr-from-file."
            )

        try:
            read_params = {}
            if args.csv_date_column:
                read_params['parse_dates'] = [args.csv_date_column]
                read_params['index_col'] = args.csv_date_column
            
            csv_df = pd.read_csv(args.atr_from_file, **read_params)
            if csv_df.empty:
                parser.error(
                    f"CSV file {args.atr_from_file} is empty or could not be read."
                )

            if args.csv_atr_column:
                if args.csv_atr_column not in csv_df.columns:
                    parser.error(
                        f"ATR column '{args.csv_atr_column}' not found "
                        f"in {args.atr_from_file}."
                    )
                atr_series_from_csv = csv_df[args.csv_atr_column].dropna()
                if atr_series_from_csv.empty:
                    parser.error(
                        f"ATR column '{args.csv_atr_column}' in "
                        f"{args.atr_from_file} contains no valid data."
                    )
                atr_value = float(atr_series_from_csv.iloc[-1])
                logger.info(
                    f"Using ATR from CSV column '{args.csv_atr_column}': "
                    f"{atr_value:.8f}"
                )
            
            elif args.csv_ohlc:
                ohlc_map = {
                    'open': args.csv_ohlc[0],
                    'high': args.csv_ohlc[1],
                    'low': args.csv_ohlc[2],
                    'close': args.csv_ohlc[3],
                }
                # Check if all specified OHLC columns exist in the CSV
                missing_cols = [
                    ohlc_map[std_col] for std_col, user_col in ohlc_map.items()
                    if user_col not in csv_df.columns
                ]
                if missing_cols:
                    parser.error(
                        f"Missing OHLC columns in {args.atr_from_file}: "
                        f"{', '.join(missing_cols)}. Expected based on --csv-ohlc."
                    )

                # Create a new DataFrame with standardized column names
                # for indicator calculation
                calc_df = pd.DataFrame()
                calc_df['open'] = pd.to_numeric(
                    csv_df[ohlc_map['open']], errors='coerce'
                )
                calc_df['high'] = pd.to_numeric(
                    csv_df[ohlc_map['high']], errors='coerce'
                )
                calc_df['low'] = pd.to_numeric(
                    csv_df[ohlc_map['low']], errors='coerce'
                )
                calc_df['close'] = pd.to_numeric(
                    csv_df[ohlc_map['close']], errors='coerce'
                )
                
                # Drop rows with NaNs that might have been
                # introduced by to_numeric
                calc_df.dropna(
                    subset=['open', 'high', 'low', 'close'], inplace=True
                )

                if len(calc_df) < args.atr_period:
                    parser.error(
                        f"Not enough valid OHLC data in {args.atr_from_file} "
                        f"(after parsing & NaN removal) to calculate ATR with "
                        f"period {args.atr_period}. Found {len(calc_df)} rows."
                    )

                indicator_output = get_indicator(
                    calc_df, indicator='atr', params={'length': args.atr_period}
                )
                if indicator_output and 'values' in indicator_output and \
                   indicator_output['values']:
                    atr_series_csv = pd.Series(indicator_output['values'])
                    if not atr_series_csv.empty:
                        last_atr_csv = atr_series_csv.dropna().iloc[-1] \
                            if not atr_series_csv.dropna().empty else None
                        if last_atr_csv is not None:
                            atr_value = float(last_atr_csv)
                            logger.info(
                                f"Calculated ATR from CSV file {args.atr_from_file} "
                                f"using OHLC columns: {atr_value:.8f}"
                            )
                        else:
                            parser.error(
                                f"Could not extract a valid ATR value from CSV "
                                f"{args.atr_from_file} (all NaNs?)."
                            )
                    else:
                        parser.error(
                            f"ATR calculation from CSV {args.atr_from_file} "
                            "returned no values."
                        )
                else:
                    parser.error(
                        f"Failed to calculate ATR from CSV {args.atr_from_file}. "
                        f"Output: {indicator_output}"
                    )
        except FileNotFoundError:
            parser.error(f"CSV file not found: {args.atr_from_file}")
        except Exception as e:
            parser.error(
                f"Error processing CSV file {args.atr_from_file}: {e}"
            )
    else:
        # This case should ideally be prevented by the 'required=True'
        # on the atr_source_group
        parser.error(
            "ATR source not specified. Use --atr, --atr-from-symbol, "
            "or --atr-from-file."
        )

    if atr_value is None or atr_value <= 0:  # ATR must be positive
        parser.error(f"Determined ATR value ({atr_value}) must be positive.")

    try:
        price_precision = 8  # Default price precision
        print("\n--- ATR Position Sizing Tool ---")
        # Common parameters for display
        print("  Inputs:")
        print(f"    ATR:                                        {atr_value:.8f}")
        print(
            f"    Position Type:                              "
            f"{args.position_type.capitalize()}"
        )
        print(
            f"    Target Risk Amount:                         "
            f"${args.risk_amount:.2f}"
        )
        print(
            f"    ATR Multiplier:                             "
            f"{args.multiplier:.2f}x"
        )

        if args.existing_units is not None and args.existing_units > 0:
            # Scenario: Analyzing an EXISTING position
            # Main entry_price arg is used as avg cost of existing position
            avg_entry_price_existing = args.entry_price
            print(
                f"    Existing Units:                             "
                f"{args.existing_units:.8f}"
            )
            print(
                f"    Average Entry Price (of existing units):    "
                f"${avg_entry_price_existing:.8f}"
            )
            print("  -----------------------------------------------------")

            # Part 1: Analyze EXISTING position based on fixed risk_amount
            print(
                "\n  Part 1: Analysis of EXISTING Position "
                "(to meet Target Risk Amount)"
            )
            # Should not happen due to outer if, but defensive
            if args.existing_units <= 0:
                print(
                    "    Cannot calculate stop-loss for zero or "
                    "negative existing units."
                )
                sl_distance_for_existing = 0
                sl_price_for_existing = avg_entry_price_existing
                sl_percentage_for_existing = 0
            else:
                sl_distance_for_existing = (
                    args.risk_amount / args.existing_units
                )
            
            if args.position_type == 'long':
                sl_price_for_existing = (
                    avg_entry_price_existing - sl_distance_for_existing
                )
            else:  # short
                sl_price_for_existing = (
                    avg_entry_price_existing + sl_distance_for_existing
                )
            
            if avg_entry_price_existing > 0:
                sl_percentage_for_existing = (
                    (sl_distance_for_existing / avg_entry_price_existing) * 100
                )
            else:
                # Avoid division by zero, though entry price should be >0
                sl_percentage_for_existing = float('inf')
            
            existing_pos_value = avg_entry_price_existing * args.existing_units

            print(
                f"    Target Risk:                              "
                f"${args.risk_amount:.2f}"
            )
            print(
                f"    Calculated SL Distance per Unit:          "
                f"${sl_distance_for_existing:.8f}"
            )
            print(
                f"    Stop-Loss Price for Existing Position:    "
                f"${sl_price_for_existing:.8f}"
            )
            print(
                f"    Stop-Loss Percentage for Existing Position: "
                f"{sl_percentage_for_existing:.2f}%"
            )
            print(
                f"    Value of Existing Position:               "
                f"${existing_pos_value:.2f}"
            )
            print("  -----------------------------------------------------")

            # Part 2: Ideal ATR-based position (for reference/scaling)
            # Uses avg_entry_price_existing as the reference price
            # for this ideal calculation
            print(
                "\n  Part 2: Ideal ATR-Based Position (for Target Risk Amount, "
                "using existing avg entry as ref price)"
            )
            atr_sl_distance = atr_value * args.multiplier
            # We can reuse calculate_atr_trade_parameters, it will use
            # avg_entry_price_existing as its 'entry_price'
            ideal_atr_calc_results = calculate_atr_trade_parameters(
                entry_price=avg_entry_price_existing, 
                atr=atr_value,
                position_type=args.position_type,
                risk_amount=args.risk_amount,  # Target risk remains the same
                multiplier=args.multiplier
            )
            ideal_atr_units = ideal_atr_calc_results['position_size']
            # This is (atr_sl_distance / avg_entry_price_existing) * 100
            ideal_atr_sl_percentage = ideal_atr_calc_results['stop_loss_percentage']
            
            if args.position_type == 'long':
                ideal_atr_sl_price = (
                    avg_entry_price_existing - atr_sl_distance
                )
            else:  # short
                ideal_atr_sl_price = (
                    avg_entry_price_existing + atr_sl_distance
                )
            
            ideal_atr_pos_value = avg_entry_price_existing * ideal_atr_units
            units_to_adjust = ideal_atr_units - args.existing_units

            print(
                f"    ATR-based SL Distance per Unit:           "
                f"${atr_sl_distance:.8f}"
            )
            print(
                f"    Ideal Position Size (Units, ATR-based):   "
                f"{ideal_atr_units:.8f}"
            )
            print(
                f"    Ideal Position Value ($, ATR-based):      "
                f"${ideal_atr_pos_value:.2f}"
            )
            print(
                f"    Stop-Loss Price (ATR-based):              "
                f"${ideal_atr_sl_price:.8f}"
            )
            print(
                f"    Stop-Loss Percentage (ATR-based):         "
                f"{ideal_atr_sl_percentage:.2f}%"
            )
            if units_to_adjust > 0:
                print(
                    f"    Units to ADD to reach ATR-Ideal:        "
                    f"{units_to_adjust:.8f} (at current market or target price)"
                )
            elif units_to_adjust < 0:
                print(
                    f"    Units to REDUCE to reach ATR-Ideal:     "
                    f"{abs(units_to_adjust):.8f}"
                )
            else:
                print(
                    "    Existing position size matches ATR-Ideal "
                    "for target risk."
                )
            print("  -----------------------------------------------------")

        else:
            # Scenario: Calculating a NEW position (no existing units provided)
            new_pos_entry_price = args.entry_price
            print(
                f"    Entry Price for New Position:             "
                f"${new_pos_entry_price:.8f}"
            )
            print("  -----------------------------------------------------")
            print("\n  ATR-Based Calculation for NEW Position")

            results = calculate_atr_trade_parameters(
                entry_price=new_pos_entry_price,
                atr=atr_value,
                position_type=args.position_type,
                risk_amount=args.risk_amount,
                multiplier=args.multiplier
            )
            position_size_units = results['position_size']
            sl_percentage = results['stop_loss_percentage']
            dollar_value_of_position = (
                new_pos_entry_price * position_size_units
            )
            atr_sl_distance = atr_value * args.multiplier
            
            if args.position_type == 'long':
                sl_price = new_pos_entry_price - atr_sl_distance
            else:  # short
                sl_price = new_pos_entry_price + atr_sl_distance

            print(
                f"    Position Size ($):                        "
                f"${dollar_value_of_position:.2f}"
            )
            print(
                f"    Position Size (Asset Units):            "
                f"{position_size_units:.8f}"
            )
            print(
                f"    Stop-Loss Distance (ATR-based):         "
                f"${atr_sl_distance:.8f}"
            )
            print(f"    Stop-Loss Price:                        ${sl_price:.8f}")
            print(
                f"    Stop-Loss Percentage:                   {sl_percentage:.2f}%"
            )
            print("  -----------------------------------------------------")

        print("\n--- End ---")

        logger.info(f"Position Size: {results['position_size']:.8f} units")
        logger.info(
            f"Stop-Loss Price: {results['stop_loss_price']:.{price_precision}f}"
        )
        logger.info(
            f"Stop-Loss Percentage: {results['stop_loss_percentage']:.2f}%"
        )
        logger.info(
            f"Risk/Reward Ratio: {results['risk_reward_ratio']:.2f}:1 "
            "(if target is provided)"
        )
        if 'account_risk_percentage' in results:
            logger.info(
                f"Account Risk: {results['account_risk_percentage']:.2f}% "
                f"of account value {results['account_value']}"  # type: ignore
            )

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 