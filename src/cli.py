import argparse
import sys
from atr_calculator import calculate_atr_trade_parameters  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate ATR-based position size and stop-loss percentage."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Long position, default risk/multiplier:\n"
            "  python src/cli.py 100.0 5.0 long\n"
            "\n"
            "  # Short position, specified risk/multiplier:\n"
            "  python src/cli.py 25000 300 short --risk_amount 500 "
            "--multiplier 1.5"
        )
    )

    parser.add_argument(
        "entry_price",
        type=float,
        help="Entry price of the asset (e.g., 100.0)"
    )
    parser.add_argument(
        "atr",
        type=float,
        help="Average True Range (ATR) value (e.g., 5.0)"
    )
    parser.add_argument(
        "position_type",
        type=str.lower,
        choices=['long', 'short'],
        help="Position type: 'long' or 'short'"
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
        help="Number of asset units currently held (optional)"
    )
    parser.add_argument(
        "--existing-avg-price",
        type=float,
        default=None,
        help="Average entry price of existing units (required if --existing-units is specified and > 0; optional)"
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Validate conditional requirement for existing_avg_price
    if args.existing_units is not None and args.existing_units > 0 and args.existing_avg_price is None:
        parser.error("--existing-avg-price is required when --existing-units is specified and greater than 0.")
    if args.existing_units is None and args.existing_avg_price is not None:
        parser.error("--existing-units must be specified if --existing-avg-price is provided.")
    if args.existing_units is not None and args.existing_units < 0:
        parser.error("--existing-units cannot be negative.")
    if args.existing_avg_price is not None and args.existing_avg_price <= 0:
        parser.error("--existing-avg-price must be positive.")

    try:
        # This core calculation is always useful for a baseline or target
        ideal_calc_results = calculate_atr_trade_parameters(
            entry_price=args.entry_price,  # entry_price arg is for new calculations/additions
            atr=args.atr,
            position_type=args.position_type,
            risk_amount=args.risk_amount,
            multiplier=args.multiplier
        )
        ideal_units = ideal_calc_results['position_size']
        # SL% from ideal_calc_results is based on args.entry_price for the new/ideal calculation
        ideal_sl_percentage_at_arg_entry_price = ideal_calc_results['stop_loss_percentage']

        stop_loss_distance = args.atr * args.multiplier

        print("\n--- ATR Trade Parameters ---")
        print(f"  Inputs:")
        print(f"    Entry Price for New Calculations/Additions: {args.entry_price}")
        print(f"    ATR:                                        {args.atr}")
        print(f"    Position Type:                              {args.position_type.capitalize()}")
        print(f"    Target Risk Amount (for total position):    {args.risk_amount}")
        print(f"    ATR Multiplier:                             {args.multiplier}")
        if args.existing_units is not None and args.existing_units > 0:
            print(f"    Existing Units:                             {args.existing_units}")
            print(f"    Existing Average Price:                     {args.existing_avg_price}")
        print("  ------------------------------------")

        if args.existing_units is not None and args.existing_units > 0 and args.existing_avg_price is not None:
            # Scenario 2: Existing position details provided
            print("\n--- Existing Position Analysis ---")
            current_sl_price = args.existing_avg_price - stop_loss_distance if args.position_type == 'long' else args.existing_avg_price + stop_loss_distance
            current_sl_percentage = (stop_loss_distance / args.existing_avg_price) * 100 if args.existing_avg_price > 0 else float('inf')
            current_value_at_risk = args.existing_units * stop_loss_distance
            current_total_value = args.existing_units * args.existing_avg_price

            print(f"  Current Position Value ($):               {current_total_value:.8f}")
            print(f"  Stop-Loss Price (for existing units):     {current_sl_price:.8f}")
            print(f"  Stop-Loss Percentage (for existing units):{current_sl_percentage:.2f}%")
            print(f"  Value at Risk (for existing units):       {current_value_at_risk:.8f}")
            print("  ------------------------------------")

            print("\n--- Target Position & Adjustment Recommendation ---")
            # Ideal calculations are based on args.entry_price and args.risk_amount
            ideal_position_value_at_arg_entry_price = args.entry_price * ideal_units
            ideal_sl_price_at_arg_entry_price = args.entry_price - stop_loss_distance if args.position_type == 'long' else args.entry_price + stop_loss_distance

            print(f"  Ideal Total Units (for target risk):      {ideal_units:.8f}")
            print(f"  Value of Ideal Total Units (at new entry price {args.entry_price}): {ideal_position_value_at_arg_entry_price:.8f}")
            print(f"  Stop-Loss Price (if new position at {args.entry_price}): {ideal_sl_price_at_arg_entry_price:.8f}")
            print(f"  Stop-Loss % (if new position at {args.entry_price}):   {ideal_sl_percentage_at_arg_entry_price:.2f}%")
            
            units_to_adjust = ideal_units - args.existing_units
            if units_to_adjust > 0:
                print(f"  Units to ADD (at price {args.entry_price}):           {units_to_adjust:.8f}")
                print("\n--- Projected Combined Position (if adding units) ---")
                new_total_units = args.existing_units + units_to_adjust # Should be ideal_units
                new_avg_entry_price = ((args.existing_avg_price * args.existing_units) + (args.entry_price * units_to_adjust)) / new_total_units if new_total_units > 0 else 0

                if new_avg_entry_price > 0 :
                    combined_sl_price = new_avg_entry_price - stop_loss_distance if args.position_type == 'long' else new_avg_entry_price + stop_loss_distance
                    combined_sl_percentage = (stop_loss_distance / new_avg_entry_price) * 100
                    combined_position_value = new_avg_entry_price * new_total_units
                    combined_value_at_risk = new_total_units * stop_loss_distance

                    print(f"  Projected Total Units:                    {new_total_units:.8f}")
                    print(f"  Projected Average Entry Price:            {new_avg_entry_price:.8f}")
                    print(f"  Projected Position Value ($):             {combined_position_value:.8f}")
                    print(f"  Projected Stop-Loss Price:                {combined_sl_price:.8f}")
                    print(f"  Projected Stop-Loss Percentage:           {combined_sl_percentage:.2f}%")
                    print(f"  Projected Value at Risk ($):              {combined_value_at_risk:.8f} (Target: {args.risk_amount})")
                else:
                    print("  Cannot project combined position with zero average entry price.")
                print("  ------------------------------------")

            elif units_to_adjust < 0:
                print(f"  Units to REDUCE:                          {abs(units_to_adjust):.8f}")
                print("  ------------------------------------")
            else:
                print("  Current position size matches ideal for target risk.")
                print("  ------------------------------------")
        else:
            # Scenario 1: No existing position details (or existing_units is 0)
            print("\n--- Target Position Calculation (New Position) ---")
            # Calculations based on args.entry_price
            position_value_dollars = args.entry_price * ideal_units
            stop_loss_price = args.entry_price - stop_loss_distance if args.position_type == 'long' else args.entry_price + stop_loss_distance

            print(f"  Position Size ($):                      {position_value_dollars:.8f}")
            print(f"  Position Size (Asset Units):            {ideal_units:.8f}")
            print(f"  Stop-Loss Price:                        {stop_loss_price:.8f}")
            print(f"  Stop-Loss Percentage:                   {ideal_sl_percentage_at_arg_entry_price:.2f}%")
            print("  ------------------------------------")

        print("\n--- End ---")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 