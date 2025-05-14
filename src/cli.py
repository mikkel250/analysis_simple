import argparse
import sys
from atr_calculator import calculate_atr_trade_parameters  # type: ignore


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate ATR-based position size and stop-loss parameters. "
            "Can also analyze existing positions."
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
        help="Entry price for a new position, OR average entry price of an existing position (if --existing-units is used)."
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
        help="Number of asset units currently held (optional). If provided, 'entry_price' is treated as their average cost."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Validate conditional requirement for existing_units
    if args.existing_units is not None and args.existing_units < 0:
        parser.error("--existing-units cannot be negative.")

    try:
        print("\n--- ATR Position Sizing Tool ---")
        # Common parameters for display
        print(f"  Inputs:")
        print(f"    ATR:                                        {args.atr:.8f}")
        print(f"    Position Type:                              {args.position_type.capitalize()}")
        print(f"    Target Risk Amount:                         ${args.risk_amount:.2f}")
        print(f"    ATR Multiplier:                             {args.multiplier:.2f}x")

        if args.existing_units is not None and args.existing_units > 0:
            # Scenario: Analyzing an EXISTING position
            avg_entry_price_existing = args.entry_price # Main entry_price arg is used as avg cost of existing position
            print(f"    Existing Units:                             {args.existing_units:.8f}")
            print(f"    Average Entry Price (of existing units):    ${avg_entry_price_existing:.8f}")
            print("  -----------------------------------------------------")

            # Part 1: Analyze EXISTING position based on fixed risk_amount
            print("\n  Part 1: Analysis of EXISTING Position (to meet Target Risk Amount)")
            if args.existing_units <= 0: # Should not happen due to outer if, but defensive
                print("    Cannot calculate stop-loss for zero or negative existing units.")
                sl_distance_for_existing = 0
                sl_price_for_existing = avg_entry_price_existing
                sl_percentage_for_existing = 0
            else:
                sl_distance_for_existing = args.risk_amount / args.existing_units
            
            if args.position_type == 'long':
                sl_price_for_existing = avg_entry_price_existing - sl_distance_for_existing
            else: # short
                sl_price_for_existing = avg_entry_price_existing + sl_distance_for_existing
            
            if avg_entry_price_existing > 0:
                sl_percentage_for_existing = (sl_distance_for_existing / avg_entry_price_existing) * 100
            else:
                sl_percentage_for_existing = float('inf') # Avoid division by zero, though entry price should be >0
            
            existing_pos_value = avg_entry_price_existing * args.existing_units

            print(f"    Target Risk:                              ${args.risk_amount:.2f}")
            print(f"    Calculated SL Distance per Unit:          ${sl_distance_for_existing:.8f}")
            print(f"    Stop-Loss Price for Existing Position:    ${sl_price_for_existing:.8f}")
            print(f"    Stop-Loss Percentage for Existing Position: {sl_percentage_for_existing:.2f}%")
            print(f"    Value of Existing Position:               ${existing_pos_value:.2f}")
            print("  -----------------------------------------------------")

            # Part 2: Ideal ATR-based position (for reference/scaling)
            # Uses avg_entry_price_existing as the reference price for this ideal calculation
            print("\n  Part 2: Ideal ATR-Based Position (for Target Risk Amount, using existing avg entry as ref price)")
            atr_sl_distance = args.atr * args.multiplier
            # We can reuse calculate_atr_trade_parameters, it will use avg_entry_price_existing as its 'entry_price'
            ideal_atr_calc_results = calculate_atr_trade_parameters(
                entry_price=avg_entry_price_existing, 
                atr=args.atr,
                position_type=args.position_type,
                risk_amount=args.risk_amount, # Target risk remains the same
                multiplier=args.multiplier
            )
            ideal_atr_units = ideal_atr_calc_results['position_size']
            ideal_atr_sl_percentage = ideal_atr_calc_results['stop_loss_percentage'] # This is (atr_sl_distance / avg_entry_price_existing) * 100
            
            if args.position_type == 'long':
                ideal_atr_sl_price = avg_entry_price_existing - atr_sl_distance
            else: # short
                ideal_atr_sl_price = avg_entry_price_existing + atr_sl_distance
            
            ideal_atr_pos_value = avg_entry_price_existing * ideal_atr_units
            units_to_adjust = ideal_atr_units - args.existing_units

            print(f"    ATR-based SL Distance per Unit:           ${atr_sl_distance:.8f}")
            print(f"    Ideal Position Size (Units, ATR-based):   {ideal_atr_units:.8f}")
            print(f"    Ideal Position Value ($, ATR-based):      ${ideal_atr_pos_value:.2f}")
            print(f"    Stop-Loss Price (ATR-based):              ${ideal_atr_sl_price:.8f}")
            print(f"    Stop-Loss Percentage (ATR-based):         {ideal_atr_sl_percentage:.2f}%")
            if units_to_adjust > 0:
                print(f"    Units to ADD to reach ATR-Ideal:        {units_to_adjust:.8f} (at current market or target price)")
            elif units_to_adjust < 0:
                print(f"    Units to REDUCE to reach ATR-Ideal:     {abs(units_to_adjust):.8f}")
            else:
                print("    Existing position size matches ATR-Ideal for target risk.")
            print("  -----------------------------------------------------")

        else:
            # Scenario: Calculating a NEW position (no existing units provided)
            new_pos_entry_price = args.entry_price
            print(f"    Entry Price for New Position:             ${new_pos_entry_price:.8f}")
            print("  -----------------------------------------------------")
            print("\n  ATR-Based Calculation for NEW Position")

            results = calculate_atr_trade_parameters(
                entry_price=new_pos_entry_price,
                atr=args.atr,
                position_type=args.position_type,
                risk_amount=args.risk_amount,
                multiplier=args.multiplier
            )
            position_size_units = results['position_size']
            sl_percentage = results['stop_loss_percentage']
            dollar_value_of_position = new_pos_entry_price * position_size_units
            atr_sl_distance = args.atr * args.multiplier # Re-calc for clarity or use from results if available
            
            if args.position_type == 'long':
                sl_price = new_pos_entry_price - atr_sl_distance
            else: # short
                sl_price = new_pos_entry_price + atr_sl_distance

            print(f"    Position Size ($):                        ${dollar_value_of_position:.2f}")
            print(f"    Position Size (Asset Units):            {position_size_units:.8f}")
            print(f"    Stop-Loss Distance (ATR-based):         ${atr_sl_distance:.8f}")
            print(f"    Stop-Loss Price:                        ${sl_price:.8f}")
            print(f"    Stop-Loss Percentage:                   {sl_percentage:.2f}%")
            print("  -----------------------------------------------------")

        print("\n--- End ---")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 