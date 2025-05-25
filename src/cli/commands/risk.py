"""
Risk Management Command Handler

Handles CLI commands for accessing risk management tools.
"""

import logging
from typing import List, Optional

import typer
import pandas as pd
from tabulate import tabulate

from src.services.risk_management import RiskManagementService
from src.cli.display import (
    display_error, display_success, display_info, display_spinner
)

logger = logging.getLogger(__name__)
risk_app = typer.Typer()

risk_service = RiskManagementService(cache_enabled=True)

@risk_app.command("var")
def value_at_risk(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTCUSDT)"),
    timeframe: str = typer.Option(
        "1d", "--timeframe", "-t", help="Data timeframe (e.g., 1d, 4h)"
    ),
    days: int = typer.Option(
        252, "--days", "-d", help="Number of historical days for calculation"
    ),
    confidence: float = typer.Option(
        0.95, "--confidence", "-c", help="Confidence level for VaR (0.0 to 1.0)"
    ),
    method: str = typer.Option(
        "historical",
        "--method",
        "-m",
        help="VaR calculation method: 'historical' or 'parametric'"
    )
):
    """Calculate Value at Risk (VaR) for a symbol."""
    spinner_text = (
        f"Calculating {method.capitalize()} VaR for {symbol} ({timeframe}) "
        f"with {confidence*100:.0f}% confidence..."
    )
    display_info(spinner_text)
    with display_spinner(text=spinner_text):
        result = risk_service.calculate_var(
            symbol, timeframe, days, confidence, method
        )

    if result:
        table_data = [
            ["Symbol", result['symbol']],
            ["Timeframe", result['timeframe']],
            ["Method", result['method'].capitalize()],
            ["Confidence Level", f"{result['confidence_level']*100:.0f}%"],
            ["Historical Period", f"{result['period_days']} days"],
            ["VaR", f"{result['var_value']:.2f}%"]
        ]
        display_success(f"{method.capitalize()} VaR Calculation Complete")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))
        display_info("\nEducational Note:")
        print(result['educational_note'])
    else:
        display_error(f"Failed to calculate VaR for {symbol}.")

@risk_app.command("correlation")
def correlation_matrix(
    symbols: List[str] = typer.Argument(
        ..., help="List of symbols (e.g., BTCUSDT ETHUSDT SOLUSDT)"
    ),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Data timeframe"),
    days: int = typer.Option(90, "--days", "-d", help="Number of historical days")
):
    """Calculate correlation matrix for a list of symbols."""
    if len(symbols) < 2:
        display_error(
            "Please provide at least two symbols for correlation analysis."
        )
        raise typer.Exit(code=1)

    spinner_text = (
        f"Calculating correlation matrix for {', '.join(symbols)} ({timeframe})..."
    )
    display_info(spinner_text)
    with display_spinner(text=spinner_text):
        result = risk_service.correlation_analysis(symbols, timeframe, days)

    if result:
        df_corr = pd.DataFrame(result['correlation_matrix'])
        display_success("Correlation Analysis Complete")
        print(
            f"Period: {result['period_days']} days, "
            f"Timeframe: {result['timeframe']}"
        )
        print(
            tabulate(
                df_corr, headers='keys', tablefmt='pretty',
                showindex=True, floatfmt=".4f"
            )
        )
        display_info("\nEducational Note:")
        print(result['educational_note'])
    else:
        display_error(
            f"Failed to calculate correlation for {', '.join(symbols)}."
        )

@risk_app.command("size")
def position_size(
    account_balance: float = typer.Option(
        ..., "--balance", "-ab", help="Total account balance"
    ),
    risk_percent: float = typer.Option(
        ..., "--risk", "-rp",
        help="Percentage of account to risk (e.g., 1 for 1%)"
    ),
    stop_loss_percent: Optional[float] = typer.Option(
        None, "--sl-percent", "-slp",
        help="Stop-loss as percentage of asset price"
    ),
    entry_price: Optional[float] = typer.Option(
        None, "--entry", "-e", help="Entry price of the asset"
    ),
    stop_loss_price: Optional[float] = typer.Option(
        None, "--sl-price", "-sl", help="Specific stop-loss price"
    ),
    asset_price: Optional[float] = typer.Option(
        None, "--asset-price", "-ap",
        help="Current asset price (used with --sl-percent)"
    )
):
    """Calculate suggested position size."""
    if not ((entry_price and stop_loss_price) or
            (asset_price and stop_loss_percent)):
        display_error(
            "Either provide (--entry and --sl-price) OR "
            "(--asset-price and --sl-percent)."
        )
        raise typer.Exit(code=1)

    display_info("Calculating position size...")
    result = risk_service.position_sizing(
        account_balance, risk_percent, stop_loss_percent,
        entry_price, stop_loss_price, asset_price
    )

    if result:
        table_data = [
            ["Account Balance", f"${result['account_balance']:,.2f}"],
            [
                "Risk Per Trade",
                f"{result['risk_per_trade_percent']:.2f}% "
                f"(${result['amount_to_risk']:,.2f})"
            ],
        ]
        if result.get('entry_price_input') is not None:
            table_data.append([
                "Entry Price Input",
                f"${result['entry_price_input']:,.2f}"
            ])
        if result.get('stop_loss_price_input') is not None:
            table_data.append([
                "Stop-Loss Price Input",
                f"${result['stop_loss_price_input']:,.2f}"
            ])
        if result.get('asset_price_input') is not None:
            table_data.append([
                "Asset Price Input",
                f"${result['asset_price_input']:,.2f}"
            ])
        if result.get('stop_loss_percent_input') is not None:
            table_data.append([
                "Stop-Loss Percent Input",
                f"{result['stop_loss_percent_input']:.2f}%"
            ])

        table_data.extend([
            [
                "Calculated Stop-Loss",
                f"{result['calculated_stop_loss_percent']:.2f}%"
            ],
            ["Position Size (Units)", f"{result['position_size_units']:.8f}"],
            ["Position Size (Value)", f"${result['position_size_value']:,.2f}"]
        ])

        display_success("Position Sizing Calculation Complete")
        print(
            tabulate(table_data, headers=["Parameter", "Value"], tablefmt="pretty")
        )
        display_info("\nEducational Note:")
        print(result['educational_note'])
    else:
        display_error("Failed to calculate position size. Check inputs.")

@risk_app.command("volatility")
def asset_volatility(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Data timeframe"),
    days: int = typer.Option(
        30, "--days", "-d", help="Number of historical days"
    ),
    window: int = typer.Option(
        20, "--window", "-w", help="Rolling window for volatility"
    )
):
    """Calculate asset volatility."""
    spinner_text = f"Calculating volatility for {symbol} ({timeframe})..."
    display_info(spinner_text)
    with display_spinner(text=spinner_text):
        result = risk_service.volatility_analysis(
            symbol, timeframe, days, window
        )

    if result:
        vol_percent = result['current_annualized_volatility_percent']
        table_data = [
            ["Symbol", result['symbol']],
            ["Timeframe", result['timeframe']],
            ["Historical Period", f"{result['period_days']} days"],
            ["Calculation Window", f"{result['calculation_window']} periods"],
            [
                "Annualized Volatility",
                f"{vol_percent:.2f}%" if vol_percent is not None else "N/A"
            ]
        ]
        display_success("Volatility Analysis Complete")
        print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="pretty"))
        display_info("\nEducational Note:")
        print(result['educational_note'])
    else:
        display_error(f"Failed to calculate volatility for {symbol}.")

@risk_app.command("summary")
def risk_summary(
    symbol: str = typer.Argument(..., help="Trading symbol"),
    timeframe: str = typer.Option("1d", "--timeframe", "-t", help="Data timeframe"),
    account_balance: float = typer.Option(
        10000, "--balance", "-ab",
        help="Account balance for position sizing example"
    ),
    risk_percent: float = typer.Option(
        1, "--risk", "-rp",
        help="Risk per trade for position sizing example (e.g. 1 for 1%)"
    ),
    sl_percent: float = typer.Option(
        5, "--sl", "-slp",
        help="Stop-loss percent for position sizing example (e.g. 5 for 5%)"
    ),
    asset_price: Optional[float] = typer.Option(
        None, "--asset-price", "-ap",
        help="Current asset price (optional, will fetch if not provided)"
    )
):
    """Display a consolidated risk assessment summary for a symbol."""
    spinner_text = (
        f"Generating risk assessment summary for {symbol} ({timeframe})..."
    )
    display_info(spinner_text)
    with display_spinner(text=spinner_text):
        summary = risk_service.get_risk_assessment_summary(
            symbol=symbol,
            timeframe=timeframe,
            account_balance=account_balance,
            risk_per_trade_percent=risk_percent,
            stop_loss_percent=sl_percent,  # Renamed to match service arg
            asset_price=asset_price
        )

    if summary:
        display_success("Risk Assessment Summary Complete")
        print(
            f"--- Risk Assessment for {summary['symbol']} "
            f"({summary['timeframe']}) ---"
        )

        if summary['value_at_risk']:
            var = summary['value_at_risk']
            print(f"\nValue at Risk ({var['method']}):")
            print(
                f"  At {var['confidence_level']*100:.0f}% confidence, "
                f"max expected loss: {var['var_value']:.2f}% "
                f"(based on {var['period_days']} days)"
            )
        else:
            print("\nValue at Risk: Not available")

        if summary['volatility']:
            vol = summary['volatility']
            print("\nVolatility:")
            print(
                f"  Current annualized volatility: "
                f"{vol['current_annualized_volatility_percent']:.2f}% "
                f"(window: {vol['calculation_window']} periods, data: "
                f"{vol['period_days']} days)"
            )
        else:
            print("\nVolatility: Not available")

        if summary['position_sizing_example']:
            pos = summary['position_sizing_example']
            print(
                f"\nPosition Sizing Example (for ${pos['account_balance']:,.2f} "
                f"account, {pos['risk_per_trade_percent']}% risk):"
            )
            print(
                f"  Asset Price (if used): ${pos['asset_price_input']:,.2f}" if
                pos.get('asset_price_input') else
                "  Asset Price Input: Not provided or fetched for example"
            )
            print(f"  Stop-Loss: {pos['calculated_stop_loss_percent']:.2f}%")
            print(
                f"  Suggested Size: {pos['position_size_units']:.8f} units "
                f"(Value: ${pos['position_size_value']:,.2f})"
            )
        else:
            print(
                "\nPosition Sizing Example: Not available or could not be "
                "calculated (e.g., missing asset price)."
            )

        display_info("\nOverall Educational Summary:")
        print(summary['educational_summary'])
    else:
        display_error(
            f"Failed to generate risk assessment summary for {symbol}."
        )

if __name__ == "__main__":
    risk_app()  # For testing CLI commands directly 