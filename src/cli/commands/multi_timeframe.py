"""
Multi-Timeframe Analysis CLI Command Handler

Handles the 'mta' (Multi-Timeframe Analysis) command.
"""

import asyncio
import typer
from rich.console import Console
from rich.table import Table
from rich.json import JSON
import logging
import json # For direct JSON output if chosen

from src.services.multi_timeframe_service import TimeframeAnalyzer, DEFAULT_TIMEFRAMES
from src.services.data_fetcher import CCXT_TIMEFRAMES
from src.cli.commands.analyzer_modules.formatters import OutputFormat, display_error, display_info, display_success, display_warning # Reusing formatters

logger = logging.getLogger(__name__)

mta_app = typer.Typer(
    name="mta",
    help="Perform Multi-Timeframe Analysis (MTA) for a given symbol.",
    add_completion=False,
    no_args_is_help=True
)

console = Console()

# Allowed timeframes for CLI input - from CCXT_TIMEFRAMES
ALLOWED_MTA_TIMEFRAMES = list(CCXT_TIMEFRAMES.keys())

# Helper to parse indicators input
def parse_indicators_input(indicators_str: Optional[str]) -> List[Dict[str, Any]]:
    """
    Parses the comma-separated indicator string into a list of dicts.
    Format: "sma:length=20,rsi:length=14,macd:fast=12;slow=26;signal=9"
    """
    if not indicators_str:
        return []
    
    parsed_indicators = []
    indicator_definitions = indicators_str.split(',')
    
    for definition in indicator_definitions:
        parts = definition.strip().split(':')
        name = parts[0].lower()
        params = {}
        if len(parts) > 1 and parts[1]:
            param_pairs = parts[1].split(';')
            for pair in param_pairs:
                key_value = pair.split('=')
                if len(key_value) == 2:
                    try:
                        # Attempt to convert to int or float if possible
                        val = key_value[1]
                        if val.isdigit():
                            params[key_value[0]] = int(val)
                        elif val.replace('.', '', 1).isdigit(): # Check for float
                            params[key_value[0]] = float(val)
                        else:
                            params[key_value[0]] = val # Keep as string if not clearly numeric
                    except ValueError:
                        params[key_value[0]] = key_value[1] # Keep as string on error
                else:
                    logger.warning(f"Could not parse param pair: {pair} for indicator {name}")
        parsed_indicators.append({"name": name, "params": params})
    logger.debug(f"Parsed indicator configurations: {parsed_indicators}")
    return parsed_indicators

@mta_app.command(
    name="analyze",
    help="Run multi-timeframe analysis for a symbol with specified indicators."
)
def analyze_mta(
    symbol: str = typer.Argument(..., help="The trading symbol to analyze (e.g., BTC/USDT)."),
    timeframes: Optional[str] = typer.Option(
        None, 
        "--timeframes", "-tf",
        help=f"Comma-separated list of timeframes (e.g., '1h,4h,1d'). Defaults to {','.join(DEFAULT_TIMEFRAMES)}. Supported: {', '.join(ALLOWED_MTA_TIMEFRAMES)}"
    ),
    indicators: str = typer.Option(
        "sma:length=20;period=20,rsi:length=14,ema:length=50", # Default example indicators
        "--indicators", "-i",
        help=("Comma-separated list of indicators and their parameters. "
              "Format: name1:param1=val1;param2=val2,name2:param1=val1. "
              "Example: 'sma:length=20,rsi:length=14,macd:fast=12;slow=26;signal=9'")
    ),
    limit: int = typer.Option(100, "--limit", "-l", help="Number of data points (candles) to fetch per timeframe."),
    output: OutputFormat = typer.Option(
        OutputFormat.TEXT.value, 
        "--output", "-o", 
        help=f"Output format. Choices: {[of.value for of in OutputFormat if of != OutputFormat.HTML]}", # HTML might be too complex for direct CLI
        case_sensitive=False
    ),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable/disable caching for data and indicators."),
    exchange: str = typer.Option("okx", "--exchange", "-ex", help="Exchange to use for fetching data.")
):
    """
    Performs Multi-Timeframe Analysis (MTA) for a given cryptocurrency symbol.
    Fetches data for multiple timeframes, calculates specified technical indicators,
    analyzes signal confluence, and presents a summary report.
    """
    
    console.print(f"[bold cyan]🚀 Starting Multi-Timeframe Analysis for {symbol}...[/bold cyan]")

    # Parse timeframes
    if timeframes:
        selected_timeframes = [tf.strip().lower() for tf in timeframes.split(',')]
        invalid_tfs = [tf for tf in selected_timeframes if tf not in ALLOWED_MTA_TIMEFRAMES]
        if invalid_tfs:
            display_error(f"Invalid timeframes specified: {', '.join(invalid_tfs)}. Allowed: {', '.join(ALLOWED_MTA_TIMEFRAMES)}")
            raise typer.Exit(code=1)
    else:
        selected_timeframes = DEFAULT_TIMEFRAMES
    
    display_info(f"Symbol: {symbol}, Timeframes: {selected_timeframes}, Exchange: {exchange}")

    # Parse indicators
    indicator_configs = parse_indicators_input(indicators)
    if not indicator_configs:
        display_warning("No indicators specified or parsed. Analysis will be limited to data summary.")
        # Optionally, one could define a minimal default set if none are provided and parsed.

    display_info(f"Indicators to calculate: {json.dumps(indicator_configs)}")

    try:
        analyzer = TimeframeAnalyzer(symbol=symbol, timeframes=selected_timeframes, exchange_name=exchange)
        
        # Run analysis asynchronously
        # Need to use asyncio.run() if called from synchronous Typer command function
        analysis_report = asyncio.run(analyzer.run_full_analysis(
            indicator_configs=indicator_configs,
            data_limit=limit,
            use_cache=use_cache
        ))

        if "error" in analysis_report:
            display_error(f"MTA failed: {analysis_report['error']}")
            raise typer.Exit(code=1)

        # Output the report based on the chosen format
        if output == OutputFormat.JSON:
            console.print(JSON(json.dumps(analysis_report, default=str))) # Use rich.json for pretty printing
        elif output == OutputFormat.TEXT:
            _display_text_report(analysis_report, console)
        # Add other formats like JSF, TXT (file) if needed
        else:
             console.print(json.dumps(analysis_report, indent=2, default=str)) # Fallback to simple json dump for now

        display_success("Multi-Timeframe Analysis complete.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during MTA for {symbol}: {e}", exc_info=True)
        display_error(f"An unexpected error occurred: {e}")
        raise typer.Exit(code=1)

def _display_text_report(report: Dict[str, Any], console: Console):
    """Helper function to display the report in a structured text format."""
    
    console.rule(f"[bold green]MTA Report for {report.get('symbol')} ({report.get('timestamp')})[/bold green]")
    console.print(f"[bold]Analyzed Timeframes:[/bold] {', '.join(report.get('analyzed_timeframes', []))}")

    # Data Summary
    console.print("\n[bold yellow]--- Data Summary ---[/bold yellow]")
    if "data_summary" in report:
        table_data = Table(title="Data per Timeframe")
        table_data.add_column("Timeframe", style="cyan")
        table_data.add_column("Candles")
        table_data.add_column("Start Date")
        table_data.add_column("End Date")
        table_data.add_column("Latest Close")
        for tf, data in report["data_summary"].items():
            table_data.add_row(
                tf,
                str(data.get("candle_count", "N/A")),
                str(data.get("start_date", "N/A")),
                str(data.get("end_date", "N/A")),
                f"{data.get('latest_close', 'N/A'):.2f}" if isinstance(data.get('latest_close'), (int, float)) else "N/A"
            )
        console.print(table_data)
    
    # Indicator Summary (simplified for text output)
    console.print("\n[bold yellow]--- Indicator Summary (Last Values) ---[/bold yellow]")
    # This part needs careful handling of indicator_summary structure
    # Assuming indicator_summary format from TimeframeAnalyzer
    if "indicator_summary" in report:
        for tf, indicators in report["indicator_summary"].items():
            console.print(f"\n[bold magenta]Timeframe: {tf}[/bold magenta]")
            if not indicators:
                console.print("  No indicators calculated or available for this timeframe.")
                continue
            
            indicator_table = Table(show_header=True, header_style="bold blue")
            indicator_table.add_column("Indicator")
            indicator_table.add_column("Last Value(s)") # Could be multiple for MACD etc.
            indicator_table.add_column("Signal") # General signal if available

            for ind_key, ind_data in indicators.items():
                if ind_data.get('error'):
                    indicator_table.add_row(ind_key, f"[red]Error: {ind_data['error']}[/red]", "N/A")
                    continue

                last_values_str = "N/A"
                # Extracting last value(s) can be complex depending on `get_indicator` output structure
                # This is a simplified example
                vals = ind_data.get('values')
                if isinstance(vals, dict):
                    # Handle multi-series like MACD {macd: {}, signal: {}, hist: {}}
                    # Or single series where values is a dict of {timestamp: value}
                    display_vals = {}
                    has_sub_series = False
                    for series_name, series_data in vals.items():
                        if isinstance(series_data, dict) and series_data: # timestamped values
                            has_sub_series = True
                            last_ts = sorted(series_data.keys())[-1]
                            last_val = series_data[last_ts]
                            display_vals[series_name] = f"{last_val:.4f}" if isinstance(last_val, (int, float)) else str(last_val)
                    if has_sub_series:
                        last_values_str = ", ".join([f"{k}: {v}" for k,v in display_vals.items()])
                    elif vals: # single series dict {timestamp: value}
                         last_ts = sorted(vals.keys())[-1]
                         last_val = vals[last_ts]
                         last_values_str = f"{last_val:.4f}" if isinstance(last_val, (int, float)) else str(last_val)
                
                signal_str = "N/A"
                sig_data = ind_data.get('signals')
                if isinstance(sig_data, dict):
                    signal_str = sig_data.get('general_signal', str(sig_data)) # Show general or dump dict
                elif isinstance(sig_data, str):
                    signal_str = sig_data
                elif isinstance(sig_data, list) and sig_data: # if list of signals, maybe take last?
                    signal_str = str(sig_data[-1])
                    
                indicator_table.add_row(ind_key, last_values_str, signal_str)
            console.print(indicator_table)

    # Confluence Analysis
    console.print("\n[bold yellow]--- Confluence Analysis ---[/bold yellow]")
    if "confluence_analysis" in report:
        confluence = report["confluence_analysis"]
        score = confluence.get('overall_sentiment_score', 0)
        score_color = "green" if score > 0 else ("red" if score < 0 else "yellow")
        console.print(f"  [bold]Overall Sentiment Score:[/bold] [{score_color}]{score:.2f}[/{score_color}]")
        
        if "details_per_indicator" in confluence:
            confluence_table = Table(title="Signal Confluence Details")
            confluence_table.add_column("Indicator", style="cyan")
            confluence_table.add_column("Aligned Signal", style="bold")
            confluence_table.add_column("Aligned Timeframes")
            # confluence_table.add_column("Signals by Timeframe") # This can be verbose
            
            for ind_key, details in confluence["details_per_indicator"].items():
                aligned_sig = details.get("aligned_signal", "N/A")
                sig_color = "green" if "bull" in aligned_sig.lower() or "buy" in aligned_sig.lower() else ("red" if "bear" in aligned_sig.lower() or "sell" in aligned_sig.lower() else "white")
                colored_aligned_sig = f"[{sig_color}]{aligned_sig}[/{sig_color}]"
                
                confluence_table.add_row(
                    ind_key, 
                    colored_aligned_sig,
                    ", ".join(details.get("aligned_timeframes", []))
                    # json.dumps(details.get("signals_by_timeframe", {}))
                )
            console.print(confluence_table)

    # Educational Notes
    console.print("\n[bold yellow]--- Educational Notes ---[/bold yellow]")
    if "educational_notes" in report:
        for key, text_or_dict in report["educational_notes"].items():
            if isinstance(text_or_dict, dict): # If notes become more structured
                console.print(f"  [bold magenta]{key.replace('_', ' ').title()}:[/bold magenta]")
                for sub_key, sub_text in text_or_dict.items():
                     console.print(f"    [bold]{sub_key.replace('_', ' ').title()}:[/bold] {sub_text}")
            else:
                console.print(f"  [bold magenta]{key.replace('_', ' ').title()}:[/bold magenta] {text_or_dict}")
    console.rule("[bold green]End of MTA Report[/bold green]")

if __name__ == "__main__":
    # For direct testing of this CLI module
    # You would typically run this via the main.py entry point of your application
    # Example: python -m src.cli.commands.multi_timeframe analyze BTC/USDT -tf "1h,4h" -i "sma:length=10"
    logging.basicConfig(level=logging.INFO)
    typer.run(mta_app) # This is incorrect, should be mta_app()
    # Correct way for Typer if __name__ == "__main__" for a module:
    # mta_app() # if you want to make this file runnable directly and parse args
    # However, Typer is usually added to a main app. 
    # For simple test, can do: 
    # import sys
    # cli_args = ["analyze", "BTC/USDT", "--timeframes", "1h,4h", "--indicators", "sma:length=10,rsi:length=7", "--output", "text"]
    # from typer.testing import CliRunner
    # runner = CliRunner()
    # result = runner.invoke(mta_app, cli_args)
    # print(result.stdout)
    # if result.exit_code != 0:
    #     print(f"Error: {result.exception}")
    pass # Typer apps are usually run by a main CLI orchestrator 