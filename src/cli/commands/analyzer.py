"""
Analyzer Command Handler

Handles the 'analyzer' command for generating market analysis.
This file now orchestrates analysis and uses modules from 'analyzer_modules' for output formatting.
"""

import logging
logging.disable(logging.CRITICAL)
import typer
from rich.console import Console

# Core analysis class
from src.analysis.market_analyzer import MarketAnalyzer 

# New utility modules
# TimeframeOption removed, direct strings will be used
from .analyzer_modules.common import AnalyzerError, OutputFormat 
from .analyzer_modules.formatters import display_market_analysis
# display_info, display_success, display_warning, display_error are already in formatters
# but keeping direct import for now if used elsewhere here, can be cleaned later
from .analyzer_modules.formatters import display_info, display_success, display_warning, display_error 

# Configure logging
logger = logging.getLogger(__name__)

# Allowed timeframes for CLI input
ALLOWED_TIMEFRAMES = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w']
DEFAULT_TIMEFRAME = "1d"

# Create the command app
analyzer_app = typer.Typer(
    name="analyze-market",
    help="Perform market analysis for a given symbol, including technical indicators, news sentiment, and fundamental data.",
    add_completion=False,
    no_args_is_help=True
)
console = Console() # Global console for simple messages if needed outside formatters

# Callback to dynamically generate help text for output format
# Timeframe help is now static in the command itself
@analyzer_app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    # Timeframe option removed from callback, handled directly in command
    output: OutputFormat = typer.Option(
        OutputFormat.TEXT.value, 
        "--output", "-o", 
        help=f"Output format. Choices: {[of.value for of in OutputFormat]}",
        case_sensitive=False
    )
):
    if ctx.invoked_subcommand is None:
        pass

@analyzer_app.command(
    name="analyze", 
    help="Analyze a financial instrument (e.g., stock symbol like AAPL, crypto like BTC-USD)."
)
def analyze(
    symbol: str = typer.Argument(..., help="The trading symbol to analyze (e.g., AAPL, BTC-USD)."),
    timeframe: str = typer.Option(
        DEFAULT_TIMEFRAME, 
        "--timeframe", "-t", 
        help=f"Trading timeframe. Allowed values: {', '.join(ALLOWED_TIMEFRAMES)}."
        # case_sensitive=False removed, handled by lowercasing
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.TEXT.value, 
        "--output", "-o", 
        help="Output format.", # Dynamic help from callback or static here
        case_sensitive=False
    ),
    explain: bool = typer.Option(False, "--explain", "-e", help="Provide detailed explanations for analysis components."),
    api_key: str = typer.Option(None, "--api-key", help="API key for premium data services (optional).")
):
    """
    Analyze a financial instrument (e.g., stock symbol like AAPL, crypto like BTC-USD).
    
    This command fetches and processes market data, technical indicators, news sentiment (placeholder),
    and fundamental data (placeholder) to provide a comprehensive analysis.
    """
    
    console = Console() # Re-instantiate or use global, ensure consistency
    
    timeframe_val = timeframe.lower()
    if timeframe_val not in ALLOWED_TIMEFRAMES:
        display_error(
            f"Invalid timeframe '{timeframe}'. Allowed values are: {', '.join(ALLOWED_TIMEFRAMES)}."
        )
        raise typer.Exit(code=1)
        
    output_format_enum = output # output is already an enum instance from Typer

    display_info(f"Market analysis for {symbol} ({timeframe_val}) started.")

    try:
        analyzer = MarketAnalyzer(symbol=symbol, timeframe=timeframe_val) # Pass validated string
        
        analysis_results = analyzer.analyze()
        
        save_to_file = output_format_enum in [OutputFormat.TXT, OutputFormat.JSF, OutputFormat.HTML]

        display_market_analysis(
            analysis_results=analysis_results,
            symbol=symbol,
            timeframe_str=timeframe_val, # Pass the string timeframe
            output_format_enum=output_format_enum,
            explain=explain,
            save_to_file=save_to_file
        )
        
        if save_to_file:
            pass
        elif output_format_enum == OutputFormat.TEXT or output_format_enum == OutputFormat.JSON:
            pass 
            
    except AnalyzerError as ae:
        display_error(f"Analyzer error for {symbol}: {ae}")
        raise typer.Exit(code=1)
    except FileNotFoundError as fe:
        display_error(f"File operation error: {fe}")
        raise typer.Exit(code=1)
    except ConnectionError as ce:
        display_error(f"Network connection error: {ce}")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Unexpected error during analysis for {symbol}: {e}", exc_info=True)
        display_error(f"An unexpected error occurred for {symbol}: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    analyzer_app() 