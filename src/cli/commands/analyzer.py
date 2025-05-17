"""
Analyzer Command Handler

Handles the 'analyzer' command for generating market analysis.
This file now orchestrates analysis and uses modules from 'analyzer_modules' for output formatting.
"""

import logging
import typer
from rich.console import Console

# Core analysis class
from src.analysis.market_analyzer import MarketAnalyzer 

# New utility modules
from .analyzer_modules.common import AnalyzerError, TimeframeOption, OutputFormat
from .analyzer_modules.formatters import display_market_analysis
from .analyzer_modules.formatters import display_info, display_success, display_warning, display_error # Direct import for now

# Configure logging
logger = logging.getLogger(__name__)

# Create the command app
analyzer_app = typer.Typer(
    name="analyze-market",
    help="Perform market analysis for a given symbol, including technical indicators, news sentiment, and fundamental data.",
    add_completion=False,
    no_args_is_help=True
)
console = Console() # Global console for simple messages if needed outside formatters

# Callback to dynamically generate help text for timeframe and output format
@analyzer_app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    timeframe: TimeframeOption = typer.Option(
        TimeframeOption.MEDIUM.value, 
        "--timeframe", "-t", 
        help=f"Trading timeframe. Choices: {[tf.value for tf in TimeframeOption]}",
        case_sensitive=False
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.TEXT.value, 
        "--output", "-o", 
        help=f"Output format. Choices: {[of.value for of in OutputFormat]}",
        case_sensitive=False
    )
):
    if ctx.invoked_subcommand is None:
        # Store choices on context for the command to use if no subcommand is called (which is our case)
        # This is a bit of a workaround as Typer callbacks are usually for pre-command logic
        # and direct app invocation doesn't pass these context objects directly to the command.
        # However, our single command `analyze` will pick these up from its own signature.
        pass

@analyzer_app.command(
    name="analyze", # Explicitly naming the command, though it's default for single command apps
    help="Analyze a financial instrument (e.g., stock symbol like AAPL, crypto like BTC-USD)."
)
def analyze(
    symbol: str = typer.Argument(..., help="The trading symbol to analyze (e.g., AAPL, BTC-USD)."),
    timeframe: TimeframeOption = typer.Option(
        TimeframeOption.MEDIUM.value, 
        "--timeframe", "-t", 
        help="Trading timeframe.", # Help text is dynamic via callback, but static here is fine too
        case_sensitive=False
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.TEXT.value, 
        "--output", "-o", 
        help="Output format.", # Dynamic help from callback
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
    
    console = Console()
    # Use the enum objects directly
    timeframe_val = timeframe.value
    output_format_enum = output 

    display_info(f"Market analysis for {symbol} ({timeframe_val}) started.")

    try:
        # Initialize MarketAnalyzer
        # This is where you'd pass the API key if your MarketAnalyzer uses one
        analyzer = MarketAnalyzer(symbol=symbol, timeframe=timeframe_val)
        
        # Perform analysis
        analysis_results = analyzer.analyze() # This should return a dictionary
        
        # Determine if saving to file is implied by output format
        save_to_file = output_format_enum in [OutputFormat.TXT, OutputFormat.JSF, OutputFormat.HTML]

        # Display results using the new centralized formatter
        display_market_analysis(
            analysis_results=analysis_results,
            symbol=symbol,
            timeframe_str=timeframe_val,
            output_format_enum=output_format_enum,
            explain=explain,
            save_to_file=save_to_file
        )
        
        if save_to_file:
            # The display_market_analysis function already handles success messages for file saving.
            pass
        elif output_format_enum == OutputFormat.TEXT or output_format_enum == OutputFormat.JSON:
            # For direct console outputs without file saving, a generic success could be here if desired.
            # However, the output itself is the success indicator.
            pass 
            
    except AnalyzerError as ae:
        display_error(f"Analyzer error for {symbol}: {ae}")
        raise typer.Exit(code=1)
    except FileNotFoundError as fe:
        display_error(f"File operation error: {fe}") # More specific error for file issues
        raise typer.Exit(code=1)
    except ConnectionError as ce:
        display_error(f"Network connection error: {ce}")
        raise typer.Exit(code=1)
    except Exception as e:
        # Catch-all for other unexpected errors during analysis or display
        logger.error(f"Unexpected error during analysis for {symbol}: {e}", exc_info=True)
        display_error(f"An unexpected error occurred for {symbol}: {e}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    analyzer_app() 