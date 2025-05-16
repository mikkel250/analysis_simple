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
analyzer_app = typer.Typer()
console = Console() # Global console for simple messages if needed outside formatters

@analyzer_app.callback()
def callback():
    """Market analyzer powered by the MarketAnalyzer class and modular output formatters."""
    console.print("📊 Market Analyzer: Comprehensive market analysis using multiple timeframes")

@analyzer_app.command()
def analyze(
    symbol: str = typer.Option("BTC-USDT", "--symbol", "-s", help="Symbol to analyze (e.g., BTC-USDT, ETH-USDT)"),
    timeframe: str = typer.Option(TimeframeOption.SHORT.value, "--timeframe", "-t", 
                                help=f"Trading timeframe ({', '.join([t.value for t in TimeframeOption])})"),
    output: str = typer.Option(OutputFormat.TEXT.value, "--output", "-o", 
                              help=f"Output format ({', '.join([f.value for f in OutputFormat])}). \
                                    Formats '{OutputFormat.TXT.value}', '{OutputFormat.JSF.value}', and '{OutputFormat.HTML.value}' save to a file."),
    # save_charts: bool = typer.Option(False, "--save-charts", "-c", 
    #                                 help="Save visualization charts to files (relevant for HTML output)"),
    # save_charts is now handled by html_generator and implicitly True for HTML output via formatters
    explain: bool = typer.Option(False, "--explain", "-e", 
                                help="Include educational explanations for indicators"),
    debug: bool = typer.Option(False, "--debug", "-d", 
                              help="Enable debug logging"),
    use_test_data: bool = typer.Option(False, "--test-data", "-td", 
                                     help="Use test data instead of fetching from API"),
):
    """
    Analyze a market symbol using the specified timeframe and output format.
    """
    try:
        if debug:
            logging.basicConfig(level=logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        console.print(f"⏳ Analyzing {symbol} for {timeframe} timeframe...")

        # Validate and convert TimeframeOption and OutputFormat from string input
        try:
            timeframe_enum = TimeframeOption(timeframe.lower())
        except ValueError:
            display_error(f"Invalid timeframe: {timeframe}. Valid options are: {', '.join([t.value for t in TimeframeOption])}")
            raise typer.Exit(code=1)

        try:
            output_format_enum = OutputFormat(output.lower())
        except ValueError:
            display_error(f"Invalid output format: {output}. Valid options are: {', '.join([f.value for f in OutputFormat])}")
            raise typer.Exit(code=1)

        # Initialize analyzer
        market_analyzer = MarketAnalyzer(symbol=symbol, timeframe=timeframe_enum.value, use_test_data=use_test_data)
        
        # --- Perform Analysis --- 
        # This structure assumes MarketAnalyzer methods might raise AnalyzerError for data fetching/processing issues.
        market_analyzer.fetch_data()
        market_analyzer.calculate_indicators()
        market_analyzer.run_analysis_pipeline() # This should populate all necessary fields in analysis_summary
        
        analysis_results = market_analyzer.get_analysis_summary() # Consolidated results
        analysis_results['visualizations'] = market_analyzer.get_visualizations() # Add visualizations if any
        # If get_summary() and present_cases() are still desired as separate items in results:
        # analysis_results['summary'] = market_analyzer.get_summary()
        # analysis_results['market_cases'] = market_analyzer.present_cases()
        # Ensure your MarketAnalyzer class structures its results in a way `display_market_analysis` expects.
        # The `display_market_analysis` in `formatters.py` expects a single `analysis_results` dict.

        # Determine if the output format implies saving to a file
        save_output_to_file = output_format_enum in [OutputFormat.TXT, OutputFormat.JSF, OutputFormat.HTML]

        display_info(f"Generating output in {output_format_enum.value} format...")
        if save_output_to_file:
            display_info("Output will be saved to a file.")

        # Call the main display function from the formatters module
        display_market_analysis(
            analysis_results=analysis_results,
            symbol=symbol,
            timeframe_str=timeframe_enum.value, # Pass the string value of the timeframe
            output_format_enum=output_format_enum,
            explain=explain,
            save_to_file=save_output_to_file
        )
        
        display_success(f"Analysis for {symbol} ({timeframe_enum.value}) completed.")

    except AnalyzerError as e:
        display_error(f"Analysis Error: {e}")
        logger.error(f"Analyzer error: {e}", exc_info=debug) # Log with traceback if debug is on
        raise typer.Exit(code=1)
    except ConnectionError as e: # Example for specific network errors from services
        display_error(f"Network Error: Could not connect to data provider. {e}")
        logger.error(f"Network error: {e}", exc_info=debug)
        raise typer.Exit(code=1)
    except ValueError as e: # Handles issues like invalid data for indicators if not caught by AnalyzerError
        display_error(f"Data Error: {e}")
        logger.error(f"Data error: {e}", exc_info=debug)
        raise typer.Exit(code=1)
    except Exception as e:
        display_error(f"An unexpected error occurred: {e}")
        logger.error(f"Unexpected error in market analysis: {e}", exc_info=True) # Always log traceback for unexpected
        raise typer.Exit(code=1)

if __name__ == "__main__":
    analyzer_app() 