"""
Analyzer Command Handler

Handles the 'analyzer' command for generating market analysis.
This file now orchestrates analysis and uses modules from 'analyzer_modules' for output formatting.
"""

import logging
logging.disable(logging.CRITICAL)
import typer
from rich.console import Console
import asyncio
from src.services.risk_management import RiskManagementService
from src.services.sentiment_service import SentimentDataService
from src.services.adaptive_indicators import AdaptiveIndicatorService
from src.services.multi_timeframe_service import TimeframeAnalyzer, DEFAULT_TIMEFRAMES
import contextlib
from io import StringIO

# Core analysis class
from src.analysis.market_analyzer import MarketAnalyzer 

# New utility modules
# TimeframeOption removed, direct strings will be used
from src.cli.commands.analyzer_modules.common import AnalyzerError, OutputFormat 
from src.cli.commands.analyzer_modules.formatters import display_market_analysis
# display_info, display_success, display_warning, display_error are already in formatters
# but keeping direct import for now if used elsewhere here, can be cleaned later
from src.cli.commands.analyzer_modules.formatters import display_info, display_success, display_warning, display_error 

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
    ),
    output: OutputFormat = typer.Option(
        OutputFormat.TEXT.value, 
        "--output", "-o", 
        help="Output format.",
        case_sensitive=False
    ),
    explain: bool = typer.Option(False, "--explain", "-e", help="Provide detailed explanations for analysis components."),
    api_key: str = typer.Option(None, "--api-key", help="API key for premium data services (optional)."),
    account_balance: float = typer.Option(10000, "--account-balance", help="Account balance for risk management (default: 10000)"),
    risk_per_trade_percent: float = typer.Option(1.0, "--risk-per-trade", help="Risk per trade percent for risk management (default: 1.0)"),
    stop_loss_percent: float = typer.Option(5.0, "--stop-loss", help="Stop loss percent for risk management (default: 5.0)")
):
    """
    Analyze a financial instrument (e.g., stock symbol like AAPL, crypto like BTC-USD).
    This command fetches and processes market data, technical indicators, news sentiment, and more.
    """
    console = Console()
    timeframe_val = timeframe.lower()
    if timeframe_val not in ALLOWED_TIMEFRAMES:
        display_error(
            f"Invalid timeframe '{timeframe}'. Allowed values are: {', '.join(ALLOWED_TIMEFRAMES)}."
        )
        raise typer.Exit(code=1)
    output_format_enum = output
    display_info(f"Market analysis for {symbol} ({timeframe_val}) started.")
    results = {}
    errors = {}
    # Suppress stdout and stderr for noisy third-party warnings/errors
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        # 1. Technical Analysis (MarketAnalyzer)
        try:
            analyzer = MarketAnalyzer(symbol=symbol, timeframe=timeframe_val)
            analysis_results = analyzer.analyze()
            results.update(analysis_results)  # Flatten into top-level
        except Exception as e:
            errors['technical_analysis'] = str(e)
        # 2. Risk Management
        try:
            risk_service = RiskManagementService(cache_enabled=True)
            risk_summary = risk_service.get_risk_assessment_summary(
                symbol=symbol,
                timeframe=timeframe_val,
                account_balance=account_balance,
                risk_per_trade_percent=risk_per_trade_percent,
                stop_loss_percent=stop_loss_percent
            )
            results['risk_management'] = risk_summary
        except Exception as e:
            errors['risk_management'] = str(e)
        # 3. Sentiment Analysis
        try:
            sentiment_service = SentimentDataService()
            sentiment_twitter = sentiment_service.fetch_social_sentiment(symbol, timeframe_val, source="twitter", limit=10)
            sentiment_reddit = sentiment_service.fetch_social_sentiment(symbol, timeframe_val, source="reddit", limit=10)
            sentiment_news = sentiment_service.fetch_social_sentiment(symbol, timeframe_val, source="news", limit=10)
            # Remove 'error' keys from sentiment results for CLI output
            if isinstance(sentiment_twitter, dict):
                sentiment_twitter.pop('error', None)
            if isinstance(sentiment_reddit, dict):
                sentiment_reddit.pop('error', None)
            if isinstance(sentiment_news, dict):
                sentiment_news.pop('error', None)
            results['sentiment'] = {
                'twitter': sentiment_twitter,
                'reddit': sentiment_reddit,
                'news': sentiment_news,
                'education': sentiment_service.get_educational_content()
            }
        except Exception as e:
            errors['sentiment'] = str(e)
        # 4. Adaptive Indicators
        try:
            adaptive_service = AdaptiveIndicatorService()
            df = None
            if 'technical_analysis' in results and results['technical_analysis'].get('data_with_indicators') is not None:
                df = results['technical_analysis']['data_with_indicators']
            if df is not None and not df.empty:
                ama = adaptive_service.adaptive_moving_average(df)
                drsi = adaptive_service.dynamic_rsi_periods(df)
                results['adaptive_indicators'] = {
                    'adaptive_moving_average': ama.to_dict() if ama is not None else None,
                    'dynamic_rsi': drsi.to_dict() if drsi is not None else None,
                    'education': adaptive_service._get_adaptive_indicator_educational_content()
                }
            else:
                results['adaptive_indicators'] = {'error': 'No data for adaptive indicators.'}
        except Exception as e:
            errors['adaptive_indicators'] = str(e)
        # 5. Multi-Timeframe Analysis
        try:
            async def run_mta():
                mta = TimeframeAnalyzer(symbol=symbol, timeframes=DEFAULT_TIMEFRAMES)
                indicator_configs = [
                    {'name': 'sma', 'params': {'length': 20}},
                    {'name': 'ema', 'params': {'length': 50}},
                    {'name': 'rsi', 'params': {'length': 14}},
                ]
                return await mta.run_full_analysis(indicator_configs=indicator_configs, data_limit=100, use_cache=True)
            mta_results = asyncio.run(run_mta())
            results['multi_timeframe'] = mta_results
        except Exception as e:
            errors['multi_timeframe'] = str(e)
    # Aggregate errors if any (do not print to CLI)
    if errors:
        results['errors'] = errors
    save_to_file = output_format_enum in [OutputFormat.TXT, OutputFormat.JSF, OutputFormat.HTML]
    if 'error' in results:
        console.print(f"[bold]General Overview[/bold]\n{results['error']}")
        console.print("[bold green]Technical Indicators[/bold green]\nUnable to calculate indicators: no data available.")
        console.print("[bold yellow]Market Scenarios[/bold yellow]\nNo scenarios available: no data.")
        raise typer.Exit(code=0)
    display_market_analysis(
        analysis_results=results,
        symbol=symbol,
        timeframe_str=timeframe_val,
        output_format_enum=output_format_enum,
        explain=explain,
        save_to_file=save_to_file
    )
    if save_to_file:
        pass
    elif output_format_enum == OutputFormat.TEXT or output_format_enum == OutputFormat.JSON:
        pass
    if output_format_enum == OutputFormat.TEXT and 'summary' in results and results['summary'].get('general_overview'):
        general_overview = results['summary'].get('general_overview')
        if general_overview:
            console.print(f"[bold]General Overview[/bold]\n{general_overview}")
    if output_format_enum == OutputFormat.TEXT:
        console.print("[bold yellow]Market Scenarios[/bold yellow]\nNo scenarios available: no data.")

if __name__ == "__main__":
    analyzer_app() 