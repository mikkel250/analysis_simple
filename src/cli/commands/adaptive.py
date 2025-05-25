"""
Adaptive Indicators CLI Command Handler

Handles commands for adaptive technical indicators.
"""

import typer
from rich.console import Console
from rich.table import Table
import pandas as pd
import logging
import json

from src.services.adaptive_indicators import AdaptiveIndicatorService
from src.services.data_fetcher import DataFetcher # For fetching data
from src.cli.commands.analyzer_modules.formatters import OutputFormat, display_error, display_info, display_success, display_warning

logger = logging.getLogger(__name__)

adaptive_app = typer.Typer(
    name="adaptive",
    help="Calculate and display adaptive technical indicators.",
    add_completion=False,
    no_args_is_help=True
)

console = Console()

@adaptive_app.command(
    name="regime",
    help="Detect market regimes using Hidden Markov Models (HMM)."
)
def get_market_regimes(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTC/USDT)."),
    timeframe: str = typer.Option("1d", "-tf", "--timeframe", help="Data timeframe."),
    limit: int = typer.Option(200, "-l", "--limit", help="Number of data points to fetch."),
    price_col: str = typer.Option("close", "-p", "--price-col", help="Price column to use for HMM features (if feature_col not set)."),
    feature_col: Optional[str] = typer.Option(None, "-fc", "--feature-col", help="Optional pre-calculated feature column for HMM."),
    n_states: int = typer.Option(2, "-s", "--states", help="Number of hidden states (regimes) for HMM."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for HMM for reproducibility."),
    use_cached_model: bool = typer.Option(True, "--cache-model/--no-cache-model", help="Use cached HMM model if available."),
    retrain_model: bool = typer.Option(False, "--retrain", help="Force retraining of HMM model even if cached model exists."),
    exchange: str = typer.Option("okx", "-ex", "--exchange", help="Exchange to fetch data from."),
    output: OutputFormat = typer.Option(OutputFormat.TEXT.value, "-o", "--output", help="Output format.", case_sensitive=False)
):
    """Detects and displays market regimes based on HMM.
    Outputs the last N regime states or full series as JSON.
    """
    display_info(f"Fetching data for {symbol}, {timeframe}, limit {limit} from {exchange}...")
    fetcher = DataFetcher(exchange_name=exchange)
    df_ohlcv = fetcher.fetch_historical_ohlcv(symbol, timeframe, limit=limit, use_cache=True)

    if df_ohlcv.empty:
        display_error(f"Could not fetch data for {symbol}.")
        raise typer.Exit(code=1)

    display_info(f"Detecting market regimes for {symbol} with {n_states} states...")
    service = AdaptiveIndicatorService()
    regime_series = service.market_regime_detector(
        df_ohlcv, 
        price_col=price_col, 
        n_states=n_states, 
        feature_col=feature_col,
        random_state_seed=seed,
        use_cached_model=use_cached_model,
        retrain_model=retrain_model
    )

    if regime_series.empty:
        display_error("Failed to detect market regimes.")
        raise typer.Exit(code=1)

    # Add regime to DataFrame for display
    df_ohlcv['regime'] = regime_series

    if output == OutputFormat.JSON:
        console.print(JSON(df_ohlcv[[price_col, 'regime']].to_json(orient='index', date_format='iso')))
    elif output == OutputFormat.TEXT:
        console.rule(f"[bold green]Market Regimes for {symbol} (Last 15 periods)[/bold green]")
        table = Table(title=f"Regimes ({n_states} states)")
        table.add_column("Timestamp", style="dim")
        table.add_column(price_col.capitalize(), justify="right")
        table.add_column("Regime", justify="center")
        
        display_df = df_ohlcv[[price_col, 'regime']].dropna().tail(15)
        for idx, row in display_df.iterrows():
            table.add_row(str(idx), f"{row[price_col]:.2f}", str(int(row['regime']))) 
        console.print(table)
        display_info("Note: Regime interpretation (e.g., state 0 = low-vol, state 1 = high-vol) depends on HMM model specifics and feature choice.")
    else:
        console.print(df_ohlcv[[price_col, 'regime']].tail(15).to_string())
        
    display_success("Market regime detection complete.")

    if output == OutputFormat.TEXT:
        educational_content = service._get_adaptive_indicator_educational_content()
        console.rule("[bold yellow]Educational Notes[/bold yellow]")
        console.print(f"[bold magenta]{educational_content['market_regimes']['title']}[/bold magenta]")
        console.print(educational_content['market_regimes']['concept'])
        console.print(f"[italic]Interpretation Note:[/] {educational_content['market_regimes']['interpretation']}")
        console.print(f"[italic]Disclaimer:[/] {educational_content['disclaimer']}")

@adaptive_app.command(
    name="ama",
    help="Calculate Volatility-Adaptive Moving Average (EMA based)."
)
def get_adaptive_ma(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTC/USDT)."),
    timeframe: str = typer.Option("1d", "-tf", "--timeframe", help="Data timeframe."),
    limit: int = typer.Option(100, "-l", "--limit", help="Number of data points to fetch."),
    price_col: str = typer.Option("close", "-p", "--price-col", help="Price column to use."),
    base_window: int = typer.Option(20, "--base-window", help="Base window for EMA and volatility."),
    vol_window: int = typer.Option(20, "--vol-window", help="Volatility calculation window."),
    vol_factor: float = typer.Option(1.0, "--vol-factor", help="Volatility scaling factor for alpha."),
    min_alpha: float = typer.Option(0.01, "--min-alpha", help="Min EMA alpha."),
    max_alpha: float = typer.Option(0.5, "--max-alpha", help="Max EMA alpha."),
    exchange: str = typer.Option("okx", "-ex", "--exchange", help="Exchange to fetch data from."),
    output: OutputFormat = typer.Option(OutputFormat.TEXT.value, "-o", "--output", help="Output format.", case_sensitive=False)
):
    """Calculates and displays a volatility-adaptive moving average."""
    display_info(f"Fetching data for {symbol}, {timeframe}, limit {limit}...")
    fetcher = DataFetcher(exchange_name=exchange)
    df_ohlcv = fetcher.fetch_historical_ohlcv(symbol, timeframe, limit=limit, use_cache=True)

    if df_ohlcv.empty:
        display_error(f"Could not fetch data for {symbol}.")
        raise typer.Exit(code=1)

    display_info(f"Calculating Adaptive MA for {symbol}...")
    service = AdaptiveIndicatorService()
    ama_series = service.adaptive_moving_average(
        df_ohlcv, price_col=price_col, base_window=base_window, 
        volatility_window=vol_window, volatility_factor=vol_factor,
        min_alpha=min_alpha, max_alpha=max_alpha
    )

    if ama_series.empty:
        display_error("Failed to calculate Adaptive MA.")
        raise typer.Exit(code=1)

    df_ohlcv['AMA'] = ama_series

    if output == OutputFormat.JSON:
        console.print(JSON(df_ohlcv[[price_col, 'AMA']].to_json(orient='index', date_format='iso')))
    elif output == OutputFormat.TEXT:
        console.rule(f"[bold green]Adaptive MA for {symbol} (Last 15 periods)[/bold green]")
        table = Table(title="Volatility-Adaptive MA")
        table.add_column("Timestamp", style="dim")
        table.add_column(price_col.capitalize(), justify="right")
        table.add_column("AMA", justify="right", style="cyan")
        
        display_df = df_ohlcv[[price_col, 'AMA']].dropna().tail(15)
        for idx, row in display_df.iterrows():
            table.add_row(str(idx), f"{row[price_col]:.2f}", f"{row['AMA']:.2f}") 
        console.print(table)
    else:
        console.print(df_ohlcv[[price_col, 'AMA']].tail(15).to_string())
        
    display_success("Adaptive MA calculation complete.")

    if output == OutputFormat.TEXT:
        educational_content = service._get_adaptive_indicator_educational_content()
        console.rule("[bold yellow]Educational Notes[/bold yellow]")
        console.print(f"[bold magenta]{educational_content['adaptive_moving_average']['title']}[/bold magenta]")
        console.print(educational_content['adaptive_moving_average']['concept'])
        console.print(f"[italic]Example Logic:[/] {educational_content['adaptive_moving_average']['example_logic']}")
        console.print(f"[italic]Disclaimer:[/] {educational_content['disclaimer']}")

@adaptive_app.command(
    name="drsi",
    help="Calculate Dynamic Period RSI based on market regimes."
)
def get_dynamic_rsi(
    symbol: str = typer.Argument(..., help="Trading symbol (e.g., BTC/USDT)."),
    timeframe: str = typer.Option("1d", "-tf", "--timeframe", help="Data timeframe."),
    limit: int = typer.Option(200, "-l", "--limit", help="Number of data points to fetch."),
    price_col: str = typer.Option("close", "-p", "--price-col", help="Price column to use."),
    base_rsi_period: int = typer.Option(14, "--base-period", help="Base RSI period."),
    min_rsi_period: int = typer.Option(7, "--min-rsi", help="Min RSI period for dynamic adjustment."),
    max_rsi_period: int = typer.Option(28, "--max-rsi", help="Max RSI period for dynamic adjustment."),
    n_hmm_states: int = typer.Option(2, "-s", "--states", help="Number of HMM states for internal regime detection."),
    hmm_seed: Optional[int] = typer.Option(None, "--hmm-seed", help="Random seed for HMM if used internally."),
    # For simplicity, regime_map CLI input is omitted. User can implement custom regime_series if needed.
    exchange: str = typer.Option("okx", "-ex", "--exchange", help="Exchange to fetch data from."),
    output: OutputFormat = typer.Option(OutputFormat.TEXT.value, "-o", "--output", help="Output format.", case_sensitive=False)
):
    """Calculates and displays RSI with dynamically adjusting periods based on HMM-detected regimes."""
    display_info(f"Fetching data for {symbol}, {timeframe}, limit {limit}...")
    fetcher = DataFetcher(exchange_name=exchange)
    df_ohlcv = fetcher.fetch_historical_ohlcv(symbol, timeframe, limit=limit, use_cache=True)

    if df_ohlcv.empty:
        display_error(f"Could not fetch data for {symbol}.")
        raise typer.Exit(code=1)

    display_info(f"Calculating Dynamic RSI for {symbol}...")
    service = AdaptiveIndicatorService()
    
    # Note: regime_series and regime_map are handled internally by dynamic_rsi_periods by default
    drsi_series = service.dynamic_rsi_periods(
        df_ohlcv, price_col=price_col, base_period=base_rsi_period,
        min_rsi_period=min_rsi_period, max_rsi_period=max_rsi_period,
        n_hmm_states=n_hmm_states, hmm_random_seed=hmm_seed
    )

    if drsi_series.empty:
        display_error("Failed to calculate Dynamic RSI.")
        raise typer.Exit(code=1)

    df_ohlcv['DynamicRSI'] = drsi_series
    # Optionally, display the regime used for context if helpful
    # df_ohlcv['regime_for_drsi'] = service.market_regime_detector(df_ohlcv, price_col=price_col, n_states=n_hmm_states, random_state_seed=hmm_seed)

    if output == OutputFormat.JSON:
        # Only include price, DRSI. Add regime if you calculate and attach it.
        console.print(JSON(df_ohlcv[[price_col, 'DynamicRSI']].to_json(orient='index', date_format='iso')))
    elif output == OutputFormat.TEXT:
        console.rule(f"[bold green]Dynamic RSI for {symbol} (Last 15 periods)[/bold green]")
        table = Table(title="Regime-Adaptive Dynamic RSI")
        table.add_column("Timestamp", style="dim")
        table.add_column(price_col.capitalize(), justify="right")
        table.add_column("DynamicRSI", justify="right", style="purple")
        # table.add_column("Regime", justify="center") # If you add regime to df_ohlcv
        
        display_df = df_ohlcv[[price_col, 'DynamicRSI']].dropna().tail(15)
        for idx, row in display_df.iterrows():
            table.add_row(str(idx), f"{row[price_col]:.2f}", f"{row['DynamicRSI']:.2f}")
            # Add regime if available: str(int(row['regime_for_drsi']))
        console.print(table)
        display_info("RSI period adapts based on HMM-detected market regimes.")
    else:
        console.print(df_ohlcv[[price_col, 'DynamicRSI']].tail(15).to_string())
        
    display_success("Dynamic RSI calculation complete.")

    if output == OutputFormat.TEXT:
        educational_content = service._get_adaptive_indicator_educational_content()
        console.rule("[bold yellow]Educational Notes[/bold yellow]")
        console.print(f"[bold magenta]{educational_content['dynamic_rsi']['title']}[/bold magenta]")
        console.print(educational_content['dynamic_rsi']['concept'])
        console.print(f"[italic]Application Note:[/] {educational_content['dynamic_rsi']['application']}")
        console.print(f"[italic]Disclaimer:[/] {educational_content['disclaimer']}")

if __name__ == '__main__':
    # For direct testing: python -m src.cli.commands.adaptive regime BTC/USDT
    logging.basicConfig(level=logging.DEBUG) # Use DEBUG for more HMM/adaptive logs
    adaptive_app() 