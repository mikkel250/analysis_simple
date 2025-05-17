import io
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from src.cli.education import get_indicator_explanation
from src.cli.display import format_price # Assuming format_price is a general display utility, may or may not be used directly here

console = Console()

def _clean_text(text: str) -> str:
    """Clean text by removing line breaks and extra spaces."""
    if not text:
        return text
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\v', ' ').replace('\f', ' ')
    return ' '.join(text.split())

def format_text_analysis(analysis_results: dict, symbol: str, timeframe: str, explain: bool = False) -> str:
    """
    Formats the market analysis data into a text string suitable for console output.
    Uses Rich library features for better formatting.
    This function captures what would be printed to the console.

    Args:
        analysis_results (dict): The dictionary containing all analysis data.
                                 Expected keys include 'summary', 'indicators', 'price_action', etc.
        symbol (str): The trading symbol (e.g., 'BTC-USD').
        timeframe (str): The timeframe of the analysis (e.g., '1d').
        explain (bool): Whether to include educational explanations for indicators.

    Returns:
        str: A formatted string representing the market analysis.
    """
    capture_console = Console(file=io.StringIO(), width=console.width)

    summary = analysis_results.get('summary', {})
    indicators = analysis_results.get('indicators', {})
    price_action = analysis_results.get('price_action', {})
    candlestick_patterns = analysis_results.get('candlestick_patterns', [])
    volume_analysis = analysis_results.get('volume_analysis', {})
    market_cases = analysis_results.get('market_cases', {})

    capture_console.print(Panel(f"[bold cyan]Market Analysis for {symbol} ({timeframe.upper()})[/bold cyan]", title="Analysis Report", expand=False))

    if 'general_overview' in summary and summary['general_overview']:
        capture_console.print(Panel(summary['general_overview'], title="[bold]General Overview[/bold]", expand=False))

    if price_action:
        price_table = Table(title="[bold]Price Action[/bold]")
        price_table.add_column("Metric", style="dim")
        price_table.add_column("Value")
        for key, value in price_action.items():
            price_table.add_row(key.replace('_', ' ').title(), str(value))
        capture_console.print(price_table)
    
    if indicators:
        capture_console.print(Panel("[bold green]Technical Indicators[/bold green]", expand=False))
        for name, data in indicators.items():
            if isinstance(data, dict):
                interpretation = data.get('interpretation', 'N/A')
                value_display = []
                if 'value' in data and data['value'] is not None:
                    val = data['value']
                    value_display.append(f"Value: {val:.4f}" if isinstance(val, float) else f"Value: {val}")
                if 'values' in data and isinstance(data['values'], dict):
                    for k, v_item in data['values'].items():
                        value_display.append(f"{k.replace('_',' ').title()}: {v_item:.2f}" if isinstance(v_item, float) else f"{k.replace('_',' ').title()}: {v_item}")
                
                indicator_text = f"[bold]{name.upper()}[/bold]: {interpretation}"
                if value_display:
                    indicator_text += f"\n  └─ " + ", ".join(value_display)

                if 'recommendation' in data and data['recommendation']:
                    rec_color = "green" if data['recommendation'] == "BUY" else "red" if data['recommendation'] == "SELL" else "yellow"
                    indicator_text += f"\n  └─ Recommendation: [{rec_color}]{data['recommendation']}[/{rec_color}]"

                capture_console.print(indicator_text)
                if explain:
                    explanation = get_indicator_explanation(name)
                    if explanation:
                        capture_console.print(Markdown(f"> {explanation}"))
                capture_console.print()
            else:
                capture_console.print(f"[bold]{name.upper()}[/bold]: {data}")
                if explain:
                    explanation = get_indicator_explanation(name)
                    if explanation:
                        capture_console.print(Markdown(f"> {explanation}"))
                capture_console.print()

    if candlestick_patterns:
        capture_console.print(Panel("[bold magenta]Candlestick Patterns Detected[/bold magenta]", expand=False))
        for p_info in candlestick_patterns:
            capture_console.print(f"- {p_info.get('name', 'Unknown Pattern')} (Date: {p_info.get('date', 'N/A')})")
        capture_console.print()

    if volume_analysis and 'interpretation' in volume_analysis:
        capture_console.print(Panel(f"[bold]Volume Analysis[/bold]: {volume_analysis['interpretation']}", expand=False))
        if 'details' in volume_analysis and isinstance(volume_analysis['details'], dict):
            vol_details_table = Table(show_header=False)
            vol_details_table.add_column("Metric", style="dim")
            vol_details_table.add_column("Value")
            for k, v_detail in volume_analysis['details'].items():
                vol_details_table.add_row(k.replace('_',' ').title(), str(v_detail))
            capture_console.print(vol_details_table)
        capture_console.print()
        
    if market_cases:
        capture_console.print(Panel("[bold yellow]Market Scenarios & Cases[/bold yellow]", expand=False))
        for case_type, case_data in market_cases.items():
            capture_console.print(f"[italic]{case_type.replace('_', ' ').title()}:[/italic]")
            if isinstance(case_data, dict) and 'summary' in case_data:
                capture_console.print(f"  Summary: {case_data['summary']}")
                if 'confidence' in case_data:
                    capture_console.print(f"  Confidence: {case_data['confidence']}")
                if 'key_levels' in case_data and case_data['key_levels']:
                    capture_console.print(f"  Key Levels: {case_data['key_levels']}")
            else:
                capture_console.print(f"  {case_data}")
            capture_console.print()

    capture_console.print(Panel(
        "[dim italic]This analysis is for informational purposes only and does not constitute financial advice. "
        "Market conditions can change rapidly. Always do your own research (DYOR) before making any trading decisions.[/dim italic]", 
        title="Disclaimer", 
        expand=False
    ))

    return capture_console.file.getvalue() 