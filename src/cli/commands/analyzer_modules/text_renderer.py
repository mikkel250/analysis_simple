import io
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from src.cli.education import get_indicator_explanation
# get_category_explanation and format_price are unused.

console = Console()

def _clean_text(text: str) -> str:
    """Clean text by removing line breaks and extra spaces."""
    if not text:
        return text
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = text.replace('\t', ' ').replace('\v', ' ').replace('\f', ' ')
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
    indicators_data = analysis_results.get('indicators', {})
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
    
    if indicators_data:
        capture_console.print(Panel("[bold green]Technical Indicators[/bold green]", expand=False))
        
        for indicator_key, data in indicators_data.items():
            if isinstance(data, dict):
                display_name = data.get('display_name', indicator_key.upper())
                interpretation = data.get('interpretation', 'N/A')
                explanation_key_for_edu = data.get('explanation_key', indicator_key)

                value_display_parts = []
                if 'value' in data and data['value'] is not None:
                    val = data['value']
                    value_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                    value_display_parts.append(f"Value: {value_str}")
                if 'values' in data and isinstance(data['values'], dict):
                    for k, v_item in data['values'].items():
                        v_str = f"{v_item:.2f}" if isinstance(v_item, float) else str(v_item)
                        value_display_parts.append(f"{k.replace('_', ' ').title()}: {v_str}")
                
                indicator_text = f"[bold]{display_name}[/bold]: {interpretation}"
                if value_display_parts:
                    indicator_text += f"\n  └─ " + ", ".join(value_display_parts)

                if 'recommendation' in data and data['recommendation']:
                    rec = data['recommendation']
                    rec_color = "green" if rec == "BUY" else "red" if rec == "SELL" else "yellow"
                    indicator_text += f"\n  └─ Recommendation: [{rec_color}]{rec}[/{rec_color}]"

                capture_console.print(indicator_text)
                if explain:
                    explanation = get_indicator_explanation(explanation_key_for_edu)
                    if explanation:
                        capture_console.print(Markdown(f"> {explanation}\n"))
                capture_console.print()
            else:
                capture_console.print(f"[bold]{indicator_key.upper()}[/bold]: {data}")
                if explain:
                    explanation = get_indicator_explanation(indicator_key)
                    if explanation:
                        capture_console.print(Markdown(f"> {explanation}\n"))
                capture_console.print()

    if candlestick_patterns:
        capture_console.print(Panel("[bold magenta]Candlestick Patterns Detected[/bold magenta]", expand=False))
        for p_info in candlestick_patterns:
            name = p_info.get('name', 'Unknown Pattern')
            date = p_info.get('date', 'N/A')
            capture_console.print(f"- {name} (Date: {date})")
        capture_console.print()

    if volume_analysis and 'interpretation' in volume_analysis:
        vol_interpretation = volume_analysis['interpretation']
        capture_console.print(Panel(f"[bold]Volume Analysis[/bold]: {vol_interpretation}", expand=False))
        if 'details' in volume_analysis and isinstance(volume_analysis['details'], dict):
            vol_details_table = Table(show_header=False)
            vol_details_table.add_column("Metric", style="dim")
            vol_details_table.add_column("Value")
            for k, v_detail in volume_analysis['details'].items():
                vol_details_table.add_row(k.replace('_', ' ').title(), str(v_detail))
            capture_console.print(vol_details_table)
        capture_console.print()
        
    if market_cases:
        capture_console.print(Panel("[bold yellow]Market Scenarios & Cases[/bold yellow]", expand=False))
        def extract_indicators_and_rationales(supporting_factors):
            # Parse indicator names and rationale from 'INDICATOR rationale: ...' or fallback
            import re
            parsed = []
            for factor in supporting_factors:
                match = re.match(r"([A-Za-z0-9_ ]+) rationale: (.+)", factor)
                if match:
                    indicator = match.group(1).upper()
                    rationale = match.group(2)
                else:
                    indicator = None
                    rationale = factor
                parsed.append((indicator, rationale))
            return parsed
        
        if isinstance(market_cases, dict):
            for case_type, case_data in market_cases.items():
                capture_console.print(f"[italic]{case_type.replace('_', ' ').title()}:[/italic]")
                if isinstance(case_data, dict) and 'summary' in case_data:
                    capture_console.print(f"  Summary: {case_data['summary']}")
                    if 'confidence' in case_data:
                        conf = case_data['confidence']
                        capture_console.print(f"  Confidence: {conf}")
                    if 'key_levels' in case_data and case_data['key_levels']:
                        levels = case_data['key_levels']
                        capture_console.print(f"  Key Levels: {levels}")
                    # Show indicators and rationale
                    parsed = extract_indicators_and_rationales(case_data.get('supporting_factors', []))
                    indicators = [ind for ind, _ in parsed if ind]
                    if indicators:
                        capture_console.print(f"  Indicators: {', '.join(sorted(set(indicators)))}")
                    if parsed:
                        capture_console.print("  Rationale:")
                        for indicator, rationale in parsed:
                            if explain and indicator:
                                if indicator == "OPEN INTEREST":
                                    explanation = get_indicator_explanation("open_interest")
                                    if explanation:
                                        capture_console.print(Markdown(f"> [bold]{indicator}[/bold]: {explanation}"))
                                else:
                                    explanation = get_indicator_explanation(indicator.lower())
                                    if explanation:
                                        capture_console.print(Markdown(f"> [bold]{indicator}[/bold]: {explanation}"))
                            capture_console.print(f"    - {rationale}")
                else:
                    capture_console.print(f"  {case_data}")
                capture_console.print()
        elif isinstance(market_cases, list):
            for case in market_cases:
                scenario = case.get('scenario', 'Scenario')
                description = case.get('description', '')
                confidence = case.get('confidence', None)
                key_levels = case.get('key_levels', None)
                potential_triggers = case.get('potential_triggers', None)
                supporting_factors = case.get('supporting_factors', [])
                capture_console.print(f"[italic]{scenario}:[/italic]")
                if description:
                    capture_console.print(f"  {description}")
                if confidence:
                    capture_console.print(f"  Confidence: {confidence}")
                if key_levels:
                    capture_console.print(f"  Key Levels: {key_levels}")
                if potential_triggers:
                    capture_console.print(f"  Potential Triggers: {potential_triggers}")
                # Show indicators and rationale
                parsed = extract_indicators_and_rationales(supporting_factors)
                indicators = [ind for ind, _ in parsed if ind]
                if indicators:
                    capture_console.print(f"  Indicators: {', '.join(sorted(set(indicators)))}")
                if parsed:
                    capture_console.print("  Rationale:")
                    for indicator, rationale in parsed:
                        if explain and indicator:
                            if indicator == "OPEN INTEREST":
                                explanation = get_indicator_explanation("open_interest")
                                if explanation:
                                    capture_console.print(Markdown(f"> [bold]{indicator}[/bold]: {explanation}"))
                            else:
                                explanation = get_indicator_explanation(indicator.lower())
                                if explanation:
                                    capture_console.print(Markdown(f"> [bold]{indicator}[/bold]: {explanation}"))
                        capture_console.print(f"    - {rationale}")
                capture_console.print()

    disclaimer_text = (
        "[dim italic]This analysis is for informational purposes only and does not constitute financial advice. "
        "Market conditions can change rapidly. Always do your own research (DYOR) before making any trading decisions.[/dim italic]"
    )
    capture_console.print(Panel(disclaimer_text, title="Disclaimer", expand=False))

    string_io_buffer = capture_console.file
    return string_io_buffer.getvalue() 