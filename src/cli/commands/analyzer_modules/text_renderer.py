import io
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from src.cli.education import get_indicator_explanation, get_category_explanation, get_summary_explanation
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
    price_snapshot = analysis_results.get('price_snapshot', {})
    trend_analysis = analysis_results.get('trend_analysis', {})
    volatility_analysis = analysis_results.get('volatility_analysis', {})
    support_resistance = analysis_results.get('support_resistance', {})
    volume_analysis = analysis_results.get('volume_analysis', {})
    key_indicators = analysis_results.get('key_indicators', {})
    indicators_data = analysis_results.get('indicators', {})  # fallback for legacy
    market_cases = analysis_results.get('market_cases', {})
    advanced_analytics = analysis_results.get('advanced_analytics', {})
    candlestick_patterns = analysis_results.get('candlestick_patterns', [])

    # 1. Market Overview
    capture_console.print(Panel(f"[bold cyan]Market Analysis for {symbol} ({timeframe.upper()})[/bold cyan]", title="Analysis Report", expand=False))
    if 'general_overview' in summary and summary['general_overview']:
        capture_console.print("[bold]General Overview[/bold]")
        capture_console.print(Panel(summary['general_overview'], title="[bold]General Overview[/bold]", expand=False))
        if explain:
            capture_console.print(Markdown(f"> {get_category_explanation('market_summary')}\n"))
    # Price snapshot
    if price_snapshot:
        price_table = Table(title="[bold]Price Snapshot[/bold]")
        price_table.add_column("Metric", style="dim")
        price_table.add_column("Value")
        for key, value in price_snapshot.items():
            price_table.add_row(key.replace('_', ' ').title(), str(value))
        capture_console.print(price_table)

    # 2. Technical Indicators
    capture_console.print(Panel("[bold green]Technical Indicators[/bold green]", expand=False))
    indicators_section = key_indicators if key_indicators else indicators_data
    for indicator_key, data in indicators_section.items():
        if indicator_key == "overall_indicator_confluence":
            continue
        if isinstance(data, dict):
            display_name = data.get('display_name', indicator_key.upper())
            interpretation = data.get('interpretation', data.get('explanation', 'N/A'))
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
                explanation = get_indicator_explanation(data.get('explanation_key', indicator_key))
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
    # Confluence summary
    confluence = indicators_section.get('overall_indicator_confluence', {})
    if confluence:
        note = confluence.get('note', '')
        capture_console.print(f"[italic]Confluence:[/italic] {note}")
        if explain:
            capture_console.print(Markdown("> When multiple indicators align, the signal is stronger. Confluence is a key concept in technical analysis.\n"))

    # 3. Risk Assessment
    if volatility_analysis:
        capture_console.print(Panel("[bold red]Risk Assessment[/bold red]", expand=False))
        for k, v in volatility_analysis.items():
            if k == 'details' and explain:
                capture_console.print(Markdown(f"> {v}\n"))
            elif k != 'details':
                capture_console.print(f"{k.replace('_', ' ').title()}: {v}")
        capture_console.print()
    # Add drawdown, support/resistance, or other risk factors if available
    if support_resistance:
        capture_console.print("[bold]Support/Resistance:[/bold]")
        for k, v in support_resistance.items():
            capture_console.print(f"{k.replace('_', ' ').title()}: {v}")
        capture_console.print()

    # 4. Sentiment Analysis (if available)
    sentiment = analysis_results.get('sentiment', {})
    if sentiment:
        capture_console.print(Panel("[bold blue]Sentiment Analysis[/bold blue]", expand=False))
        for k, v in sentiment.items():
            if k == 'education':
                # Always show educational content if present
                if v:
                    capture_console.print("[bold]Educational Content:[/bold]")
                    if isinstance(v, dict):
                        for ek, ev in v.items():
                            if isinstance(ev, list):
                                capture_console.print(f"  {ek.title()}: ")
                                for item in ev:
                                    capture_console.print(f"    - {item}")
                            else:
                                capture_console.print(f"  {ek.title()}: {ev}")
                    else:
                        capture_console.print(str(v))
                continue
            # v is a dict for each source (twitter, reddit, news)
            if isinstance(v, dict):
                status = v.get('status', '')
                if status != 'success':
                    # Show user-friendly error or no-data message
                    msg = status.replace('_', ' ').capitalize()
                    capture_console.print(f"[bold]{k.title()}:[/bold] [yellow]{msg}[/yellow]")
                else:
                    # Show sentiment scores
                    score = v.get('sentiment_score', 0)
                    polarity = v.get('avg_polarity', 0)
                    subjectivity = v.get('avg_subjectivity', 0)
                    count = v.get('content_items_analyzed_count', 0)
                    capture_console.print(f"[bold]{k.title()}:[/bold] Score: {score:.3f}, Polarity: {polarity:.3f}, Subjectivity: {subjectivity:.3f}, Items analyzed: {count}")
            else:
                capture_console.print(f"[bold]{k.title()}:[/bold] {v}")
        if explain:
            capture_console.print(Markdown("> Sentiment analysis reflects the mood of market participants, often derived from news, social media, or order flow.\n"))
        capture_console.print()

    # 5. Adaptive Analysis (if available)
    adaptive = advanced_analytics.get('adaptive_analysis') or analysis_results.get('adaptive_analysis')
    if adaptive:
        capture_console.print(Panel("[bold magenta]Adaptive Analysis[/bold magenta]", expand=False))
        for k, v in adaptive.items():
            capture_console.print(f"{k.replace('_', ' ').title()}: {v}")
        if explain:
            capture_console.print(Markdown("> Adaptive analysis uses dynamic models to adjust to changing market conditions.\n"))
        capture_console.print()

    # 6. Multi-Timeframe Confluence (if available)
    mtf = advanced_analytics.get('multi_timeframe_confluence') or analysis_results.get('multi_timeframe_confluence')
    if mtf:
        capture_console.print(Panel("[bold yellow]Multi-Timeframe Confluence[/bold yellow]", expand=False))
        for k, v in mtf.items():
            capture_console.print(f"{k.replace('_', ' ').title()}: {v}")
        if explain:
            capture_console.print(Markdown("> Multi-timeframe confluence means signals agree across several timeframes, increasing reliability.\n"))
        capture_console.print()

    # 7. Pattern Recognition (NEW)
    pattern_recognition = analysis_results.get("pattern_recognition", {})
    if pattern_recognition:
        capture_console.print(Panel("[bold magenta]Pattern Recognition[/bold magenta]", title="Pattern Recognition", expand=False))
        # Harmonic Patterns
        harmonic_patterns = pattern_recognition.get("harmonic_patterns", [])
        if harmonic_patterns:
            capture_console.print("[bold]Harmonic Patterns Detected:[/bold]")
            for p in harmonic_patterns:
                capture_console.print(f"- [bold]{p['name']}[/bold] | Probability: {p['probability']:.2f}")
                capture_console.print(f"  Points: {p['points']}")
                if p.get('educational_notes'):
                    capture_console.print(f"  [italic yellow]{p['educational_notes']}[/italic yellow]")
        else:
            capture_console.print("No harmonic patterns detected - data is present but no patterns found at this time.")
            capture_console.print("[dim]Note: Harmonic patterns (Gartley, Butterfly, Bat, Crab) are rare and require very specific Fibonacci ratio conditions. They typically appear at major market turning points.[/dim]")
        # Elliott Wave Analysis
        elliott_wave = pattern_recognition.get("elliott_wave_analysis", {})
        if elliott_wave:
            capture_console.print("[bold]Elliott Wave Analysis:[/bold]")
            impulse = elliott_wave.get("impulse_waves", [])
            corrective = elliott_wave.get("corrective_waves", [])
            education = elliott_wave.get("education", {})
            capture_console.print(f"  Impulse Waves Found: {len(impulse)}")
            capture_console.print(f"  Corrective Waves Found: {len(corrective)}")
            if education:
                capture_console.print("  [bold]Educational Content:[/bold]")
                for k, v in education.items():
                    capture_console.print(f"    {k.capitalize()}: [italic yellow]{v}[/italic yellow]")
        else:
            capture_console.print("No Elliott Wave analysis available.")

    # 8. Market Scenarios (Cases)
    if market_cases:
        capture_console.print(Panel("[bold yellow]Market Scenarios & Cases[/bold yellow]", expand=False))
        def extract_indicators_and_rationales(supporting_factors):
            import re
            parsed = []
            for factor in supporting_factors:
                match = re.match(r"([A-Za-z0-9_ ]+) rationale: (.+)", factor)
                if match:
                    indicator = match.group(1).upper()
                    rationale = match.group(2)
                    parsed.append((indicator, rationale))
                else:
                    parsed.append(("OTHER", factor))
            return parsed
        cases = market_cases if isinstance(market_cases, list) else market_cases.get('cases', [])
        for case in cases:
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
            indicators_and_rationales = extract_indicators_and_rationales(supporting_factors)
            for indicator, rationale in indicators_and_rationales:
                capture_console.print(f"  {indicator}: {rationale}")
            capture_console.print()

    # 9. Puzzle Pieces Analogy / Overall Summary
    capture_console.print(Panel("[bold]How the Puzzle Pieces Fit Together[/bold]", expand=False))
    puzzle_summary = "This report brings together multiple perspectives: trend, momentum, volatility, sentiment, and pattern recognition. When several pieces point in the same direction, confidence in the analysis increases. Always consider the full picture before making trading decisions."
    capture_console.print(puzzle_summary)
    if explain:
        capture_console.print(Markdown("> Each section is a piece of the puzzle. The more pieces that align, the stronger the signal.\n"))

    # 10. Disclaimer
    disclaimer_text = (
        "[dim italic]This analysis is for informational purposes only and does not constitute financial advice. "
        "Market conditions can change rapidly. Always do your own research (DYOR) before making any trading decisions.[/dim italic]"
    )
    capture_console.print(Panel(disclaimer_text, title="Disclaimer", expand=False))

    string_io_buffer = capture_console.file
    return string_io_buffer.getvalue() 