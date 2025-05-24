import json
import os
import webbrowser
from rich.console import Console

from src.cli.commands.analyzer_modules.common import OutputFormat, AnalyzerError
from src.cli.commands.analyzer_modules.json_utils import NumpyEncoder, preprocess_for_json
from src.cli.commands.analyzer_modules.html_generator import generate_html_report
from src.cli.commands.analyzer_modules.text_renderer import format_text_analysis
from src.cli.commands.analyzer_modules.file_utils import _ensure_output_directory, _generate_output_filename, _strip_ansi_codes

# It might be better to have these display functions in a more central UI/display utility module
# For now, keeping them here if formatters.py is the main interaction point for these messages.
def display_success(message: str):
    Console().print(f"[bold green]SUCCESS:[/] {message}")

def display_warning(message: str):
    Console().print(f"[bold yellow]WARNING:[/] {message}")

def display_info(message: str):
    Console().print(f"[bold blue]INFO:[/] {message}")

def display_error(message: str):
    Console().print(f"[bold red]ERROR:[/] {message}")

def display_market_analysis(
    analysis_results: dict,
    symbol: str,
    timeframe_str: str,
    output_format_enum: OutputFormat,
    explain: bool = False,
    save_to_file: bool = False,
):
    """
    Main dispatch function for displaying market analysis in various formats.

    Args:
        analysis_results: The core analysis data.
        symbol: Trading symbol.
        timeframe_str: Timeframe of the analysis.
        output_format_enum: The desired OutputFormat (enum member).
        explain: Whether to include educational explanations (passed to renderers).
        save_to_file: If True, output is saved to a file. Otherwise, printed to console where appropriate.
    """
    output_content = ""
    output_filename = None
    output_dir_path = None # Will be set if saving to file

    # Determine output directory if saving
    if save_to_file:
        # _ensure_output_directory uses the string value of the enum for subfolder creation
        output_dir_path = _ensure_output_directory(output_format_enum.value) 
        if not output_dir_path:
            raise AnalyzerError(f"Could not create or access output directory for {output_format_enum.value}")
    
    # Generate content based on format
    if output_format_enum == OutputFormat.JSON or output_format_enum == OutputFormat.JSF:
        processed_data = preprocess_for_json(analysis_results)
        output_content = json.dumps(processed_data, indent=2, cls=NumpyEncoder)
        if save_to_file:
            output_filename = _generate_output_filename(symbol, timeframe_str, output_format_enum, output_dir_path)

    elif output_format_enum == OutputFormat.HTML:
        # HTML is always saved to a file as per original logic implied
        # The generate_html_report now handles saving charts internally based on its own param if needed.
        if not output_dir_path: # HTML must be saved
            output_dir_path = _ensure_output_directory(OutputFormat.HTML.value)
            if not output_dir_path:
                raise AnalyzerError("Could not create or access output directory for HTML")
                
        output_content = generate_html_report(
            analysis_results=analysis_results, 
            symbol=symbol, 
            timeframe=timeframe_str, 
            output_dir=output_dir_path, # For saving images by html_generator
            explain=explain,
            save_charts=True # Assume charts are saved for HTML by default
        )
        # Filename for HTML content itself
        output_filename = _generate_output_filename(symbol, timeframe_str, OutputFormat.HTML, output_dir_path)

    elif output_format_enum == OutputFormat.TEXT or output_format_enum == OutputFormat.TXT:
        output_content = format_text_analysis(analysis_results, symbol, timeframe_str, explain)
        if save_to_file: # TXT implies saving
            output_filename = _generate_output_filename(symbol, timeframe_str, output_format_enum, output_dir_path)
            output_content_for_file = _strip_ansi_codes(output_content) # Ensure clean text for file

    else:
        raise AnalyzerError(f"Unsupported output format: {output_format_enum}")

    # Handle file saving or console printing
    if output_filename and output_dir_path: # This implies save_to_file was true and filename generated
        try:
            # Use specific content for TXT if it was further processed (e.g. ANSI stripped)
            content_to_write = output_content_for_file if output_format_enum == OutputFormat.TXT and 'output_content_for_file' in locals() else output_content
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(content_to_write)
            display_success(f"Analysis report saved to: {os.path.abspath(output_filename)}")

            if output_format_enum == OutputFormat.HTML:
                try:
                    webbrowser.open(f"file://{os.path.abspath(output_filename)}")
                except Exception as e_web:
                    display_warning(f"Could not open HTML report in browser: {e_web}")
        except IOError as e:
            raise AnalyzerError(f"Error saving report to {output_filename}: {e}")
    else:
        # Print to console if not saving to file (for TEXT and JSON)
        # HTML is primarily a file output; direct console print of HTML is not useful.
        if output_format_enum == OutputFormat.TEXT:
            Console().print(output_content) # Output from format_text_analysis is Rich-compatible
        elif output_format_enum == OutputFormat.JSON:
            Console().print(output_content) # JSON string 