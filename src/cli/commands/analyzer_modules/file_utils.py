import os
import re
import datetime
from .common import OutputFormat # Assuming common.py is in the same directory

def _ensure_output_directory(output_type: str) -> str:
    """
    Ensure the output directory exists for the given output type.
    
    Args:
        output_type: Type of output (txt, json, html)
        
    Returns:
        Full path to the output directory
    """
    # Convert output type to lowercase and handle special cases
    output_type_lower = output_type.lower()
    if output_type_lower == 'txt':
        directory = 'txt'
    elif output_type_lower == 'jsf':
        directory = 'json'
    elif output_type_lower == 'html':
        directory = 'html'
    else:
        directory = output_type_lower
    
    # Create base directory
    base_dir = os.path.join(os.getcwd(), 'saved_analysis')
    if not os.path.exists(base_dir):
        try:
            os.makedirs(base_dir)
            # Consider adding logging here if needed
        except OSError as e:
            # Consider how to handle this error, maybe raise it
            print(f"Error creating base directory: {e}") # For now, print
            return os.getcwd()  # Fallback to current directory
    
    # Create output type subdirectory
    output_dir = os.path.join(base_dir, directory)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            # Consider adding logging here if needed
        except OSError as e:
            # Consider how to handle this error, maybe raise it
            print(f"Error creating output directory: {e}") # For now, print
            return base_dir  # Fallback to base directory
    
    return output_dir

def _generate_output_filename(symbol: str, timeframe: str, output_format: OutputFormat, output_dir: str) -> str:
    """
    Generate a filename for the output file.
    
    Args:
        symbol: Market symbol
        timeframe: Trading timeframe
        output_format: Type of output (enum member)
        output_dir: The directory where the file will be saved.
        
    Returns:
        Full path to the output file
    """
    # Clean up symbol and timeframe for filename
    clean_symbol = symbol.replace('-', '_').replace('/', '_')
    clean_timeframe = timeframe.lower()
    
    # Get current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine file extension from OutputFormat enum value
    extension = output_format.value
    if output_format == OutputFormat.JSF: # Special case for .jsf if it maps to .json extension
        extension = 'json'
        
    filename = f"{clean_symbol}_{clean_timeframe}_{timestamp}.{extension}"
    
    return os.path.join(output_dir, filename)

def _strip_ansi_codes(text: str) -> str:
    """
    Remove all ANSI color and formatting codes from text.
    
    Args:
        text: Text string that may contain ANSI codes
        
    Returns:
        Clean text with all ANSI codes removed
    """
    # Standard ANSI escape pattern
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-9;]*[ -/]*[@-~])')
    text = ansi_escape.sub('', text)
    
    # Remove any square brackets with content that might be color codes
    text = re.sub(r'\[[0-9;]*[mGKH]', '', text)
    
    # Remove any remaining escape sequences
    text = re.sub(r'\x1B|\033', '', text)
    
    return text 