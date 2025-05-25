import subprocess
import sys
import os
import pytest

def test_cli_analyzer_basic():
    """End-to-end test: Run the CLI tool for BTC-USDT 1h and check output."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    cmd = [
        sys.executable, '-m', 'src.cli.commands.analyzer',
        'analyze', 'BTC-USDT', '--timeframe', '1h'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
    output = result.stdout + '\n' + result.stderr
    # Check for key output sections
    assert 'Market Analysis for BTC-USDT' in output
    assert 'General Overview' in output or 'general overview' in output.lower()
    assert 'Technical Indicators' in output or 'technical indicators' in output.lower()
    assert 'Market Scenarios' in output or 'market scenarios' in output.lower()
    # New: Fail if error or warning present
    # Only fail if 'error:' or 'error -' appears at the start of a line, or if a traceback is present
    error_lines = [line for line in output.lower().splitlines() if line.strip().startswith('error:') or line.strip().startswith('error -')]
    assert not error_lines, f"Error found in CLI output:\n{output}"
    assert 'traceback' not in output.lower(), f"Traceback found in CLI output:\n{output}"
    assert '[x]' not in output.lower(), f"Warning found in CLI output:\n{output}"
    assert 'strategy requires the following argument' not in output.lower(), f"Strategy argument error in CLI output:\n{output}"
    assert result.returncode == 0, f"CLI exited with code {result.returncode}. Output:\n{output}"
    # Optionally print output for debugging if test fails
    if result.returncode != 0 or 'error' in output.lower():
        print('CLI Output:', output) 