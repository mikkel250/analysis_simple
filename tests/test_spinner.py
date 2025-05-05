#!/usr/bin/env python3
"""
Test script for the spinner function.
"""

from src.cli.display import display_spinner
import time

# Test spinner
with display_spinner("Processing data") as spinner:
    # Simulate some processing
    for i in range(5):
        time.sleep(0.5)
        spinner.set_description(f"Processing data: {i+1}/5")

print("Processing complete!") 