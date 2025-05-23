"""
CLI Commands Package

This package contains the command handlers for the CLI.
"""

from .indicator import indicator_app
from .price import price_app
from .status import status_app
from .clean import clean_app
# from .analyzer import analyzer_app  # Removed to prevent runpy RuntimeWarning
from .config import config_app

__all__ = [
    "indicator_app",
    "price_app",
    "status_app",
    "clean_app",
    "config_app"
] 