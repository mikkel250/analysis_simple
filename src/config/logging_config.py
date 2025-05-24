import logging
import logging.config
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL = os.environ.get("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "[%(module)s.%(funcName)s:%(lineno)d] - %(message)s"
            ),
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s - %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": LOG_LEVEL,
            "formatter": "standard",
            "filename": "app.log",  # Default log file name
            "maxBytes": 1024 * 1024 * 5,  # 5 MB
            "backupCount": 5,
            "encoding": "utf-8",
        },
        # You can add more handlers here, e.g., for specific modules or error reporting
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file"],  # Default to console and file
            "level": LOG_LEVEL,
            "propagate": False,  # Avoid duplicating logs to parent if handlers are set
        },
        "httpx": {  # Example: Quieter logging for noisy libraries
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
        "aiohttp": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        },
        "ccxt": {
            "handlers": ["console", "file"],
            "level": "WARNING",
            "propagate": False,
        }
        # Add other specific loggers if needed
    },
}

def setup_logging():
    """Initializes logging configuration for the application."""
    # Ensure the directory for the log file exists if file handler is used
    log_file_path = LOGGING_CONFIG.get('handlers', {}).get('file', {}).get('filename')
    if log_file_path:
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except OSError as e:
                # Use a basic print here since logging might not be fully set up
                print(
                    f"Warning: Could not create log directory {log_dir}. "
                    f"File logging might fail. Error: {e}",
                    file=sys.stderr
                )

    logging.config.dictConfig(LOGGING_CONFIG)
    # Test message
    # logging.getLogger(__name__).info(
    #     "Logging configured successfully. LOG_LEVEL set to %s.", LOG_LEVEL
    # )

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance with the given name.
    It's recommended to call setup_logging() once at application startup.
    """
    return logging.getLogger(name)

def set_log_level(level: str):
    """
    Programmatically set the log level for all handlers and loggers at runtime.
    Useful for CLI flags like --verbose or --debug.
    """
    level = level.upper()
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    # Update all loggers in logging.root.manager.loggerDict
    for logger_name, logger_obj in logging.root.manager.loggerDict.items():
        if isinstance(logger_obj, logging.Logger):
            logger_obj.setLevel(level)
            for handler in getattr(logger_obj, 'handlers', []):
                handler.setLevel(level)

# Automatically setup logging when this module is imported for the first time
# This is a common pattern, but consider if explicit setup at app entry point is better.
# For now, let's do it here for simplicity.
_logging_initialized = False
if not _logging_initialized:
    setup_logging()
    _logging_initialized = True 