import os
import sys

# Ensure src is in the path for import
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# It's important to set the environment variable *before* importing the logging config if you want to test its effect on initial setup.
# For this test, we'll assume the default LOG_LEVEL or one set externally.
# To test LOG_LEVEL override: run like `LOG_LEVEL=DEBUG python test_logging_setup.py`

try:
    from src.config.logging_config import get_logger, setup_logging # setup_logging is called on import of logging_config
except ImportError as e:
    print(f"Error importing logging configuration: {e}")
    print("Ensure src/config/__init__.py and src/config/logging_config.py are correctly set up.")
    sys.exit(1)

logger = get_logger(__name__) # Use the name of the current module

def run_tests():
    print("--- Starting Logging Test ---")
    logger.debug("This is a DEBUG message. It should appear if LOG_LEVEL is DEBUG.")
    logger.info("This is an INFO message. It should appear if LOG_LEVEL is INFO or DEBUG.")
    logger.warning("This is a WARNING message.")
    logger.error("This is an ERROR message.")
    logger.critical("This is a CRITICAL message.")

    print("\n--- Testing Exception Logging ---")
    # The following block intentionally triggers a ZeroDivisionError to test error logging.
    # Comment out or guard this block if you do not want to trigger the error during normal test runs.
    test_error_logging = True  # Set to False to skip ZeroDivisionError test
    if test_error_logging:
        try:
            x = 1 / 0
        except ZeroDivisionError:
            logger.error("A ZeroDivisionError occurred!", exc_info=True)
            # For critical errors that should always be logged with full detail
            # logger.exception("A ZeroDivisionError occurred via logger.exception!")

    print("\n--- Logging Test Complete ---")
    print(f"Please check the console output and the 'app.log' file in the workspace root.")
    print(f"To test different log levels, run: LOG_LEVEL=DEBUG python {__file__}")
    print("Or set the LOG_LEVEL environment variable persistently.")

if __name__ == "__main__":
    run_tests()

# Example of getting a logger for another module (hypothetical)
# other_logger = get_logger("src.analysis.market_data")
# other_logger.info("Log message from a hypothetical other module's logger.") 