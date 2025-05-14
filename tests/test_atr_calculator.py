import unittest
import os
import sys

# This block ensures that the 'src' directory is in the Python path
# when this test file is run directly (e.g., python tests/test_atr_calculator.py).
# This allows the import 'from atr_calculator import ...' to work,
# assuming a project structure like:
# project_root/
#  ├── src/
#  │   └── atr_calculator.py
#  └── tests/
#      └── test_atr_calculator.py
if __name__ == '__main__':
    # Get the absolute path of the directory containing this test file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the absolute path to the project root directory (one level up)
    project_root = os.path.dirname(current_dir)
    # Construct the absolute path to the 'src' directory
    src_path = os.path.join(project_root, 'src')
    # Add 'src' directory to sys.path if it's not already there
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# This import will intentionally fail if src/atr_calculator.py or the function
# calculate_atr_trade_parameters doesn't exist. This is the "Red" phase of TDD.
from atr_calculator import calculate_atr_trade_parameters

class TestATRCalculator(unittest.TestCase):

    def test_long_position_example(self):
        """Test calculation for a long position based on a typical example."""
        entry_price = 100.0
        atr = 5.0
        risk_amount = 200.0
        multiplier = 2.0
        position_type = 'long'
        
        expected_position_size = 20.0
        # Stop Loss Distance = ATR * Multiplier = 5.0 * 2.0 = 10.0
        # Stop Loss Percentage = Stop Loss Distance / Entry Price * 100
        #                      = 10.0 / 100.0 * 100 = 10.0%
        expected_stop_loss_percentage = 10.0
        
        result = calculate_atr_trade_parameters(entry_price, atr, risk_amount, multiplier, position_type)
        self.assertIn('position_size', result, "Result should contain 'position_size'")
        self.assertIn('stop_loss_percentage', result, "Result should contain 'stop_loss_percentage'")
        self.assertAlmostEqual(result['position_size'], expected_position_size, places=5)
        self.assertAlmostEqual(result['stop_loss_percentage'], expected_stop_loss_percentage, places=5)

    def test_short_position_example(self):
        """Test calculation for a short position."""
        entry_price = 100.0
        atr = 5.0
        risk_amount = 200.0
        multiplier = 2.0
        position_type = 'short'

        expected_position_size = 20.0 # Position size calculation is independent of long/short
        # Stop Loss Distance = ATR * Multiplier = 5.0 * 2.0 = 10.0
        # Stop Loss Percentage = Stop Loss Distance / Entry Price * 100
        #                      = 10.0 / 100.0 * 100 = 10.0%
        expected_stop_loss_percentage = 10.0

        result = calculate_atr_trade_parameters(entry_price, atr, risk_amount, multiplier, position_type)
        self.assertIn('position_size', result)
        self.assertIn('stop_loss_percentage', result)
        self.assertAlmostEqual(result['position_size'], expected_position_size, places=5)
        self.assertAlmostEqual(result['stop_loss_percentage'], expected_stop_loss_percentage, places=5)

    def test_default_parameters_long(self):
        """Test with default risk amount ($100) and multiplier (1) for a long position."""
        entry_price = 50.0
        atr = 2.0
        # risk_amount = 100.0 (default)
        # multiplier = 1.0 (default)
        position_type = 'long'

        # Stop Loss Distance = ATR * Multiplier = 2.0 * 1.0 = 2.0
        # Position Size = Risk Amount / Stop Loss Distance = 100.0 / 2.0 = 50.0
        expected_position_size = 50.0
        # Stop Loss Percentage = Stop Loss Distance / Entry Price * 100
        #                      = 2.0 / 50.0 * 100 = 4.0%
        expected_stop_loss_percentage = 4.0
        
        # Assumes calculate_atr_trade_parameters has defaults for risk_amount and multiplier
        result = calculate_atr_trade_parameters(entry_price, atr, position_type=position_type)
        self.assertIn('position_size', result)
        self.assertIn('stop_loss_percentage', result)
        self.assertAlmostEqual(result['position_size'], expected_position_size, places=5)
        self.assertAlmostEqual(result['stop_loss_percentage'], expected_stop_loss_percentage, places=5)

    def test_input_validation_raises_value_error(self):
        """Test that invalid inputs for calculations raise ValueError."""
        # Test cases: (entry_price, atr, risk_amount, multiplier, position_type, description)
        invalid_inputs = [
            (100.0, 0.0, 100.0, 1.0, 'long', "Zero ATR"),
            (100.0, -1.0, 100.0, 1.0, 'long', "Negative ATR"),
            (0.0, 5.0, 100.0, 1.0, 'long', "Zero entry price for percentage"),
            (100.0, 5.0, 100.0, 0.0, 'long', "Zero multiplier"),
            (100.0, 5.0, 100.0, -1.0, 'long', "Negative multiplier"),
        ]
        
        for entry, atr_val, risk, mult, p_type, desc in invalid_inputs:
            with self.subTest(desc=desc, params=(entry, atr_val, risk, mult, p_type)):
                with self.assertRaises(ValueError):
                    calculate_atr_trade_parameters(
                        entry_price=entry, 
                        atr=atr_val, 
                        risk_amount=risk, 
                        multiplier=mult, 
                        position_type=p_type
                    )
    
    def test_invalid_parameter_types_or_values(self):
        """Test that other invalid parameter types or values raise ValueError."""
        invalid_param_sets = [
            (100.0, 5.0, -100.0, 1.0, 'long', "Negative risk amount"),
            (100.0, 5.0, 0.0, 1.0, 'long', "Zero risk amount"), # Typically not useful
            (100.0, 5.0, 100.0, 1.0, 'invalid_type', "Invalid position type"),
            ('abc', 5.0, 100.0, 1.0, 'long', "Non-numeric entry_price"),
            (100.0, 'abc', 100.0, 1.0, 'long', "Non-numeric atr"),
            (100.0, 5.0, 'abc', 1.0, 'long', "Non-numeric risk_amount"),
            (100.0, 5.0, 100.0, 'abc', 'long', "Non-numeric multiplier"),
        ]

        for entry, atr_val, risk, mult, p_type, desc in invalid_param_sets:
            with self.subTest(desc=desc, params=(entry, atr_val, risk, mult, p_type)):
                with self.assertRaises(ValueError): # Could also be TypeError for wrong types
                    calculate_atr_trade_parameters(
                        entry_price=entry, 
                        atr=atr_val, 
                        risk_amount=risk, 
                        multiplier=mult, 
                        position_type=p_type
                    )

if __name__ == '__main__':
    unittest.main() 