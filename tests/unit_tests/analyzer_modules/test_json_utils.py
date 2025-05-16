import unittest
import json
import numpy as np
import pandas as pd

# Attempt to import from the new module structure
try:
    from src.cli.commands.analyzer_modules.json_utils import NumpyEncoder, preprocess_for_json
except ImportError:
    print("Error: Could not import json_utils module. Ensure PYTHONPATH is set correctly or tests are run from project root.")
    # Fallback definitions for the test to be parsable by the edit_file tool
    class NumpyEncoder(json.JSONEncoder): pass
    def preprocess_for_json(obj): return obj


class TestJsonUtils(unittest.TestCase):

    def test_numpy_encoder(self):
        """Test the NumpyEncoder for various numpy and pandas types."""
        data = {
            "np_int": np.int64(10),
            "np_float": np.float32(3.14),
            "np_array": np.array([1, 2, 3]),
            "pd_series": pd.Series([4, 5, 6], name="test_series"),
            "pd_dataframe": pd.DataFrame({"A": [7, 8], "B": [9, 10]}),
            "np_bool": np.bool_(True),
            "regular_int": 100,
            "regular_float": 2.718,
            "regular_list": [11, 12],
            "regular_bool": False,
            "non_serializable": object() # Test fallback to string
        }
        
        expected_after_numpy_encoder = {
            "np_int": 10,
            "np_float": 3.14,
            "np_array": [1, 2, 3],
            "pd_series": {"0": 4, "1": 5, "2": 6}, # pd.Series.to_dict() default
            "pd_dataframe": {"A": {"0": 7, "1": 8}, "B": {"0": 9, "1": 10}}, # pd.DataFrame.to_dict() default
            "np_bool": True,
            "regular_int": 100,
            "regular_float": 2.718,
            "regular_list": [11, 12],
            "regular_bool": False,
            "non_serializable": str(data["non_serializable"])
        }

        # Test encoding
        json_string = json.dumps(data, cls=NumpyEncoder, sort_keys=True)
        decoded_data = json.loads(json_string)

        # Due to float precision and dict ordering, compare key by key
        self.assertEqual(decoded_data["np_int"], expected_after_numpy_encoder["np_int"])
        self.assertAlmostEqual(decoded_data["np_float"], expected_after_numpy_encoder["np_float"], places=5)
        self.assertEqual(decoded_data["np_array"], expected_after_numpy_encoder["np_array"])
        self.assertEqual(decoded_data["pd_series"], expected_after_numpy_encoder["pd_series"])
        self.assertEqual(decoded_data["pd_dataframe"], expected_after_numpy_encoder["pd_dataframe"])
        self.assertEqual(decoded_data["np_bool"], expected_after_numpy_encoder["np_bool"])
        self.assertEqual(decoded_data["regular_int"], expected_after_numpy_encoder["regular_int"])
        self.assertAlmostEqual(decoded_data["regular_float"], expected_after_numpy_encoder["regular_float"], places=5)
        self.assertEqual(decoded_data["regular_list"], expected_after_numpy_encoder["regular_list"])
        self.assertEqual(decoded_data["regular_bool"], expected_after_numpy_encoder["regular_bool"])
        self.assertEqual(decoded_data["non_serializable"], expected_after_numpy_encoder["non_serializable"])


    def test_preprocess_for_json(self):
        """Test the preprocess_for_json function for various types including NaN/Inf."""
        class MockObjectWithToDict:
            def to_dict(self):
                return {"key": "value", "nested_np": np.int32(5)}

        data = {
            "np_int": np.int64(20),
            "np_float_nan": np.nan,
            "np_float_inf": np.inf,
            "np_float_neg_inf": -np.inf,
            "np_array_with_nan": np.array([1.0, np.nan, 3.0]),
            "pd_series_with_nan": pd.Series([np.nan, 10.0]),
            "nested_dict": {
                "np_bool": np.bool_(False),
                "np_float_val": np.float64(1.618)
            },
            "list_of_mixed": [
                np.int16(1), 
                {"key_inf": np.inf}, 
                pd.DataFrame({"C": [np.nan]})
            ],
            "custom_obj": MockObjectWithToDict(),
            "already_serializable_str": "hello",
            "already_serializable_none": None,
        }

        expected_preprocessed = {
            "np_int": 20,
            "np_float_nan": None, # NaN becomes None
            "np_float_inf": None, # Inf becomes None
            "np_float_neg_inf": None, # -Inf becomes None
            "np_array_with_nan": [1.0, None, 3.0],
            "pd_series_with_nan": {"0": None, "1": 10.0},
            "nested_dict": {
                "np_bool": False,
                "np_float_val": 1.618
            },
            "list_of_mixed": [
                1, 
                {"key_inf": None}, 
                {"C": {"0": None}}
            ],
            "custom_obj": {"key": "value", "nested_np": 5},
            "already_serializable_str": "hello",
            "already_serializable_none": None,
        }
        
        processed_data = preprocess_for_json(data)

        # Using json.dumps to compare complex nested structures after processing
        # This ensures that the output of preprocess_for_json IS actually JSON serializable with standard encoder
        processed_json_str = json.dumps(processed_data, sort_keys=True)
        expected_json_str = json.dumps(expected_preprocessed, sort_keys=True)
        
        self.assertEqual(processed_json_str, expected_json_str)

    def test_preprocess_non_serializable_fallback(self):
        """Test preprocess_for_json falls back to str for unknown non-serializable objects."""
        class SomeObject:
            pass
        
        obj = SomeObject()
        processed = preprocess_for_json(obj)
        self.assertEqual(processed, str(obj))

        data_with_obj = {"a": 1, "b": obj}
        processed_data = preprocess_for_json(data_with_obj)
        self.assertEqual(processed_data["a"], 1)
        self.assertEqual(processed_data["b"], str(obj))

if __name__ == '__main__':
    unittest.main() 