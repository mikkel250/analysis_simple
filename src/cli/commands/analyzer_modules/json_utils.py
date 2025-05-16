import json
import numpy as np
import pandas as pd

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types and other special data types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        # Handle any non-serializable objects by converting to string
        try:
            return super(NumpyEncoder, self).default(obj)
        except TypeError:
            return str(obj)

def preprocess_for_json(obj):
    """
    Recursively preprocess data before JSON serialization to ensure it's serializable.
    
    Args:
        obj: Object to preprocess
        
    Returns:
        Preprocessed object
    """
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: preprocess_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [preprocess_for_json(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return preprocess_for_json(obj.to_dict())
    else:
        # Try to serialize, if not possible, convert to string
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj) 