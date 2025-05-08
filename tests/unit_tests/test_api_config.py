import unittest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.config.api_config import (
    get_api_credentials,
    load_from_env,
    load_from_file,
    save_to_file,
    validate_credentials,
    mask_credentials,
    SUPPORTED_EXCHANGES
)

class TestApiConfig(unittest.TestCase):
    """Test cases for the API configuration module."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name) / "config"
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "api_keys.json"
        
        # Sample API credentials
        self.sample_credentials = {
            "api_key": "test_key_12345",
            "api_secret": "test_secret_abcde"
        }
        
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    @patch("src.config.api_config.CONFIG_FILE")
    def test_save_to_file(self, mock_config_file):
        """Test saving API credentials to file."""
        # Set up the mock config file to use our temporary file
        mock_config_file.return_value = self.config_file
        
        # Save credentials to file
        with patch("src.config.api_config.CONFIG_FILE", self.config_file):
            result = save_to_file("binance", self.sample_credentials["api_key"], self.sample_credentials["api_secret"])
            
        # Verify the result
        self.assertTrue(result)
        
        # Verify the file was created with the correct content
        self.assertTrue(self.config_file.exists())
        
        with open(self.config_file, "r") as f:
            saved_config = json.load(f)
            
        self.assertIn("binance", saved_config)
        self.assertEqual(saved_config["binance"]["api_key"], self.sample_credentials["api_key"])
        self.assertEqual(saved_config["binance"]["api_secret"], self.sample_credentials["api_secret"])
    
    @patch("src.config.api_config.CONFIG_FILE")
    def test_load_from_file(self, mock_config_file):
        """Test loading API credentials from file."""
        # Set up the mock config file to use our temporary file
        mock_config_file.return_value = self.config_file
        
        # Create a test config file
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        test_config = {
            "binance": self.sample_credentials,
            "bybit": {
                "api_key": "bybit_key",
                "api_secret": "bybit_secret"
            }
        }
        
        with open(self.config_file, "w") as f:
            json.dump(test_config, f)
            
        # Load credentials from file
        with patch("src.config.api_config.CONFIG_FILE", self.config_file):
            binance_creds = load_from_file("binance")
            bybit_creds = load_from_file("bybit")
            nonexistent_creds = load_from_file("nonexistent")
            
        # Verify the loaded credentials
        self.assertEqual(binance_creds, self.sample_credentials)
        self.assertEqual(bybit_creds["api_key"], "bybit_key")
        self.assertEqual(bybit_creds["api_secret"], "bybit_secret")
        self.assertEqual(nonexistent_creds, {})
    
    @patch.dict(os.environ, {
        "BINANCE_API_KEY": "env_binance_key",
        "BINANCE_API_SECRET": "env_binance_secret"
    })
    def test_load_from_env(self):
        """Test loading API credentials from environment variables."""
        binance_creds = load_from_env("binance")
        
        # Verify the loaded credentials
        self.assertEqual(binance_creds["api_key"], "env_binance_key")
        self.assertEqual(binance_creds["api_secret"], "env_binance_secret")
        
        # Test with non-existent exchange
        nonexistent_creds = load_from_env("nonexistent")
        self.assertEqual(nonexistent_creds, {})
    
    def test_validate_credentials(self):
        """Test validating API credentials."""
        # Valid credentials
        valid_creds = {
            "api_key": "valid_key",
            "api_secret": "valid_secret"
        }
        self.assertTrue(validate_credentials(valid_creds))
        
        # Invalid credentials - missing key
        invalid_creds1 = {
            "api_secret": "valid_secret"
        }
        self.assertFalse(validate_credentials(invalid_creds1))
        
        # Invalid credentials - missing secret
        invalid_creds2 = {
            "api_key": "valid_key"
        }
        self.assertFalse(validate_credentials(invalid_creds2))
        
        # Invalid credentials - empty
        self.assertFalse(validate_credentials({}))
        self.assertFalse(validate_credentials(None))
        
    def test_mask_credentials(self):
        """Test masking API credentials for display."""
        # Test with standard credentials
        masked = mask_credentials(self.sample_credentials)
        self.assertEqual(masked["api_key"][:4], self.sample_credentials["api_key"][:4])
        self.assertEqual(masked["api_key"][-4:], self.sample_credentials["api_key"][-4:])
        self.assertIn("*", masked["api_key"])
        
        # Test with short credentials
        short_creds = {
            "api_key": "abc",
            "api_secret": "xyz"
        }
        masked_short = mask_credentials(short_creds)
        self.assertEqual(masked_short["api_key"], "****")
        
        # Test with empty credentials
        self.assertEqual(mask_credentials({}), {})
        self.assertEqual(mask_credentials(None), {})
    
    @patch("src.config.api_config.load_from_env")
    @patch("src.config.api_config.load_from_file")    
    def test_get_api_credentials_precedence(self, mock_load_file, mock_load_env):
        """Test that environment variables take precedence over config files."""
        # Set up mocks
        env_creds = {"api_key": "env_key", "api_secret": "env_secret"}
        file_creds = {"api_key": "file_key", "api_secret": "file_secret"}
        
        # Case 1: Both env and file have credentials
        mock_load_env.return_value = env_creds
        mock_load_file.return_value = file_creds
        
        result = get_api_credentials("binance")
        self.assertEqual(result, env_creds)
        
        # Case 2: Only file has credentials
        mock_load_env.return_value = {}
        mock_load_file.return_value = file_creds
        
        result = get_api_credentials("binance")
        self.assertEqual(result, file_creds)
        
        # Case 3: Neither has credentials
        mock_load_env.return_value = {}
        mock_load_file.return_value = {}
        
        result = get_api_credentials("binance")
        self.assertEqual(result, {})
    
    def test_supported_exchanges(self):
        """Test that we have a list of supported exchanges."""
        self.assertIsInstance(SUPPORTED_EXCHANGES, list)
        self.assertGreater(len(SUPPORTED_EXCHANGES), 0)
        self.assertIn("binance", SUPPORTED_EXCHANGES)

if __name__ == "__main__":
    unittest.main() 