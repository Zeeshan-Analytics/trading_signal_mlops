"""
Configuration loader utility.
Loads and validates project configuration from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import os


class ConfigLoader:
    """Load and manage project configuration."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the configuration YAML file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate that required configuration keys exist."""
        required_keys = ['project', 'data', 'features', 'model', 'signals']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the configuration value (e.g., 'data.tickers')
            default: Default value if key is not found
        
        Returns:
            Configuration value or default
        
        Example:
            config = ConfigLoader()
            tickers = config.get('data.tickers')
            batch_size = config.get('model.training.batch_size', 32)
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary."""
        return self.config


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.get_all()


if __name__ == "__main__":
    # Test the configuration loader
    try:
        config = load_config()
        print("✅ Configuration loaded successfully!")
        print(f"\nProject: {config['project']['name']}")
        print(f"Version: {config['project']['version']}")
        print(f"Tickers: {config['data']['tickers']}")
        print(f"Interval: {config['data']['interval']}")
        print(f"Signal Classes: {list(config['signals']['classes'].values())}")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")