"""
Configuration Loader for Seoul Market Risk ML System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to config file. If None, uses default config.yaml
        
    Returns:
        Configuration dictionary
    """
    # Load environment variables
    load_dotenv()
    
    # Determine config file path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Substitute environment variables
    config = _substitute_env_vars(config)
    
    # Validate required configuration sections
    _validate_config(config)
    
    return config


def _substitute_env_vars(obj: Any) -> Any:
    """Recursively substitute environment variables in configuration."""
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_substitute_env_vars(item) for item in obj]
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        # Extract environment variable name
        env_var = obj[2:-1]
        value = os.getenv(env_var)
        if value is None:
            logger.warning(f"Environment variable {env_var} not found, using placeholder")
            return obj
        return value
    else:
        return obj


def _validate_config(config: Dict[str, Any]) -> None:
    """Validate that required configuration sections exist."""
    required_sections = [
        'data',
        'models', 
        'risk_scoring',
        'clustering',
        'loan_calculation'
    ]
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section '{section}' not found")
    
    # Validate critical sub-configurations
    if 'weights' not in config['risk_scoring']:
        raise ValueError("Risk scoring weights not configured")
    
    if 'levels' not in config['risk_scoring']:
        raise ValueError("Risk scoring levels not configured")
    
    logger.info("Configuration validation passed")


def get_data_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get data paths as Path objects."""
    data_config = config['data']
    base_path = Path.cwd()
    
    return {
        'raw': base_path / data_config['raw_data_path'],
        'processed': base_path / data_config['processed_data_path'],
        'external': base_path / data_config['external_data_path']
    }


def get_model_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """Get model paths as Path objects."""
    models_config = config['models']
    base_path = Path.cwd()
    
    return {
        'global': base_path / models_config['global']['save_path'],
        'regional': base_path / models_config['regional']['save_path'],
        'local': base_path / models_config['local']['save_path']
    }


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    
    level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file_path', 'logs/seoul_market_risk.log')
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging configured: level={level}, file={log_file}")


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Data paths: {get_data_paths(config)}")
    print(f"Model paths: {get_model_paths(config)}")