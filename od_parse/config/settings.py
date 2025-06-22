"""
Configuration settings for od-parse library.

This module provides functions for loading and accessing configuration settings
from various sources (environment variables, config files, etc.).
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Optional import for YAML support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    # API keys (empty by default, should be provided by user)
    "api_keys": {
        "openai": "",
        "anthropic": "",
        "cohere": "",
        "huggingface": "",
        "google": ""
    },
    
    # API endpoints
    "api_endpoints": {
        "openai": "https://api.openai.com/v1",
        "anthropic": "https://api.anthropic.com",
        "cohere": "https://api.cohere.ai/v1",
        "huggingface": "https://api-inference.huggingface.co/models",
        "google": "https://generativelanguage.googleapis.com/v1"
    },
    
    # VLM models
    "vlm_models": {
        "qwen": "qwen2.5-vl",
        "claude": "claude-3-opus-vision",
        "gemini": "gemini-pro-vision"
    },
    
    # Embedding models
    "embedding_models": {
        "openai": "text-embedding-3-small",
        "cohere": "embed-english-v3.0",
        "huggingface": "sentence-transformers/all-mpnet-base-v2"
    },
    
    # Vector database configuration
    "vector_db": {
        "pgvector": {
            "default_dimension": 1536
        },
        "qdrant": {
            "default_host": "localhost",
            "default_port": 6333
        },
        "chroma": {
            "default_path": "./chroma_db"
        }
    },
    
    # System prompts
    "system_prompts": {
        "document_analysis": "You are an expert document analyzer. Analyze the document image and extract all relevant information. "
                            "Focus on understanding the document structure, identifying tables, forms, and key information. "
                            "Provide a detailed analysis of what you see in the document.",
        "table_extraction": "Focus only on tables in this document. For each table, extract all cells with their data, "
                           "preserve the table structure, and include headers if present. "
                           "Format each table as a JSON object with 'headers' and 'data' properties.",
        "form_extraction": "Focus only on form fields in this document. For each form field, identify the field label/name, "
                          "extract the field value, and determine the field type (text, checkbox, radio button, etc.). "
                          "Format your response as a JSON object with field names as keys."
    }
}

# Global configuration
_CONFIG = {}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file and environment variables.
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        Dictionary containing configuration values
    """
    global _CONFIG
    
    # Start with default configuration
    _CONFIG = DEFAULT_CONFIG.copy()
    
    # Load from configuration file if provided
    if config_path:
        file_config = _load_from_file(config_path)
        if file_config:
            _deep_update(_CONFIG, file_config)
    
    # Load from environment variables
    env_config = _load_from_env()
    if env_config:
        _deep_update(_CONFIG, env_config)
    
    # Look for config file in standard locations if not provided
    if not config_path:
        standard_locations = [
            os.path.expanduser("~/.config/od-parse/config.yaml"),
            os.path.expanduser("~/.config/od-parse/config.json"),
            os.path.join(os.getcwd(), "od-parse.yaml"),
            os.path.join(os.getcwd(), "od-parse.json")
        ]
        
        for location in standard_locations:
            if os.path.exists(location):
                file_config = _load_from_file(location)
                if file_config:
                    _deep_update(_CONFIG, file_config)
                break
    
    return _CONFIG


def get_config(key: Optional[str] = None, default: Any = None) -> Any:
    """
    Get configuration value by key.
    
    Args:
        key: Configuration key (dot notation supported for nested values)
        default: Default value if key not found
        
    Returns:
        Configuration value
    """
    global _CONFIG
    
    # Load config if not already loaded
    if not _CONFIG:
        load_config()
    
    # Return entire config if no key provided
    if key is None:
        return _CONFIG
    
    # Handle nested keys with dot notation
    if "." in key:
        parts = key.split(".")
        value = _CONFIG
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    # Simple key lookup
    return _CONFIG.get(key, default)


def _load_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    try:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return {}
        
        with open(path, "r") as f:
            if path.suffix.lower() in [".yaml", ".yml"]:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f)
                else:
                    logger.warning("PyYAML not installed. Please install it with: pip install pyyaml")
                    return {}
            else:
                return json.load(f)
    
    except Exception as e:
        logger.error(f"Error loading configuration from file: {str(e)}")
        return {}


def _load_from_env() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = {}
    
    # API keys - check standard environment variables
    if os.environ.get("OPENAI_API_KEY"):
        _set_nested_dict(config, "api_keys.openai", os.environ.get("OPENAI_API_KEY"))
    
    if os.environ.get("ANTHROPIC_API_KEY"):
        _set_nested_dict(config, "api_keys.anthropic", os.environ.get("ANTHROPIC_API_KEY"))
    
    if os.environ.get("COHERE_API_KEY"):
        _set_nested_dict(config, "api_keys.cohere", os.environ.get("COHERE_API_KEY"))
    
    if os.environ.get("HUGGINGFACE_API_KEY"):
        _set_nested_dict(config, "api_keys.huggingface", os.environ.get("HUGGINGFACE_API_KEY"))
    
    if os.environ.get("GOOGLE_API_KEY"):
        _set_nested_dict(config, "api_keys.google", os.environ.get("GOOGLE_API_KEY"))
    
    # API keys - check library-specific environment variables
    if os.environ.get("OD_PARSE_OPENAI_API_KEY"):
        _set_nested_dict(config, "api_keys.openai", os.environ.get("OD_PARSE_OPENAI_API_KEY"))
    
    if os.environ.get("OD_PARSE_ANTHROPIC_API_KEY"):
        _set_nested_dict(config, "api_keys.anthropic", os.environ.get("OD_PARSE_ANTHROPIC_API_KEY"))
    
    if os.environ.get("OD_PARSE_COHERE_API_KEY"):
        _set_nested_dict(config, "api_keys.cohere", os.environ.get("OD_PARSE_COHERE_API_KEY"))
    
    if os.environ.get("OD_PARSE_HUGGINGFACE_API_KEY"):
        _set_nested_dict(config, "api_keys.huggingface", os.environ.get("OD_PARSE_HUGGINGFACE_API_KEY"))
    
    if os.environ.get("OD_PARSE_GOOGLE_API_KEY"):
        _set_nested_dict(config, "api_keys.google", os.environ.get("OD_PARSE_GOOGLE_API_KEY"))
    
    # API endpoints
    if os.environ.get("OD_PARSE_OPENAI_API_URL"):
        _set_nested_dict(config, "api_endpoints.openai", os.environ.get("OD_PARSE_OPENAI_API_URL"))
    
    if os.environ.get("OD_PARSE_ANTHROPIC_API_URL"):
        _set_nested_dict(config, "api_endpoints.anthropic", os.environ.get("OD_PARSE_ANTHROPIC_API_URL"))
    
    if os.environ.get("OD_PARSE_COHERE_API_URL"):
        _set_nested_dict(config, "api_endpoints.cohere", os.environ.get("OD_PARSE_COHERE_API_URL"))
    
    if os.environ.get("OD_PARSE_HUGGINGFACE_API_URL"):
        _set_nested_dict(config, "api_endpoints.huggingface", os.environ.get("OD_PARSE_HUGGINGFACE_API_URL"))
    
    if os.environ.get("OD_PARSE_GOOGLE_API_URL"):
        _set_nested_dict(config, "api_endpoints.google", os.environ.get("OD_PARSE_GOOGLE_API_URL"))
    
    # VLM models
    if os.environ.get("OD_PARSE_QWEN_MODEL"):
        _set_nested_dict(config, "vlm_models.qwen", os.environ.get("OD_PARSE_QWEN_MODEL"))
    
    if os.environ.get("OD_PARSE_CLAUDE_MODEL"):
        _set_nested_dict(config, "vlm_models.claude", os.environ.get("OD_PARSE_CLAUDE_MODEL"))
    
    if os.environ.get("OD_PARSE_GEMINI_MODEL"):
        _set_nested_dict(config, "vlm_models.gemini", os.environ.get("OD_PARSE_GEMINI_MODEL"))
    
    # Embedding models
    if os.environ.get("OD_PARSE_OPENAI_EMBEDDING_MODEL"):
        _set_nested_dict(config, "embedding_models.openai", os.environ.get("OD_PARSE_OPENAI_EMBEDDING_MODEL"))
    
    if os.environ.get("OD_PARSE_COHERE_EMBEDDING_MODEL"):
        _set_nested_dict(config, "embedding_models.cohere", os.environ.get("OD_PARSE_COHERE_EMBEDDING_MODEL"))
    
    if os.environ.get("OD_PARSE_HUGGINGFACE_EMBEDDING_MODEL"):
        _set_nested_dict(config, "embedding_models.huggingface", os.environ.get("OD_PARSE_HUGGINGFACE_EMBEDDING_MODEL"))
    
    return config


def _deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update nested dictionaries."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            target[key] = _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _set_nested_dict(d: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set value in nested dictionary using dot notation for key path."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value
