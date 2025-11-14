"""
Environment Configuration Management

This module provides secure configuration management for od-parse,
loading API keys and settings from environment variables or .env files.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def load_env_file(env_path: Optional[str] = None) -> Dict[str, str]:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file. If None, looks for .env in current directory.

    Returns:
        Dictionary of environment variables loaded from file.
    """
    if env_path is None:
        env_path = ".env"

    env_vars = {}
    env_file = Path(env_path)

    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        env_vars[key] = value
                        # Set in os.environ if not already set
                        if key not in os.environ:
                            os.environ[key] = value

            logger.info(f"Loaded {len(env_vars)} environment variables from {env_path}")
        except Exception as e:
            logger.warning(f"Failed to load .env file {env_path}: {e}")
    else:
        logger.info(f"No .env file found at {env_path}")

    return env_vars


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for a specific provider.

    Args:
        provider: API provider name ('openai', 'google', 'anthropic', 'azure')

    Returns:
        API key if found, None otherwise.
    """
    # Load .env file if it exists
    load_env_file()

    key_mappings = {
        "openai": ["OPENAI_API_KEY", "OPENAI_KEY"],
        "google": ["GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_KEY"],
        "azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_KEY"],
    }

    provider = provider.lower()
    if provider not in key_mappings:
        logger.warning(f"Unknown API provider: {provider}")
        return None

    for key_name in key_mappings[provider]:
        api_key = os.environ.get(key_name)
        if api_key and api_key != "your-api-key-here":
            logger.info(f"Found API key for {provider} from {key_name}")
            return api_key

    logger.warning(
        f"No API key found for {provider}. Check your .env file or environment variables."
    )
    return None


def get_model_config(provider: str) -> Dict[str, Any]:
    """
    Get model configuration for a specific provider.

    Args:
        provider: API provider name

    Returns:
        Dictionary with model configuration.
    """
    load_env_file()

    default_models = {
        "openai": "gpt-4o-mini",
        "google": "gemini-1.5-flash",
        "anthropic": "claude-3-haiku-20240307",
        "azure": "gpt-4o-mini",
    }

    provider = provider.lower()
    config = {
        "api_key": get_api_key(provider),
        "model": os.environ.get(
            f"{provider.upper()}_MODEL", default_models.get(provider)
        ),
        "timeout": int(os.environ.get("OD_PARSE_TIMEOUT", "30")),
        "max_retries": int(os.environ.get("OD_PARSE_MAX_RETRIES", "3")),
    }

    # Azure-specific configuration
    if provider == "azure":
        config.update(
            {
                "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
                "api_version": os.environ.get(
                    "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
                ),
            }
        )

    return config


def setup_secure_environment() -> bool:
    """
    Setup secure environment for od-parse.

    Returns:
        True if at least one API key is configured, False otherwise.
    """
    load_env_file()

    # Check for any configured API keys
    providers = ["openai", "google", "anthropic", "azure"]
    configured_providers = []

    for provider in providers:
        if get_api_key(provider):
            configured_providers.append(provider)

    if configured_providers:
        logger.info(f"Configured API providers: {', '.join(configured_providers)}")
        return True
    else:
        logger.warning("No API keys configured. Please set up your .env file.")
        logger.info("Copy .env.example to .env and add your API keys.")
        return False


def validate_api_key(api_key: str) -> bool:
    """
    Validate that an API key looks reasonable.

    Args:
        api_key: API key to validate

    Returns:
        True if key looks valid, False otherwise.
    """
    if not api_key:
        return False

    # Check for placeholder values
    placeholder_values = [
        "your-api-key-here",
        "your-openai-api-key-here",
        "your-google-api-key-here",
        "your-anthropic-api-key-here",
        "sk-placeholder",
        "api-key-placeholder",
    ]

    if api_key.lower() in [p.lower() for p in placeholder_values]:
        return False

    # Basic length check (most API keys are at least 20 characters)
    if len(api_key) < 20:
        return False

    return True


def get_feature_flags() -> Dict[str, bool]:
    """
    Get feature flags from environment.

    Returns:
        Dictionary of feature flags.
    """
    load_env_file()

    def str_to_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on", "enabled")

    return {
        "enable_deep_learning": str_to_bool(
            os.environ.get("ENABLE_DEEP_LEARNING", "true")
        ),
        "enable_advanced_ocr": str_to_bool(
            os.environ.get("ENABLE_ADVANCED_OCR", "true")
        ),
        "enable_multilingual": str_to_bool(
            os.environ.get("ENABLE_MULTILINGUAL", "true")
        ),
        "enable_caching": str_to_bool(os.environ.get("ENABLE_CACHING", "true")),
    }


# Auto-setup when module is imported
if __name__ != "__main__":
    setup_secure_environment()
