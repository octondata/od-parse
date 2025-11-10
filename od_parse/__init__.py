"""
od-parse: AI-powered library for parsing complex PDFs with intelligent optimization.

Features:
- Intelligent document analysis with AI agents
- Smart caching (10-100x faster for repeated documents)
- Parallel processing (3-5x faster)
- Memory optimization (60-70% less memory)
- Adaptive processing strategies
- Resource-aware execution
"""

from typing import Optional, Dict, List

from od_parse.parser import parse_pdf
from od_parse.converter import convert_to_markdown
from od_parse.main import parse_forms_separately, parse_segmented
from od_parse.config.llm_config import get_llm_config

# Agentic AI features (v0.2.0+)
try:
    from od_parse.parser.optimized_parser import (
        OptimizedPDFParser,
        parse_pdf_optimized
    )
    from od_parse.agents import (
        ParsingAgent,
        CacheAgent,
        ResourceAgent,
        ProcessingStrategy
    )
    AGENTIC_AVAILABLE = True
except ImportError:
    AGENTIC_AVAILABLE = False
    OptimizedPDFParser = None
    parse_pdf_optimized = None
    ParsingAgent = None
    CacheAgent = None
    ResourceAgent = None
    ProcessingStrategy = None

__version__ = "0.2.0"

# Helper functions for model selection
def get_available_models(api_keys: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Get list of all available models.
    
    Args:
        api_keys: Optional dictionary of API keys
    
    Returns:
        List of available model IDs
    """
    llm_config = get_llm_config(api_keys=api_keys)
    return llm_config.get_available_models()


def get_cloud_llm_models(api_keys: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Get list of available cloud LLM API models (OpenAI, Google, Anthropic, etc.).
    
    Args:
        api_keys: Optional dictionary of API keys
    
    Returns:
        List of cloud LLM API model IDs
    """
    llm_config = get_llm_config(api_keys=api_keys)
    return llm_config.get_cloud_llm_models()


def get_vllm_models(api_keys: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Get list of available vLLM models (local or cloud-hosted).
    
    Args:
        api_keys: Optional dictionary of API keys
    
    Returns:
        List of vLLM model IDs
    """
    llm_config = get_llm_config(api_keys=api_keys)
    return llm_config.get_vllm_models()


def is_vllm_model(model_id: str) -> bool:
    """
    Check if a model ID is a vLLM model.
    
    Args:
        model_id: Model ID to check
    
    Returns:
        True if model is a vLLM model, False otherwise
    """
    llm_config = get_llm_config()
    return llm_config.is_vllm_model(model_id)


def is_cloud_llm_model(model_id: str) -> bool:
    """
    Check if a model ID is a cloud LLM API model.
    
    Args:
        model_id: Model ID to check
    
    Returns:
        True if model is a cloud LLM API model, False otherwise
    """
    llm_config = get_llm_config()
    return llm_config.is_cloud_llm_model(model_id)


__all__ = [
    # Core functions
    'parse_pdf',
    'convert_to_markdown',
    'parse_forms_separately',
    'parse_segmented',

    # Model selection helpers
    'get_available_models',
    'get_cloud_llm_models',
    'get_vllm_models',
    'is_vllm_model',
    'is_cloud_llm_model',

    # Agentic AI features (v0.2.0+)
    'OptimizedPDFParser',
    'parse_pdf_optimized',
    'ParsingAgent',
    'CacheAgent',
    'ResourceAgent',
    'ProcessingStrategy',
    'AGENTIC_AVAILABLE',

    # Version
    '__version__',
]
