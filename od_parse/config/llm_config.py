"""
LLM-First Configuration System for od-parse.

This module provides LLM-centric configuration for advanced document parsing
using state-of-the-art language models for complex PDF understanding.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers for document parsing."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    LOCAL = "local"


class DocumentComplexity(Enum):
    """Document complexity levels for LLM processing."""

    SIMPLE = "simple"  # Basic text extraction
    MODERATE = "moderate"  # Tables and forms
    COMPLEX = "complex"  # Multi-column, handwriting, complex layouts
    EXPERT = "expert"  # Scientific papers, legal documents, technical drawings


@dataclass
class LLMModelConfig:
    """Configuration for specific LLM models."""

    provider: LLMProvider
    model_name: str
    api_key_env: str
    max_tokens: int = 4096
    temperature: float = 0.1
    supports_vision: bool = False
    cost_per_1k_tokens: float = 0.0
    context_window: int = 4096
    recommended_complexity: List[DocumentComplexity] = None


class LLMConfig:
    """
    LLM-First configuration manager for od-parse.

    This class manages LLM providers, models, and parsing strategies
    for complex document understanding.
    """

    def __init__(self):
        """Initialize LLM-first configuration."""
        self.logger = get_logger(__name__)
        self.models = self._initialize_models()
        self.default_provider = None
        self.fallback_providers = []
        self.parsing_strategies = self._initialize_strategies()
        self._validate_api_keys()

    def _initialize_models(self) -> Dict[str, LLMModelConfig]:
        """Initialize supported LLM models with their configurations."""
        return {
            # OpenAI Models
            "gpt-4o": LLMModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.03,
                context_window=128000,
                recommended_complexity=[
                    DocumentComplexity.COMPLEX,
                    DocumentComplexity.EXPERT,
                ],
            ),
            "gpt-4o-mini": LLMModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o-mini",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.0015,
                context_window=128000,
                recommended_complexity=[
                    DocumentComplexity.SIMPLE,
                    DocumentComplexity.MODERATE,
                ],
            ),
            "gpt-4-turbo": LLMModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4-turbo",
                api_key_env="OPENAI_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.01,
                context_window=128000,
                recommended_complexity=[
                    DocumentComplexity.COMPLEX,
                    DocumentComplexity.EXPERT,
                ],
            ),
            # Anthropic Models
            "claude-3-5-sonnet": LLMModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.003,
                context_window=200000,
                recommended_complexity=[
                    DocumentComplexity.COMPLEX,
                    DocumentComplexity.EXPERT,
                ],
            ),
            "claude-3-haiku": LLMModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-haiku-20240307",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.00025,
                context_window=200000,
                recommended_complexity=[
                    DocumentComplexity.SIMPLE,
                    DocumentComplexity.MODERATE,
                ],
            ),
            # Google Models
            "gemini-1.5-pro": LLMModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-1.5-pro",
                api_key_env="GOOGLE_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.0035,
                context_window=1000000,
                recommended_complexity=[
                    DocumentComplexity.COMPLEX,
                    DocumentComplexity.EXPERT,
                ],
            ),
            "gemini-1.5-flash": LLMModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-1.5-flash",
                api_key_env="GOOGLE_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.00015,
                context_window=1000000,
                recommended_complexity=[
                    DocumentComplexity.SIMPLE,
                    DocumentComplexity.MODERATE,
                ],
            ),
            # Azure OpenAI
            "azure-gpt-4o": LLMModelConfig(
                provider=LLMProvider.AZURE_OPENAI,
                model_name="gpt-4o",
                api_key_env="AZURE_OPENAI_API_KEY",
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.03,
                context_window=128000,
                recommended_complexity=[
                    DocumentComplexity.COMPLEX,
                    DocumentComplexity.EXPERT,
                ],
            ),
            # Local/Ollama Models
            "llama-3.2-vision": LLMModelConfig(
                provider=LLMProvider.OLLAMA,
                model_name="llama3.2-vision",
                api_key_env="",  # No API key needed for local
                max_tokens=4096,
                temperature=0.1,
                supports_vision=True,
                cost_per_1k_tokens=0.0,  # Free local model
                context_window=8192,
                recommended_complexity=[
                    DocumentComplexity.SIMPLE,
                    DocumentComplexity.MODERATE,
                ],
            ),
        }

    def _initialize_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize parsing strategies for different document types."""
        return {
            "tax_documents": {
                "preferred_models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"],
                "system_prompt": """You are an expert tax document analyzer. Extract all relevant tax information including:
- Personal information (names, SSNs, addresses)
- Income sources and amounts
- Deductions and credits
- Tax calculations and payments
- Form-specific fields and line items
Provide structured JSON output with high accuracy.""",
                "complexity": DocumentComplexity.COMPLEX,
                "requires_vision": True,
            },
            "financial_statements": {
                "preferred_models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"],
                "system_prompt": """You are a financial document expert. Extract and structure:
- Account information and balances
- Transaction details and dates
- Financial metrics and ratios
- Tables and numerical data
- Investment holdings and performance
Ensure numerical accuracy and proper formatting.""",
                "complexity": DocumentComplexity.COMPLEX,
                "requires_vision": True,
            },
            "legal_contracts": {
                "preferred_models": ["gpt-4o", "claude-3-5-sonnet"],
                "system_prompt": """You are a legal document specialist. Extract and organize:
- Party information and roles
- Contract terms and conditions
- Key dates and deadlines
- Financial obligations
- Legal clauses and provisions
- Signature blocks and execution details
Maintain legal accuracy and completeness.""",
                "complexity": DocumentComplexity.EXPERT,
                "requires_vision": True,
            },
            "medical_records": {
                "preferred_models": ["gpt-4o", "claude-3-5-sonnet"],
                "system_prompt": """You are a medical document expert. Extract and structure:
- Patient demographics and identifiers
- Medical history and diagnoses
- Treatment plans and medications
- Test results and measurements
- Provider information and dates
- Insurance and billing information
Ensure medical accuracy and HIPAA compliance considerations.""",
                "complexity": DocumentComplexity.EXPERT,
                "requires_vision": True,
            },
            "invoices_receipts": {
                "preferred_models": [
                    "gpt-4o-mini",
                    "claude-3-haiku",
                    "gemini-1.5-flash",
                ],
                "system_prompt": """You are a business document processor. Extract:
- Vendor/customer information
- Invoice/receipt numbers and dates
- Line items with descriptions and amounts
- Tax calculations and totals
- Payment terms and methods
Ensure accurate numerical extraction.""",
                "complexity": DocumentComplexity.MODERATE,
                "requires_vision": True,
            },
            "research_papers": {
                "preferred_models": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"],
                "system_prompt": """You are an academic document analyst. Extract and structure:
- Title, authors, and affiliations
- Abstract and keywords
- Section headings and content
- Figures, tables, and captions
- References and citations
- Mathematical formulas and equations
Maintain academic rigor and citation accuracy.""",
                "complexity": DocumentComplexity.EXPERT,
                "requires_vision": True,
            },
            "general_documents": {
                "preferred_models": [
                    "gpt-4o-mini",
                    "claude-3-haiku",
                    "gemini-1.5-flash",
                ],
                "system_prompt": """You are a general document processor. Extract and organize:
- Document structure and headings
- Key information and data points
- Tables and lists
- Important dates and numbers
- Contact information
Provide clear, structured output.""",
                "complexity": DocumentComplexity.SIMPLE,
                "requires_vision": False,
            },
        }

    def _validate_api_keys(self) -> None:
        """Validate available API keys and set default provider."""
        available_providers = []

        for model_id, config in self.models.items():
            if config.api_key_env and os.getenv(config.api_key_env):
                available_providers.append((config.provider, model_id))
                if not self.default_provider:
                    self.default_provider = model_id
                    self.logger.info(f"Set default LLM provider: {model_id}")

        if not available_providers:
            self.logger.warning(
                "No LLM API keys found. od-parse requires LLM access for advanced document parsing."
            )
            self.logger.info("Please set one of the following environment variables:")
            for model_id, config in self.models.items():
                if config.api_key_env:
                    self.logger.info(f"  {config.api_key_env} for {model_id}")
        else:
            self.logger.info(
                f"Found {len(available_providers)} available LLM providers"
            )

    def get_recommended_model(
        self,
        document_type: str = "general_documents",
        complexity: DocumentComplexity = DocumentComplexity.MODERATE,
    ) -> Optional[str]:
        """Get recommended model for document type and complexity."""
        strategy = self.parsing_strategies.get(
            document_type, self.parsing_strategies["general_documents"]
        )

        # Find first available model from preferred list
        for model_id in strategy["preferred_models"]:
            if model_id in self.models:
                config = self.models[model_id]
                if config.api_key_env and os.getenv(config.api_key_env):
                    return model_id

        # Fallback to default provider
        return self.default_provider

    def has_api_key(self, provider: LLMProvider) -> bool:
        """Check if API key is available for provider."""
        for config in self.models.values():
            if config.provider == provider and config.api_key_env:
                return bool(os.getenv(config.api_key_env))
        return False

    def get_system_prompt(self, document_type: str) -> str:
        """Get system prompt for document type."""
        strategy = self.parsing_strategies.get(
            document_type, self.parsing_strategies["general_documents"]
        )
        return strategy["system_prompt"]

    def requires_llm_key(self) -> bool:
        """Check if LLM key is required (always True for LLM-first approach)."""
        return True

    def get_available_models(self) -> List[str]:
        """Get list of available models with valid API keys."""
        available = []
        for model_id, config in self.models.items():
            if not config.api_key_env or os.getenv(config.api_key_env):
                available.append(model_id)
        return available


# Global configuration instance
_llm_config = None


def get_llm_config() -> LLMConfig:
    """Get global LLM configuration instance."""
    global _llm_config
    if _llm_config is None:
        _llm_config = LLMConfig()
    return _llm_config
