"""
od-parse: AI-powered library for parsing complex PDFs with intelligent optimization.

Features:
- LLM-powered document understanding (OpenAI, Anthropic, Google)
- Intelligent document analysis with AI agents
- Smart caching (10-100x faster for repeated documents)
- Parallel processing (3-5x faster)
- Memory optimization (60-70% less memory)
- Adaptive processing strategies
- Resource-aware execution
"""

from __future__ import annotations

from typing import Any, Optional

from od_parse.converter import convert_to_markdown

# Import the LLM-enhanced parse_pdf from main module (NOT the basic parser)
from od_parse.main import parse_pdf

# Also expose the basic parser for users who don't need LLM
from od_parse.parser import parse_pdf as parse_pdf_basic

# Optional: Excel processing with DuckDB
EXCEL_AVAILABLE: bool = False

try:
    from od_parse.excel import (
        ExcelProcessor,
        excel_to_json,
        excel_to_markdown,
        parse_excel,
    )

    EXCEL_AVAILABLE = True
except ImportError:
    ExcelProcessor = None  # type: ignore[assignment, misc]
    parse_excel = None  # type: ignore[assignment]
    excel_to_json = None  # type: ignore[assignment]
    excel_to_markdown = None  # type: ignore[assignment]

# Optional: Agentic AI features (v0.2.0+)
AGENTIC_AVAILABLE: bool = False

try:
    from od_parse.agents import (
        CacheAgent,
        ParsingAgent,
        ProcessingStrategy,
        ResourceAgent,
    )
    from od_parse.parser.optimized_parser import (
        OptimizedPDFParser,
        parse_pdf_optimized,
    )

    AGENTIC_AVAILABLE = True
except ImportError:
    OptimizedPDFParser = None  # type: ignore[assignment, misc]
    parse_pdf_optimized = None  # type: ignore[assignment]
    ParsingAgent = None  # type: ignore[assignment, misc]
    CacheAgent = None  # type: ignore[assignment, misc]
    ResourceAgent = None  # type: ignore[assignment, misc]
    ProcessingStrategy = None  # type: ignore[assignment, misc]

__version__ = "0.2.0"

__all__ = [
    # Core functions (LLM-enhanced)
    "parse_pdf",  # Full LLM-powered parsing with require_llm option
    "parse_pdf_basic",  # Basic parsing without LLM (faster, no API keys needed)
    "convert_to_markdown",
    # Excel processing (DuckDB)
    "ExcelProcessor",
    "parse_excel",
    "excel_to_json",
    "excel_to_markdown",
    "EXCEL_AVAILABLE",
    # Agentic AI features (v0.2.0+)
    "OptimizedPDFParser",
    "parse_pdf_optimized",
    "ParsingAgent",
    "CacheAgent",
    "ResourceAgent",
    "ProcessingStrategy",
    "AGENTIC_AVAILABLE",
    # Version
    "__version__",
]
