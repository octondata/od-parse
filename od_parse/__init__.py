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

from od_parse.parser import parse_pdf
from od_parse.converter import convert_to_markdown

# Agentic AI features (v0.2.0+)
try:
    from od_parse.parser.optimized_parser import OptimizedPDFParser, parse_pdf_optimized
    from od_parse.agents import (
        ParsingAgent,
        CacheAgent,
        ResourceAgent,
        ProcessingStrategy,
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

__all__ = [
    # Core functions
    "parse_pdf",
    "convert_to_markdown",
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
