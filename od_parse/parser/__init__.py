"""Parser module for extracting content from PDF files."""
from __future__ import annotations

from od_parse.parser.pdf_parser import (
    extract_forms,
    extract_images,
    extract_tables,
    extract_text,
    parse_pdf,
)

# Alias for backward compatibility
core_parse_pdf = parse_pdf

__all__ = [
    "parse_pdf",
    "core_parse_pdf",
    "extract_text",
    "extract_images",
    "extract_tables",
    "extract_forms",
]
