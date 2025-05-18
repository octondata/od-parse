"""
Parser module for extracting content from PDF files.
"""

from od_parse.parser.pdf_parser import (
    extract_forms,
    extract_images,
    extract_tables,
    extract_text,
    parse_pdf,
)

__all__ = [
    "parse_pdf",
    "extract_text",
    "extract_images",
    "extract_tables",
    "extract_forms",
]
