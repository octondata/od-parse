"""
Parser module for extracting content from PDF files.
"""

from od_parse.parser.pdf_parser import parse_pdf, extract_text, extract_images, extract_tables, extract_forms

# Alias for backward compatibility
core_parse_pdf = parse_pdf

__all__ = ["parse_pdf", "core_parse_pdf", "extract_text", "extract_images", "extract_tables", "extract_forms"]
