"""
Multilingual processing module for od-parse.

This module provides comprehensive multilingual support for document
processing, including language detection, text processing, and translation.
"""

from od_parse.multilingual.language_processor import (
    MultilingualProcessor,
    detect_document_language,
    process_multilingual_document
)

__all__ = [
    "MultilingualProcessor",
    "detect_document_language",
    "process_multilingual_document"
]
