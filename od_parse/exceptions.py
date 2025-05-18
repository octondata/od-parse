"""
Custom exceptions for od-parse library.

This module defines specific exceptions that can occur during document processing,
allowing users to handle different error cases appropriately.
"""


class ODParseError(Exception):
    """Base exception for all od-parse errors."""

    pass


class FileError(ODParseError):
    """Errors related to file operations."""

    pass


class FileNotFoundError(FileError):
    """File does not exist or is not accessible."""

    pass


class FileTypeError(FileError):
    """File type is not supported."""

    pass


class FileCorruptedError(FileError):
    """File is corrupted or invalid."""

    pass


class ProcessingError(ODParseError):
    """Errors during document processing."""

    pass


class OCRError(ProcessingError):
    """Errors during OCR processing."""

    pass


class TableExtractionError(ProcessingError):
    """Errors during table extraction."""

    pass


class MemoryError(ProcessingError):
    """Not enough memory to process the document."""

    pass


class ConfigurationError(ODParseError):
    """Invalid configuration provided."""

    pass


class ResourceError(ODParseError):
    """Required resource (e.g., OCR engine) not available."""

    pass


class ConversionError(ODParseError):
    """Error during format conversion."""

    pass
