"""
High-level API for od-parse library.

This module provides a clean, intuitive interface for the most common use cases
of the od-parse library. It implements a fluent interface pattern for better
usability and includes comprehensive error handling.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from .advanced.optimized_processor import OptimizedProcessor
from .exceptions import (
    ConfigurationError,
    FileError,
    ProcessingError,
)
from .utils.error_handler import handle_errors
from .utils.logging_utils import get_logger


class ParseBuilder:
    """
    Builder class for configuring and executing document parsing.

    Examples:
        >>> from od_parse import ParseBuilder
        >>> # Simple usage
        >>> result = ParseBuilder().parse_file("document.pdf").to_markdown()
        >>>
        >>> # Advanced usage with configuration
        >>> result = (
        ...     ParseBuilder()
        ...     .with_ocr(enable_handwriting=True)
        ...     .with_table_detection()
        ...     .with_progress_callback(lambda p: print(f"Progress: {p}%"))
        ...     .parse_file("document.pdf")
        ...     .to_markdown()
        ... )
    """

    def __init__(self):
        self.logger = get_logger(__name__)
        self._config: Dict[str, Any] = {}
        self._processor: Optional[OptimizedProcessor] = None
        self._progress_callback: Optional[Callable[[float], None]] = None

    @handle_errors({ValueError: ConfigurationError})
    def with_ocr(
        self,
        enable: bool = True,
        enable_handwriting: bool = False,
        language: str = "eng",
    ) -> "ParseBuilder":
        """Configure OCR settings.

        Args:
            enable: Whether to enable OCR
            enable_handwriting: Whether to enable handwriting recognition
            language: Language code for OCR

        Returns:
            Self for method chaining

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self._config["ocr"] = {
            "enabled": enable,
            "handwriting": enable_handwriting,
            "language": language,
        }
        return self

    @handle_errors({ValueError: ConfigurationError})
    def with_table_detection(
        self, enable: bool = True, min_confidence: float = 0.8
    ) -> "ParseBuilder":
        """Enable and configure table detection.

        Args:
            enable: Whether to enable table detection
            min_confidence: Minimum confidence score for table detection

        Returns:
            Self for method chaining

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self._config["tables"] = {"enabled": enable, "min_confidence": min_confidence}
        return self

    def with_progress_callback(
        self, callback: Callable[[float], None]
    ) -> "ParseBuilder":
        """Add progress callback for monitoring long operations."""
        self._progress_callback = callback
        return self

    @handle_errors(
        {
            FileNotFoundError: FileError,
            ValueError: ConfigurationError,
            Exception: ProcessingError,
        }
    )
    def parse_file(
        self, file_path: Union[str, Path], batch_size: int = 5
    ) -> "DocumentProcessor":
        """Start parsing a document file.

        Args:
            file_path: Path to the document file
            batch_size: Number of pages to process at once

        Returns:
            DocumentProcessor instance

        Raises:
            FileError: If file doesn't exist or is inaccessible
            ConfigurationError: If configuration is invalid
            ProcessingError: If initialization fails
        """
        self._config["batch_size"] = batch_size
        self._processor = OptimizedProcessor(self._config)
        return DocumentProcessor(self._processor, file_path, self._progress_callback)


class DocumentProcessor:
    """
    Handles document processing after configuration.

    This class provides methods for executing the parse operation and
    converting the results to different formats.
    """

    def __init__(
        self,
        processor: OptimizedProcessor,
        file_path: Union[str, Path],
        progress_callback: Optional[Callable[[float], None]] = None,
    ):
        self.processor = processor
        self.file_path = Path(file_path)
        self.progress_callback = progress_callback
        self._result: Optional[Dict[str, Any]] = None

    @handle_errors(
        {
            FileNotFoundError: FileError,
            OSError: ProcessingError,
            MemoryError: ProcessingError,
            Exception: ProcessingError,
        }
    )
    async def process_async(self) -> Dict[str, Any]:
        """Process the document asynchronously.

        Returns:
            Dictionary containing processed results

        Raises:
            FileError: If file becomes inaccessible
            ProcessingError: If processing fails
            OCRError: If OCR fails
            TableExtractionError: If table extraction fails
        """
        if self._result is None:
            with ThreadPoolExecutor() as executor:
                self._result = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    self.processor.process_document,
                    self.file_path,
                    self.progress_callback,
                )
        return self._result

    @handle_errors(
        {
            FileNotFoundError: FileError,
            OSError: ProcessingError,
            MemoryError: ProcessingError,
            Exception: ProcessingError,
        }
    )
    def process(self) -> Dict[str, Any]:
        """Process the document synchronously.

        Returns:
            Dictionary containing processed results

        Raises:
            FileError: If file becomes inaccessible
            ProcessingError: If processing fails
            OCRError: If OCR fails
            TableExtractionError: If table extraction fails
        """
        if self._result is None:
            self._result = self.processor.process_document(
                self.file_path, self.progress_callback
            )
        return self._result

    def to_markdown(self, **kwargs) -> str:
        """Convert processed document to markdown."""
        result = self.process()
        return self.processor.to_markdown(result, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Get the raw processing results."""
        return self.process()

    def to_text(self) -> str:
        """Convert processed document to plain text."""
        result = self.process()
        return self.processor.to_text(result)


# Convenience function for quick parsing
@handle_errors(
    {
        FileNotFoundError: FileError,
        ValueError: ConfigurationError,
        Exception: ProcessingError,
    }
)
def parse_document(file_path: Union[str, Path]) -> str:
    """
    Quick parse a document to markdown with default settings.

    Args:
        file_path: Path to the document file

    Returns:
        Markdown string of the document contents

    Raises:
        FileError: If file doesn't exist or is inaccessible
        ConfigurationError: If default configuration is invalid
        ProcessingError: If processing fails
        OCRError: If OCR fails
        TableExtractionError: If table extraction fails

    Examples:
        >>> from od_parse import parse_document
        >>> markdown = parse_document("document.pdf")
    """
    return ParseBuilder().parse_file(file_path).to_markdown()
