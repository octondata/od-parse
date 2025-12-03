"""OCR module for extracting text from images and handwritten content."""

from __future__ import annotations

from od_parse.ocr.handwritten import extract_handwritten_content

# Optional: TrOCR engine for enhanced OCR
try:
    from od_parse.ocr.trocr_engine import TrOCREngine, extract_text_with_trocr

    __all__ = ["extract_handwritten_content", "TrOCREngine", "extract_text_with_trocr"]
except ImportError:
    __all__ = ["extract_handwritten_content"]
