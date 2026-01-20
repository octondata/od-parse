"""OCR module for extracting text from images and handwritten content."""

from __future__ import annotations

from od_parse.ocr.handwritten import extract_handwritten_content

# Image enhancement
from od_parse.ocr.image_enhancer import (
    EnhancementConfig,
    ImageEnhancer,
    ImageQuality,
    QualityScore,
    assess_image_quality,
    binarize_for_ocr,
    enhance_for_ocr,
)

__all__ = [
    # Handwritten content extraction
    "extract_handwritten_content",
    # Image enhancement
    "ImageEnhancer",
    "EnhancementConfig",
    "ImageQuality",
    "QualityScore",
    "assess_image_quality",
    "enhance_for_ocr",
    "binarize_for_ocr",
]

# Optional: TrOCR engine for enhanced OCR
try:
    from od_parse.ocr.trocr_engine import TrOCREngine, extract_text_with_trocr

    __all__.extend(["TrOCREngine", "extract_text_with_trocr"])
except ImportError:
    pass

# Optional: Multi-engine OCR support
try:
    from od_parse.ocr.multi_engine import (
        DocTREngine,
        EasyOCREngine,
        MultiEngineOCR,
        OCRResult,
        TesseractEngine,
        extract_text_multi_engine,
        get_available_ocr_engines,
    )

    __all__.extend(
        [
            "MultiEngineOCR",
            "OCRResult",
            "EasyOCREngine",
            "DocTREngine",
            "TesseractEngine",
            "extract_text_multi_engine",
            "get_available_ocr_engines",
        ]
    )
except ImportError:
    pass
