"""
Multi-Engine OCR Module.

This module provides a unified interface for multiple OCR engines:
- EasyOCR: Scene text, multi-language support
- docTR: Document layout aware OCR
- TrOCR: Transformer-based OCR (handwritten)
- Tesseract: Traditional OCR (fast, printed text)

Includes ensemble voting for improved accuracy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from od_parse.ocr.image_enhancer import (
    EnhancementConfig,
    ImageEnhancer,
    ImageQuality,
    QualityScore,
    assess_image_quality,
)
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


# Engine availability flags
EASYOCR_AVAILABLE = False
DOCTR_AVAILABLE = False
TESSERACT_AVAILABLE = False
TROCR_AVAILABLE = False

try:
    import easyocr

    EASYOCR_AVAILABLE = True
except ImportError:
    pass

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor

    DOCTR_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class OCRResult:
    """Result from an OCR engine."""

    text: str
    confidence: float
    engine: str
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class OCREngine(ABC):
    """Abstract base class for OCR engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine name."""
        pass

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the engine is available."""
        pass

    @abstractmethod
    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs,
    ) -> OCRResult:
        """Extract text from an image."""
        pass


class EasyOCREngine(OCREngine):
    """
    EasyOCR engine wrapper.

    Best for: Scene text, receipts, photos, multi-language documents.
    """

    def __init__(
        self,
        languages: List[str] = None,
        gpu: bool = True,
    ):
        """
        Initialize EasyOCR engine.

        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "easyocr"

    @property
    def is_available(self) -> bool:
        return EASYOCR_AVAILABLE

    def _get_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._reader is None and EASYOCR_AVAILABLE:
            self.logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        return self._reader

    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs,
    ) -> OCRResult:
        """Extract text using EasyOCR."""
        if not EASYOCR_AVAILABLE:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                error="EasyOCR not installed. Run: pip install easyocr",
            )

        try:
            reader = self._get_reader()

            # Convert image to format EasyOCR expects
            if isinstance(image, (str, Path)):
                img_input = str(image)
            elif isinstance(image, Image.Image):
                img_input = np.array(image)
            else:
                img_input = image

            # Extract text
            results = reader.readtext(img_input, **kwargs)

            # Process results
            all_text = []
            bounding_boxes = []
            confidences = []

            for bbox, text, conf in results:
                all_text.append(text)
                confidences.append(conf)
                bounding_boxes.append(
                    {
                        "text": text,
                        "confidence": conf,
                        "bbox": bbox,
                    }
                )

            combined_text = " ".join(all_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine=self.name,
                bounding_boxes=bounding_boxes,
                metadata={"num_detections": len(results)},
            )

        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                error=str(e),
            )


class DocTREngine(OCREngine):
    """
    docTR engine wrapper.

    Best for: Structured documents, forms, documents with complex layouts.
    """

    def __init__(
        self,
        det_arch: str = "db_resnet50",
        reco_arch: str = "crnn_vgg16_bn",
        pretrained: bool = True,
    ):
        """
        Initialize docTR engine.

        Args:
            det_arch: Detection architecture
            reco_arch: Recognition architecture
            pretrained: Whether to use pretrained models
        """
        self.det_arch = det_arch
        self.reco_arch = reco_arch
        self.pretrained = pretrained
        self._predictor = None
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "doctr"

    @property
    def is_available(self) -> bool:
        return DOCTR_AVAILABLE

    def _get_predictor(self):
        """Lazy initialization of docTR predictor."""
        if self._predictor is None and DOCTR_AVAILABLE:
            self.logger.info(
                f"Initializing docTR with det={self.det_arch}, reco={self.reco_arch}"
            )
            self._predictor = ocr_predictor(
                det_arch=self.det_arch,
                reco_arch=self.reco_arch,
                pretrained=self.pretrained,
            )
        return self._predictor

    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs,
    ) -> OCRResult:
        """Extract text using docTR."""
        if not DOCTR_AVAILABLE:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                error="docTR not installed. Run: pip install python-doctr[torch]",
            )

        try:
            predictor = self._get_predictor()

            # Load document
            if isinstance(image, (str, Path)):
                doc = DocumentFile.from_images(str(image))
            elif isinstance(image, Image.Image):
                # Save temporarily for docTR
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    image.save(f.name)
                    doc = DocumentFile.from_images(f.name)
            else:
                # numpy array
                import tempfile

                import cv2

                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    cv2.imwrite(f.name, image)
                    doc = DocumentFile.from_images(f.name)

            # Run prediction
            result = predictor(doc)

            # Extract text and metadata
            all_text = []
            bounding_boxes = []
            confidences = []

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        line_conf = sum(w.confidence for w in line.words) / len(
                            line.words
                        )
                        all_text.append(line_text)
                        confidences.append(line_conf)
                        bounding_boxes.append(
                            {
                                "text": line_text,
                                "confidence": line_conf,
                                "geometry": line.geometry,
                            }
                        )

            combined_text = "\n".join(all_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                engine=self.name,
                bounding_boxes=bounding_boxes,
                metadata={
                    "num_pages": len(result.pages),
                    "num_lines": len(bounding_boxes),
                },
            )

        except Exception as e:
            self.logger.error(f"docTR extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                error=str(e),
            )


class TesseractEngine(OCREngine):
    """
    Tesseract OCR engine wrapper.

    Best for: Clean printed documents, fast processing.
    """

    def __init__(
        self,
        lang: str = "eng",
        config: str = "--oem 3 --psm 6",
    ):
        """
        Initialize Tesseract engine.

        Args:
            lang: Tesseract language code
            config: Tesseract configuration string
        """
        self.lang = lang
        self.config = config
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "tesseract"

    @property
    def is_available(self) -> bool:
        return TESSERACT_AVAILABLE

    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs,
    ) -> OCRResult:
        """Extract text using Tesseract."""
        if not TESSERACT_AVAILABLE:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                error="pytesseract not installed. Run: pip install pytesseract",
            )

        try:
            # Convert to PIL Image
            if isinstance(image, (str, Path)):
                pil_image = Image.open(image)
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            config = kwargs.get("config", self.config)
            lang = kwargs.get("lang", self.lang)

            # Extract text
            text = pytesseract.image_to_string(pil_image, lang=lang, config=config)

            # Get confidence data
            try:
                data = pytesseract.image_to_data(
                    pil_image,
                    lang=lang,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                avg_confidence = (
                    sum(confidences) / len(confidences) / 100.0 if confidences else 0.5
                )

                # Build bounding boxes
                bounding_boxes = []
                for i, word in enumerate(data["text"]):
                    if word.strip() and int(data["conf"][i]) > 0:
                        bounding_boxes.append(
                            {
                                "text": word,
                                "confidence": int(data["conf"][i]) / 100.0,
                                "bbox": [
                                    data["left"][i],
                                    data["top"][i],
                                    data["width"][i],
                                    data["height"][i],
                                ],
                            }
                        )
            except Exception:
                avg_confidence = 0.5
                bounding_boxes = []

            return OCRResult(
                text=text.strip(),
                confidence=avg_confidence,
                engine=self.name,
                bounding_boxes=bounding_boxes,
                metadata={"lang": lang, "config": config},
            )

        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                engine=self.name,
                error=str(e),
            )


class MultiEngineOCR:
    """
    Multi-engine OCR processor with intelligent routing and ensemble support.

    Routes images to appropriate OCR engine based on quality assessment,
    with optional ensemble voting for improved accuracy.
    """

    def __init__(
        self,
        engines: Optional[List[str]] = None,
        enable_enhancement: bool = True,
        enhancement_config: Optional[EnhancementConfig] = None,
        vlm_fallback: bool = True,
        vlm_confidence_threshold: float = 0.7,  # Raised from 0.3 to 0.7
    ):
        """
        Initialize multi-engine OCR.

        Args:
            engines: List of engines to use (default: all available)
            enable_enhancement: Whether to apply image enhancement
            enhancement_config: Configuration for image enhancement
            vlm_fallback: Whether to use Vision LLM for low-quality images
            vlm_confidence_threshold: Confidence below which to trigger VLM
        """
        self.logger = get_logger(__name__)
        self.enable_enhancement = enable_enhancement
        self.enhancer = (
            ImageEnhancer(enhancement_config) if enable_enhancement else None
        )
        self.vlm_fallback = vlm_fallback
        self.vlm_confidence_threshold = vlm_confidence_threshold

        # Initialize requested engines
        self._engines: Dict[str, OCREngine] = {}
        available_engines = engines or ["tesseract", "easyocr", "doctr"]

        for engine_name in available_engines:
            engine = self._create_engine(engine_name)
            if engine and engine.is_available:
                self._engines[engine_name] = engine
                self.logger.info(f"Initialized OCR engine: {engine_name}")
            else:
                self.logger.warning(f"OCR engine not available: {engine_name}")

        if not self._engines:
            self.logger.warning("No OCR engines available!")

    def _create_engine(self, name: str) -> Optional[OCREngine]:
        """Create an OCR engine by name."""
        if name == "tesseract":
            return TesseractEngine()
        elif name == "easyocr":
            return EasyOCREngine()
        elif name == "doctr":
            return DocTREngine()
        else:
            self.logger.warning(f"Unknown engine: {name}")
            return None

    @property
    def available_engines(self) -> List[str]:
        """List of available engine names."""
        return list(self._engines.keys())

    def extract_text(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        engine: Optional[str] = None,
        use_ensemble: bool = False,
        **kwargs,
    ) -> OCRResult:
        """
        Extract text from an image.

        Args:
            image: Input image
            engine: Specific engine to use (auto-selects if None)
            use_ensemble: Use ensemble voting from multiple engines
            **kwargs: Additional arguments passed to engines

        Returns:
            OCRResult with extracted text
        """
        # Assess image quality
        quality = assess_image_quality(image)
        self.logger.info(
            f"Image quality: {quality.overall_score:.2f} ({quality.quality_level.value})"
        )

        # Apply enhancement if needed
        enhanced_image = image
        enhancement_metadata = {}
        if self.enable_enhancement and quality.quality_level != ImageQuality.HIGH:
            enhanced_image, enhancement_metadata = self.enhancer.enhance(image, quality)

        # Select processing strategy
        if use_ensemble:
            return self._ensemble_extract(enhanced_image, quality, **kwargs)
        elif engine:
            return self._single_engine_extract(enhanced_image, engine, **kwargs)
        else:
            return self._smart_extract(enhanced_image, quality, **kwargs)

    def _single_engine_extract(
        self,
        image: Union[np.ndarray, Image.Image],
        engine_name: str,
        **kwargs,
    ) -> OCRResult:
        """Extract using a specific engine."""
        if engine_name not in self._engines:
            return OCRResult(
                text="",
                confidence=0.0,
                engine=engine_name,
                error=f"Engine '{engine_name}' not available",
            )

        return self._engines[engine_name].extract_text(image, **kwargs)

    def _smart_extract(
        self,
        image: Union[np.ndarray, Image.Image],
        quality: QualityScore,
        **kwargs,
    ) -> OCRResult:
        """
        Intelligently select and use the best engine based on image quality.
        """
        # Route based on quality
        if quality.quality_level == ImageQuality.HIGH:
            # Fast path: use Tesseract for high-quality images
            preferred_engines = ["tesseract", "doctr", "easyocr"]
        elif quality.quality_level == ImageQuality.MEDIUM:
            # Enhanced path: prefer docTR or EasyOCR
            preferred_engines = ["doctr", "easyocr", "tesseract"]
        else:
            # Low quality: try multiple engines, consider VLM fallback
            preferred_engines = ["easyocr", "doctr", "tesseract"]

        # Try engines in order
        for engine_name in preferred_engines:
            if engine_name in self._engines:
                result = self._engines[engine_name].extract_text(image, **kwargs)
                if result.confidence >= self.vlm_confidence_threshold:
                    return result
                self.logger.info(
                    f"{engine_name} confidence ({result.confidence:.2f}) below threshold"
                )

        # If all engines have low confidence, try VLM fallback
        if self.vlm_fallback and quality.quality_level == ImageQuality.LOW:
            vlm_result = self._vlm_extract(image, **kwargs)
            if vlm_result and vlm_result.confidence > 0:
                return vlm_result

        # Return best result from traditional engines
        return (
            result
            if "result" in locals()
            else OCRResult(
                text="",
                confidence=0.0,
                engine="none",
                error="No OCR engines available",
            )
        )

    def _ensemble_extract(
        self,
        image: Union[np.ndarray, Image.Image],
        quality: QualityScore,
        **kwargs,
    ) -> OCRResult:
        """
        Extract text using ensemble voting from multiple engines.
        """
        results = []

        for engine_name, engine in self._engines.items():
            result = engine.extract_text(image, **kwargs)
            if result.text:
                results.append(result)
                self.logger.info(
                    f"{engine_name}: confidence={result.confidence:.2f}, "
                    f"length={len(result.text)}"
                )

        if not results:
            return OCRResult(
                text="",
                confidence=0.0,
                engine="ensemble",
                error="All engines failed",
            )

        # Select best result based on confidence-weighted voting
        best_result = max(results, key=lambda r: r.confidence)

        # If confidences are close, prefer longer text (usually more complete)
        for result in results:
            if (
                result.confidence >= best_result.confidence * 0.9
                and len(result.text) > len(best_result.text) * 1.2
            ):
                best_result = result

        return OCRResult(
            text=best_result.text,
            confidence=best_result.confidence,
            engine=f"ensemble({best_result.engine})",
            bounding_boxes=best_result.bounding_boxes,
            metadata={
                "ensemble_results": [
                    {
                        "engine": r.engine,
                        "confidence": r.confidence,
                        "length": len(r.text),
                    }
                    for r in results
                ],
                "selected_engine": best_result.engine,
            },
        )

    def _vlm_extract(
        self,
        image: Union[np.ndarray, Image.Image],
        **kwargs,
    ) -> Optional[OCRResult]:
        """
        Extract text using Vision LLM as fallback.

        Uses the existing LLM infrastructure for difficult images.
        """
        try:
            from od_parse.llm.document_processor import LLMDocumentProcessor

            self.logger.info("Using Vision LLM fallback for OCR")

            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image

            # Use document processor with vision model
            processor = LLMDocumentProcessor()

            # Prepare prompt for OCR
            result = processor.process_document(
                parsed_data={"text": "", "tables": [], "forms": []},
                document_images=[pil_image],
            )

            if result.get("llm_analysis", {}).get("processing_success"):
                extracted = result["llm_analysis"].get("extracted_data", {})
                text = extracted.get("extracted_text", "")
                if isinstance(text, str):
                    return OCRResult(
                        text=text,
                        confidence=0.8,  # Reasonable confidence for VLM
                        engine="vlm",
                        metadata=result.get("llm_analysis", {}).get("model_info", {}),
                    )

            return None

        except ImportError:
            self.logger.warning("LLM document processor not available for VLM fallback")
            return None
        except Exception as e:
            self.logger.error(f"VLM extraction failed: {e}")
            return None


# Convenience functions
def extract_text_multi_engine(
    image: Union[str, Path, Image.Image, np.ndarray],
    engines: Optional[List[str]] = None,
    use_ensemble: bool = False,
    **kwargs,
) -> OCRResult:
    """
    Convenience function for multi-engine OCR.

    Args:
        image: Input image
        engines: List of engines to use
        use_ensemble: Whether to use ensemble voting
        **kwargs: Additional arguments

    Returns:
        OCRResult with extracted text
    """
    processor = MultiEngineOCR(engines=engines)
    return processor.extract_text(image, use_ensemble=use_ensemble, **kwargs)


def get_available_ocr_engines() -> Dict[str, bool]:
    """Get availability status of all OCR engines."""
    return {
        "easyocr": EASYOCR_AVAILABLE,
        "doctr": DOCTR_AVAILABLE,
        "tesseract": TESSERACT_AVAILABLE,
        "trocr": TROCR_AVAILABLE,
    }
