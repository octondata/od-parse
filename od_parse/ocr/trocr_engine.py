"""
TrOCR (Transformer-based OCR) Engine.

This module provides TrOCR integration for superior text recognition,
with fallback to traditional OCR methods when TrOCR is not available,
and Vision LLM fallback for difficult low-quality images.
"""

from __future__ import annotations

import base64
import io
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

# VLM availability
VLM_AVAILABLE = False
try:
    from google import genai

    VLM_AVAILABLE = True
except ImportError:
    pass


class TrOCREngine:
    """
    TrOCR-based text recognition engine with fallback support.

    This class provides advanced text recognition using Microsoft's TrOCR
    (Transformer-based OCR) models, with automatic fallback to Tesseract
    when TrOCR dependencies are not available, and Vision LLM fallback
    for difficult low-quality images.
    """

    def __init__(
        self,
        model_name: str = "microsoft/trocr-base-printed",
        device: str = "auto",
        enable_vlm_fallback: bool = True,
        vlm_confidence_threshold: float = 0.7,  # Raised from 0.3 to 0.7
        vlm_model: str = "gemini-2.0-flash",
    ):
        """
        Initialize the TrOCR engine.

        Args:
            model_name: TrOCR model to use. Options:
                - microsoft/trocr-base-printed (default)
                - microsoft/trocr-base-handwritten
                - microsoft/trocr-large-printed
                - microsoft/trocr-large-handwritten
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            enable_vlm_fallback: Whether to use Vision LLM for difficult images
            vlm_confidence_threshold: Confidence below which to trigger VLM
            vlm_model: Vision LLM model to use (e.g., 'gemini-2.0-flash')
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self._is_available = False
        self._fallback_engine = None

        # VLM fallback settings
        self.enable_vlm_fallback = enable_vlm_fallback
        self.vlm_confidence_threshold = vlm_confidence_threshold
        self.vlm_model = vlm_model
        self._vlm_client = None

        # Check if TrOCR is available and initialize
        self._initialize_trocr()

        # Initialize fallback engine if TrOCR is not available
        if not self._is_available:
            self._initialize_fallback()

    def _initialize_trocr(self) -> bool:
        """Initialize TrOCR model and processor."""
        config = get_advanced_config()

        # Auto-enable TrOCR feature if not enabled (for convenience)
        if not config.is_feature_enabled("trocr"):
            self.logger.info("TrOCR feature not enabled. Auto-enabling...")
            config.enable_feature("trocr", check_dependencies=False)

        if not config.is_feature_available("trocr"):
            self.logger.warning(
                "TrOCR dependencies not available. Install with: pip install torch transformers"
            )
            self.logger.info("Will fallback to Tesseract OCR if available")
            return False

        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.logger.info(f"Loading TrOCR model: {self.model_name} on {self.device}")

            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            self._is_available = True
            self.logger.info("TrOCR initialized successfully")
            return True

        except ImportError as e:
            self.logger.warning(f"TrOCR dependencies not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize TrOCR: {e}")
            return False

    def _initialize_fallback(self):
        """Initialize fallback OCR engine (Tesseract)."""
        try:
            import pytesseract

            self._fallback_engine = pytesseract
            self.logger.info("Initialized fallback OCR engine (Tesseract)")
        except ImportError:
            self.logger.error("Neither TrOCR nor Tesseract is available")

    def _initialize_vlm(self) -> bool:
        """Initialize Vision LLM client for fallback."""
        if not VLM_AVAILABLE:
            return False

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            self.logger.warning("GOOGLE_API_KEY not set, VLM fallback disabled")
            return False

        try:
            self._vlm_client = genai.Client(api_key=api_key)
            self.logger.info(f"Initialized VLM fallback with model: {self.vlm_model}")
            return True
        except Exception as e:
            self.logger.warning(f"Failed to initialize VLM: {e}")
            return False

    def extract_text(
        self, image: Union[str, Path, Image.Image, np.ndarray], **kwargs
    ) -> Dict[str, Any]:
        """
        Extract text from an image using TrOCR or fallback engine.

        Args:
            image: Input image (file path, PIL Image, or numpy array)
            **kwargs: Additional arguments for OCR

        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Convert input to PIL Image
            pil_image = self._prepare_image(image)

            # Try primary engine
            if self._is_available:
                result = self._extract_with_trocr(pil_image, **kwargs)
            else:
                result = self._extract_with_fallback(pil_image, **kwargs)

            # Check if we should try VLM fallback
            if (
                self.enable_vlm_fallback
                and result.get("confidence", 0) < self.vlm_confidence_threshold
                and VLM_AVAILABLE
            ):
                self.logger.info(
                    f"Low confidence ({result.get('confidence', 0):.2f}), "
                    f"trying VLM fallback"
                )
                vlm_result = self._extract_with_vlm(pil_image, **kwargs)
                if vlm_result and vlm_result.get("confidence", 0) > result.get(
                    "confidence", 0
                ):
                    vlm_result["traditional_result"] = result
                    return vlm_result

            return result

        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return {"text": "", "confidence": 0.0, "engine": "error", "error": str(e)}

    def _prepare_image(
        self, image: Union[str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    def _extract_with_trocr(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Extract text using TrOCR."""
        try:
            import torch

            # Process image
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values)

            # Decode generated text
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            # Calculate confidence (simplified approach)
            confidence = self._estimate_confidence(generated_text)

            return {
                "text": generated_text.strip(),
                "confidence": confidence,
                "engine": "trocr",
                "model": self.model_name,
                "device": self.device,
            }

        except Exception as e:
            self.logger.error(f"TrOCR extraction failed: {e}")
            # Fallback to traditional OCR
            return self._extract_with_fallback(image, **kwargs)

    def _extract_with_fallback(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """Extract text using fallback OCR engine (Tesseract)."""
        if self._fallback_engine is None:
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "none",
                "error": "No OCR engine available",
            }

        try:
            # Use Tesseract with custom config if provided
            config = kwargs.get("config", "--oem 1 --psm 6")

            # Extract text
            text = self._fallback_engine.image_to_string(image, config=config)

            # Get confidence data if available
            try:
                data = self._fallback_engine.image_to_data(
                    image, config=config, output_type="dict"
                )
                confidences = [int(conf) for conf in data["conf"] if int(conf) > 0]
                avg_confidence = (
                    sum(confidences) / len(confidences) if confidences else 0
                )
                confidence = avg_confidence / 100.0  # Convert to 0-1 scale
            except:
                confidence = 0.5  # Default confidence

            return {
                "text": text.strip(),
                "confidence": confidence,
                "engine": "tesseract",
                "config": config,
            }

        except Exception as e:
            self.logger.error(f"Fallback OCR failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "tesseract",
                "error": str(e),
            }

    def _extract_with_vlm(
        self, image: Image.Image, **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Extract text using Vision LLM.

        This method uses Google Gemini Vision or similar VLMs for
        high-quality OCR on difficult images.
        """
        if not VLM_AVAILABLE:
            return None

        # Initialize VLM client if needed
        if self._vlm_client is None:
            if not self._initialize_vlm():
                return None

        try:
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            # Create prompt for OCR
            prompt = """You are an OCR engine. Extract ALL text from this image exactly as it appears.

Rules:
1. Preserve the original layout and structure as much as possible
2. Include all text, numbers, and symbols
3. Maintain line breaks where they appear in the image
4. If text is unclear, make your best guess with [unclear] marker
5. Return ONLY the extracted text, no explanations

Extract the text now:"""

            # Call Gemini Vision
            response = self._vlm_client.models.generate_content(
                model=self.vlm_model,
                contents=[
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/png",
                                    "data": base64.b64encode(img_bytes).decode("utf-8"),
                                }
                            },
                        ]
                    }
                ],
            )

            extracted_text = response.text.strip()

            # Estimate confidence based on response quality
            confidence = self._estimate_vlm_confidence(extracted_text)

            return {
                "text": extracted_text,
                "confidence": confidence,
                "engine": "vlm",
                "model": self.vlm_model,
            }

        except Exception as e:
            self.logger.error(f"VLM extraction failed: {e}")
            return None

    def _estimate_vlm_confidence(self, text: str) -> float:
        """Estimate confidence for VLM-extracted text."""
        if not text or not text.strip():
            return 0.0

        score = 0.85  # Base confidence for VLM

        # Check for uncertainty markers
        if "[unclear]" in text.lower():
            unclear_count = text.lower().count("[unclear]")
            score -= 0.1 * min(unclear_count, 3)

        # Check text length (very short might indicate issues)
        if len(text.strip()) < 10:
            score *= 0.8

        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.2:
            score *= 0.9

        return max(0.1, min(score, 1.0))

    def _estimate_confidence(self, text: str) -> float:
        """
        Estimate confidence score for TrOCR output.

        This is a simplified approach. In practice, you might want to use
        the model's attention weights or other metrics.
        """
        if not text or not text.strip():
            return 0.0

        # Simple heuristics for confidence estimation
        score = 0.8  # Base confidence for TrOCR

        # Adjust based on text characteristics
        if len(text.strip()) < 3:
            score *= 0.7

        # Check for common OCR artifacts
        artifacts = ["|||", "___", "...", "???"]
        for artifact in artifacts:
            if artifact in text:
                score *= 0.6

        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.3:
            score *= 0.7

        return min(score, 1.0)

    def batch_extract_text(
        self, images: List[Union[str, Path, Image.Image, np.ndarray]], **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract text from multiple images.

        Args:
            images: List of images to process
            **kwargs: Additional arguments for OCR

        Returns:
            List of extraction results
        """
        results = []

        for i, image in enumerate(images):
            try:
                result = self.extract_text(image, **kwargs)
                result["image_index"] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process image {i}: {e}")
                results.append(
                    {
                        "text": "",
                        "confidence": 0.0,
                        "engine": "error",
                        "image_index": i,
                        "error": str(e),
                    }
                )

        return results

    def is_available(self) -> bool:
        """Check if TrOCR is available."""
        return self._is_available

    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current OCR engine."""
        return {
            "trocr_available": self._is_available,
            "model_name": self.model_name if self._is_available else None,
            "device": self.device if self._is_available else None,
            "fallback_available": self._fallback_engine is not None,
            "current_engine": "trocr" if self._is_available else "tesseract",
            "vlm_fallback_enabled": self.enable_vlm_fallback,
            "vlm_available": VLM_AVAILABLE,
            "vlm_model": self.vlm_model if self.enable_vlm_fallback else None,
            "vlm_confidence_threshold": self.vlm_confidence_threshold,
        }


# Convenience function for easy usage
def extract_text_with_trocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    model_name: str = "microsoft/trocr-base-printed",
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to extract text using TrOCR.

    Args:
        image: Input image
        model_name: TrOCR model to use
        **kwargs: Additional arguments

    Returns:
        Extraction result dictionary
    """
    engine = TrOCREngine(model_name=model_name)
    return engine.extract_text(image, **kwargs)
