"""
TrOCR (Transformer-based OCR) Engine

This module provides TrOCR integration for superior text recognition,
with fallback to traditional OCR methods when TrOCR is not available.
"""

import os
import warnings
from typing import Union, Optional, Dict, Any, List
from pathlib import Path

import numpy as np
from PIL import Image

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TrOCREngine:
    """
    TrOCR-based text recognition engine with fallback support.
    
    This class provides advanced text recognition using Microsoft's TrOCR
    (Transformer-based OCR) models, with automatic fallback to Tesseract
    when TrOCR dependencies are not available.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-printed", device: str = "auto"):
        """
        Initialize the TrOCR engine.
        
        Args:
            model_name: TrOCR model to use. Options:
                - microsoft/trocr-base-printed (default)
                - microsoft/trocr-base-handwritten
                - microsoft/trocr-large-printed
                - microsoft/trocr-large-handwritten
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.device = device
        self.processor = None
        self.model = None
        self._is_available = False
        self._fallback_engine = None
        
        # Check if TrOCR is available and initialize
        self._initialize_trocr()
        
        # Initialize fallback engine if TrOCR is not available
        if not self._is_available:
            self._initialize_fallback()
    
    def _initialize_trocr(self) -> bool:
        """Initialize TrOCR model and processor."""
        config = get_advanced_config()
        
        if not config.is_feature_enabled("trocr"):
            self.logger.info("TrOCR feature is disabled. Use config.enable_feature('trocr') to enable.")
            return False
        
        if not config.is_feature_available("trocr"):
            self.logger.warning("TrOCR dependencies not available. Install with: pip install od-parse[trocr]")
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
    
    def extract_text(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        **kwargs
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
            
            if self._is_available:
                return self._extract_with_trocr(pil_image, **kwargs)
            else:
                return self._extract_with_fallback(pil_image, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error extracting text: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "error",
                "error": str(e)
            }
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
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
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Calculate confidence (simplified approach)
            confidence = self._estimate_confidence(generated_text)
            
            return {
                "text": generated_text.strip(),
                "confidence": confidence,
                "engine": "trocr",
                "model": self.model_name,
                "device": self.device
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
                "error": "No OCR engine available"
            }
        
        try:
            # Use Tesseract with custom config if provided
            config = kwargs.get('config', '--oem 1 --psm 6')
            
            # Extract text
            text = self._fallback_engine.image_to_string(image, config=config)
            
            # Get confidence data if available
            try:
                data = self._fallback_engine.image_to_data(image, config=config, output_type='dict')
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                confidence = avg_confidence / 100.0  # Convert to 0-1 scale
            except:
                confidence = 0.5  # Default confidence
            
            return {
                "text": text.strip(),
                "confidence": confidence,
                "engine": "tesseract",
                "config": config
            }
            
        except Exception as e:
            self.logger.error(f"Fallback OCR failed: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "tesseract",
                "error": str(e)
            }
    
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
        artifacts = ['|||', '___', '...', '???']
        for artifact in artifacts:
            if artifact in text:
                score *= 0.6
        
        # Check for reasonable character distribution
        alpha_ratio = sum(c.isalpha() for c in text) / len(text) if text else 0
        if alpha_ratio < 0.3:
            score *= 0.7
        
        return min(score, 1.0)
    
    def batch_extract_text(
        self, 
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        **kwargs
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
                result['image_index'] = i
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process image {i}: {e}")
                results.append({
                    "text": "",
                    "confidence": 0.0,
                    "engine": "error",
                    "image_index": i,
                    "error": str(e)
                })
        
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
            "current_engine": "trocr" if self._is_available else "tesseract"
        }


# Convenience function for easy usage
def extract_text_with_trocr(
    image: Union[str, Path, Image.Image, np.ndarray],
    model_name: str = "microsoft/trocr-base-printed",
    **kwargs
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
