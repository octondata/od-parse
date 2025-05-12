"""
Transformer-based OCR module for extracting text from handwritten content.

This module implements state-of-the-art transformer models for optical character recognition,
specifically optimized for handwritten text extraction from complex documents.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import os
import logging
from pathlib import Path
import json

# Check for optional dependencies
try:
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import cv2
    from PIL import Image
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False


class TransformerOCR:
    """
    Advanced OCR for handwritten and printed text using transformer models.
    
    This class uses state-of-the-art vision-language models to extract text from
    images, with a focus on accurately recognizing handwritten content that often
    challenges traditional OCR engines.
    """
    
    def __init__(self,
                 handwritten_model: str = "microsoft/trocr-base-handwritten",
                 printed_model: str = "microsoft/trocr-base-printed",
                 use_gpu: bool = False,
                 confidence_threshold: float = 0.7):
        """
        Initialize the transformer OCR engine.
        
        Args:
            handwritten_model: Pre-trained model for handwritten text
            printed_model: Pre-trained model for printed text
            use_gpu: Whether to use GPU acceleration if available
            confidence_threshold: Minimum confidence for text detection
        """
        self.handwritten_model_name = handwritten_model
        self.printed_model_name = printed_model
        self.use_gpu = use_gpu and torch.cuda.is_available() if HAVE_TRANSFORMERS else False
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if self.use_gpu else "cpu") if HAVE_TRANSFORMERS else None
        
        # Initialize models
        self.handwritten_model = None
        self.handwritten_processor = None
        self.printed_model = None
        self.printed_processor = None
        
        if HAVE_TRANSFORMERS:
            self._initialize_models()
        
        self.logger = logging.getLogger(__name__)
    
    def _initialize_models(self):
        """Initialize the transformer models for OCR."""
        try:
            # Initialize handwritten text recognition model
            self.logger.info(f"Loading handwritten text recognition model: {self.handwritten_model_name}")
            self.handwritten_processor = TrOCRProcessor.from_pretrained(self.handwritten_model_name)
            self.handwritten_model = VisionEncoderDecoderModel.from_pretrained(self.handwritten_model_name)
            self.handwritten_model.to(self.device)
            self.handwritten_model.eval()
            
            # Initialize printed text recognition model
            self.logger.info(f"Loading printed text recognition model: {self.printed_model_name}")
            self.printed_processor = TrOCRProcessor.from_pretrained(self.printed_model_name)
            self.printed_model = VisionEncoderDecoderModel.from_pretrained(self.printed_model_name)
            self.printed_model.to(self.device)
            self.printed_model.eval()
            
            self.logger.info("Successfully initialized transformer OCR models")
        except Exception as e:
            self.logger.error(f"Error initializing transformer OCR models: {str(e)}")
            self.handwritten_model = None
            self.printed_model = None
    
    def extract_text(self, 
                     image_path: Union[str, Path, np.ndarray], 
                     is_handwritten: Optional[bool] = None,
                     detect_orientation: bool = True,
                     preprocess_image: bool = True) -> Dict[str, Any]:
        """
        Extract text from an image using transformer models.
        
        Args:
            image_path: Path to image or image array
            is_handwritten: Whether the text is handwritten (if None, autodetect)
            detect_orientation: Whether to detect and correct image orientation
            preprocess_image: Whether to apply preprocessing to improve recognition
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        if not HAVE_TRANSFORMERS or (self.handwritten_model is None and self.printed_model is None):
            return self._fallback_extraction(image_path)
        
        try:
            # Load image
            image = self._load_image(image_path)
            
            # Preprocess image if requested
            if preprocess_image:
                image = self._preprocess_image(image)
            
            # Detect orientation and rotate if needed
            if detect_orientation:
                image, angle = self._detect_and_correct_orientation(image)
            else:
                angle = 0
            
            # Auto-detect if text is handwritten if not specified
            if is_handwritten is None:
                is_handwritten = self._detect_handwritten(image)
            
            # Choose appropriate model
            model = self.handwritten_model if is_handwritten else self.printed_model
            processor = self.handwritten_processor if is_handwritten else self.printed_processor
            
            # Convert to PIL Image if it's a numpy array
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Extract text using transformer model
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
                
            # Decode the generated IDs to text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Post-process text
            processed_text = self._post_process_text(generated_text)
            
            return {
                "text": processed_text,
                "is_handwritten": is_handwritten,
                "confidence": 0.9,  # Placeholder - transformers don't always provide confidence
                "orientation_angle": angle,
                "method": "transformer",
                "model_used": self.handwritten_model_name if is_handwritten else self.printed_model_name
            }
        
        except Exception as e:
            self.logger.error(f"Error in transformer OCR: {str(e)}")
            return self._fallback_extraction(image_path)
    
    def _load_image(self, image_path: Union[str, Path, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """
        Load an image from path or array.
        
        Args:
            image_path: Path to image or image array
            
        Returns:
            Loaded image
        """
        if isinstance(image_path, np.ndarray):
            return image_path
            
        if isinstance(image_path, (str, Path)):
            # Try PIL first
            try:
                return Image.open(image_path).convert("RGB")
            except Exception:
                # Fall back to OpenCV
                return cv2.imread(str(image_path))
        
        raise ValueError(f"Unsupported image input type: {type(image_path)}")
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        # If image is PIL Image, convert to numpy array
        if isinstance(image, Image.Image):
            image_array = np.array(image)
            is_pil = True
        else:
            image_array = image
            is_pil = False
        
        # Convert to grayscale if it's color
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
            
        # Apply adaptive thresholding
        # This helps separate text from background even with uneven lighting
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Noise removal
        processed = cv2.medianBlur(processed, 3)
        
        # Dilation to enhance text connectivity
        kernel = np.ones((1, 1), np.uint8)
        processed = cv2.dilate(processed, kernel, iterations=1)
        
        # Convert back to PIL Image if input was PIL
        if is_pil:
            return Image.fromarray(processed)
        return processed
    
    def _detect_and_correct_orientation(self, image: Union[Image.Image, np.ndarray]) -> Tuple[Union[Image.Image, np.ndarray], float]:
        """
        Detect and correct image orientation.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (corrected image, rotation angle)
        """
        # Convert to numpy if it's PIL
        if isinstance(image, Image.Image):
            image_array = np.array(image)
            is_pil = True
        else:
            image_array = image
            is_pil = False
            
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Simple approach: try different angles and see which gives best text confidence
        angles = [0, 90, 180, 270]
        best_angle = 0
        best_conf = -1
        
        for angle in angles:
            # Skip 0 for efficiency
            if angle == 0:
                rotated = gray
            else:
                # Rotate image
                height, width = gray.shape[:2]
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
            
            # Check for text features in the rotated image
            conf = self._estimate_text_confidence(rotated)
            
            if conf > best_conf:
                best_conf = conf
                best_angle = angle
        
        # If best angle is not 0, rotate original image
        if best_angle != 0:
            height, width = image_array.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
            rotated_image = cv2.warpAffine(image_array, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
            
            # Convert back to PIL if needed
            if is_pil:
                return Image.fromarray(rotated_image), best_angle
            return rotated_image, best_angle
        
        return image, 0
    
    def _estimate_text_confidence(self, image: np.ndarray) -> float:
        """
        Estimate how likely an image contains readable text.
        
        Args:
            image: Grayscale image
            
        Returns:
            Confidence score (0-1)
        """
        # Use edge detection to find text-like features
        edges = cv2.Canny(image, 50, 150)
        
        # Calculate horizontal and vertical projection profiles
        h_proj = np.sum(edges, axis=1)
        v_proj = np.sum(edges, axis=0)
        
        # Calculate variance of projections (text typically has high variance)
        h_var = np.var(h_proj)
        v_var = np.var(v_proj)
        
        # Normalize variance
        h_var_norm = min(1.0, h_var / 10000.0)
        v_var_norm = min(1.0, v_var / 10000.0)
        
        # Combine scores
        score = (h_var_norm + v_var_norm) / 2.0
        
        return score
    
    def _detect_handwritten(self, image: Union[Image.Image, np.ndarray]) -> bool:
        """
        Detect if text in image is handwritten.
        
        Args:
            image: Input image
            
        Returns:
            True if likely handwritten, False if likely printed
        """
        # Convert to numpy if it's PIL
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        
        # Features that distinguish handwritten from printed text:
        # 1. Line straightness
        # 2. Character uniformity
        # 3. Stroke width variation
        
        # Edge detection to find text strokes
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to connect strokes
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours of connected components
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate stroke width variation
        stroke_widths = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Skip very small contours
                continue
                
            # Find minimum area rectangle
            rect = cv2.minAreaRect(contour)
            width = min(rect[1])
            
            if width > 0:
                stroke_widths.append(width)
        
        # High variation in stroke width suggests handwriting
        if stroke_widths:
            stroke_std = np.std(stroke_widths)
            stroke_mean = np.mean(stroke_widths)
            variation = stroke_std / stroke_mean if stroke_mean > 0 else 0
            
            # Variation threshold based on empirical testing
            if variation > 0.3:
                return True
        
        # Calculate contour irregularity
        irregularity_scores = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:  # Skip very small contours
                continue
                
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            # Calculate area
            area = cv2.contourArea(contour)
            
            # Circularity/irregularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                irregularity_scores.append(1 - circularity)
        
        # High irregularity suggests handwriting
        if irregularity_scores:
            avg_irregularity = np.mean(irregularity_scores)
            
            # Irregularity threshold based on empirical testing
            if avg_irregularity > 0.7:
                return True
        
        # Default to printed text
        return False
    
    def _post_process_text(self, text: str) -> str:
        """
        Apply post-processing to improve OCR results.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Processed text
        """
        # Remove excessive whitespace
        processed = ' '.join(text.split())
        
        # Fix common OCR errors
        # These could be expanded based on domain-specific needs
        replacements = {
            'l': 'l',  # Fix lowercase L
            '0': 'O',  # Fix zero vs capital O
            'I': 'I',  # Fix capital I
        }
        
        # Apply replacements
        for old, new in replacements.items():
            processed = processed.replace(old, new)
        
        return processed
    
    def _fallback_extraction(self, image_path: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Fallback text extraction using simpler methods.
        
        Args:
            image_path: Path to image or image array
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Try to use Tesseract if available
            import pytesseract
            
            # Load image
            image = self._load_image(image_path)
            
            # Convert to numpy if it's PIL
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Preprocess
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY) if len(image_array.shape) == 3 else image_array
            
            # Use pytesseract
            text = pytesseract.image_to_string(gray)
            
            return {
                "text": text,
                "is_handwritten": None,  # Unknown
                "confidence": 0.5,
                "orientation_angle": 0,
                "method": "tesseract_fallback"
            }
        except Exception as e:
            self.logger.error(f"Error in fallback OCR: {str(e)}")
            return {
                "text": "",
                "is_handwritten": None,
                "confidence": 0,
                "orientation_angle": 0,
                "method": "failed"
            }


class HandwrittenTextRecognizer:
    """
    Specialized class for recognizing handwritten text in documents.
    
    This class combines layout analysis, transformer-based OCR, and post-processing
    to accurately extract handwritten content from complex documents.
    """
    
    def __init__(self, 
                 use_transformers: bool = True,
                 use_layout_analysis: bool = True,
                 use_context_correction: bool = True):
        """
        Initialize the handwritten text recognizer.
        
        Args:
            use_transformers: Whether to use transformer models
            use_layout_analysis: Whether to use layout analysis
            use_context_correction: Whether to use context for correction
        """
        self.use_transformers = use_transformers
        self.use_layout_analysis = use_layout_analysis
        self.use_context_correction = use_context_correction
        
        # Initialize OCR engine
        self.ocr_engine = TransformerOCR() if use_transformers and HAVE_TRANSFORMERS else None
        
        self.logger = logging.getLogger(__name__)
    
    def process_document(self, image_path: Union[str, Path], output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a document image to extract handwritten text.
        
        Args:
            image_path: Path to document image
            output_path: Path to save results (optional)
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Load document image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Step 1: Layout analysis to find regions
            regions = self._analyze_layout(image)
            
            # Step 2: Classify regions as handwritten or printed
            classified_regions = self._classify_regions(image, regions)
            
            # Step 3: Extract text from each region
            extracted_content = []
            for region in classified_regions:
                # Extract region from image
                x1, y1, x2, y2 = region["bbox"]
                height, width = image.shape[:2]
                x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                region_image = image[y1:y2, x1:x2]
                
                if region_image.size == 0:
                    continue
                
                # Extract text using appropriate method
                is_handwritten = region["type"] == "handwritten"
                
                if self.ocr_engine:
                    ocr_result = self.ocr_engine.extract_text(
                        region_image,
                        is_handwritten=is_handwritten
                    )
                    extracted_text = ocr_result["text"]
                else:
                    # Fallback to built-in OCR function
                    import pytesseract
                    extracted_text = pytesseract.image_to_string(region_image)
                
                # Add result
                region_result = {
                    "bbox": region["bbox"],
                    "type": region["type"],
                    "text": extracted_text,
                    "confidence": ocr_result.get("confidence", 0.5) if self.ocr_engine else 0.5
                }
                extracted_content.append(region_result)
            
            # Step 4: Context-based correction if enabled
            if self.use_context_correction:
                extracted_content = self._apply_context_correction(extracted_content)
            
            # Prepare result
            result = {
                "content": extracted_content,
                "method": "transformer" if self.ocr_engine else "fallback",
                "regions_count": len(extracted_content),
                "handwritten_count": sum(1 for r in extracted_content if r["type"] == "handwritten")
            }
            
            # Save results if output path provided
            if output_path:
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            return {
                "content": [],
                "method": "failed",
                "error": str(e)
            }
    
    def _analyze_layout(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyze document layout to find text regions.
        
        Args:
            image: Document image
            
        Returns:
            List of detected regions
        """
        if not self.use_layout_analysis:
            # If layout analysis is disabled, treat whole image as one region
            return [{
                "bbox": [0, 0, 1, 1],  # Normalized coordinates
                "type": "unknown"
            }]
        
        # Simple layout analysis using contours
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and process contours
        regions = []
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 1000:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to normalized coordinates
            bbox = [
                float(x) / width,
                float(y) / height,
                float(x + w) / width,
                float(y + h) / height
            ]
            
            # Add region
            regions.append({
                "bbox": bbox,
                "type": "unknown"  # Will be classified in next step
            })
        
        # If no regions detected, use whole image
        if not regions:
            regions.append({
                "bbox": [0, 0, 1, 1],
                "type": "unknown"
            })
        
        return regions
    
    def _classify_regions(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify regions as handwritten or printed.
        
        Args:
            image: Document image
            regions: Detected regions
            
        Returns:
            Regions with classification
        """
        height, width = image.shape[:2]
        classified_regions = []
        
        for region in regions:
            # Extract region from image
            x1, y1, x2, y2 = region["bbox"]
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            region_image = image[y1:y2, x1:x2]
            
            if region_image.size == 0:
                continue
            
            # Detect if handwritten
            if self.ocr_engine:
                is_handwritten = self.ocr_engine._detect_handwritten(region_image)
            else:
                # Simple fallback classification based on stroke width variation
                gray = cv2.cvtColor(region_image, cv2.COLOR_BGR2GRAY) if len(region_image.shape) == 3 else region_image
                _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                kernel = np.ones((3, 3), np.uint8)
                dilated = cv2.dilate(binary, kernel, iterations=1)
                contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate stroke width variation
                stroke_widths = []
                for contour in contours:
                    if cv2.contourArea(contour) < 50:  # Skip very small contours
                        continue
                    rect = cv2.minAreaRect(contour)
                    width = min(rect[1])
                    if width > 0:
                        stroke_widths.append(width)
                
                if stroke_widths:
                    variation = np.std(stroke_widths) / np.mean(stroke_widths) if np.mean(stroke_widths) > 0 else 0
                    is_handwritten = variation > 0.3
                else:
                    is_handwritten = False
            
            # Update region type
            region["type"] = "handwritten" if is_handwritten else "printed"
            classified_regions.append(region)
        
        return classified_regions
    
    def _apply_context_correction(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply context-based correction to extracted text.
        
        Args:
            regions: Extracted regions with text
            
        Returns:
            Regions with corrected text
        """
        # This would be a more sophisticated implementation using language models
        # For now, just apply basic corrections
        
        for region in regions:
            text = region["text"]
            
            # Basic corrections
            # Correct common errors
            corrections = {
                "l": "1" if region["type"] == "handwritten" else "l",  # Common confusion in handwriting
                "O": "0" if region["type"] == "handwritten" and text.isdigit() else "O",
                "S": "5" if region["type"] == "handwritten" and text.isdigit() else "S"
            }
            
            for old, new in corrections.items():
                text = text.replace(old, new)
            
            # Update region text
            region["text"] = text
        
        return regions
