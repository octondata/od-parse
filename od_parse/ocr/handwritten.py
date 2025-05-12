"""
Module for extracting handwritten content from images using OCR.
"""

import os
from typing import Union, Optional, Dict, Any
from pathlib import Path

import pytesseract
from PIL import Image
import cv2
import numpy as np

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

def extract_handwritten_content(image_path: Union[str, Path], **kwargs) -> Optional[str]:
    """
    Extract handwritten content from an image using OCR.
    
    Args:
        image_path: Path to the image file
        **kwargs: Additional arguments for OCR
    
    Returns:
        Extracted text content or None if extraction failed
    """
    try:
        # Load the image
        image = Image.open(image_path)
        
        # Convert to OpenCV format for preprocessing
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Preprocess the image to improve OCR accuracy
        processed_img = preprocess_image_for_ocr(img_cv)
        
        # Use Tesseract OCR to extract text
        # Configure Tesseract for handwritten text
        custom_config = r'--oem 1 --psm 6 -l eng'
        
        if 'config' in kwargs:
            custom_config = kwargs['config']
        
        text = pytesseract.image_to_string(processed_img, config=custom_config)
        
        # Clean up the extracted text
        text = text.strip()
        
        return text if text else None
    except Exception as e:
        logger.error(f"Error extracting handwritten content from {image_path}: {e}")
        return None

def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image to improve OCR accuracy for handwritten content.
    
    Args:
        image: OpenCV image
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Dilate to connect broken parts of handwritten text
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # Erode to remove small noise
    eroded = cv2.erode(dilated, kernel, iterations=1)
    
    return eroded

def detect_handwritten_regions(image_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Detect regions in an image that likely contain handwritten content.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Dictionary with information about detected handwritten regions
    """
    try:
        # Load the image
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and shape
        handwritten_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area > 100 and w > 10 and h > 10:
                # Calculate aspect ratio
                aspect_ratio = w / float(h)
                
                # Handwritten text typically has a certain aspect ratio
                if 0.1 < aspect_ratio < 10:
                    roi = image[y:y+h, x:x+w]
                    handwritten_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'roi': roi
                    })
        
        return {
            'image_path': image_path,
            'num_regions': len(handwritten_regions),
            'regions': handwritten_regions
        }
    except Exception as e:
        logger.error(f"Error detecting handwritten regions in {image_path}: {e}")
        return {
            'image_path': image_path,
            'num_regions': 0,
            'regions': [],
            'error': str(e)
        }
