"""
Core PDF parsing functionality.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import pdfminer
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams, LTTextContainer, LTImage, LTFigure
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
import pdf2image
import tabula
import cv2
import numpy as np

from od_parse.ocr import extract_handwritten_content
from od_parse.utils.file_utils import validate_file
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

def parse_pdf(file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Parse a PDF file and extract its content.
    
    Args:
        file_path: Path to the PDF file
        **kwargs: Additional arguments for parsing
    
    Returns:
        Dictionary containing extracted content:
        {
            'text': extracted text content,
            'images': list of extracted images,
            'tables': list of extracted tables,
            'forms': list of extracted form elements,
            'handwritten_content': list of extracted handwritten content
        }
    """
    file_path = validate_file(file_path, extension='.pdf')
    
    logger.info(f"Parsing PDF file: {file_path}")
    
    # Extract content
    text = extract_text(file_path)
    images = extract_images(file_path)
    tables = extract_tables(file_path)
    forms = extract_forms(file_path)
    
    # Process images for handwritten content
    handwritten_content = []
    for img_path in images:
        try:
            content = extract_handwritten_content(img_path)
            if content:
                handwritten_content.append(content)
        except Exception as e:
            logger.error(f"Error extracting handwritten content from {img_path}: {e}")
    
    return {
        'text': text,
        'images': images,
        'tables': tables,
        'forms': forms,
        'handwritten_content': handwritten_content
    }

def extract_text(file_path: Union[str, Path]) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        Extracted text content
    """
    try:
        return pdfminer_extract_text(file_path)
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_images(file_path: Union[str, Path], output_dir: Optional[str] = None) -> List[str]:
    """
    Extract images from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted images
    
    Returns:
        List of paths to extracted images
    """
    file_path = Path(file_path)
    if output_dir is None:
        output_dir = file_path.parent / f"{file_path.stem}_images"
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Convert PDF pages to images
        images = pdf2image.convert_from_path(file_path)
        image_paths = []
        
        for i, img in enumerate(images):
            img_path = os.path.join(output_dir, f"page_{i+1}.png")
            img.save(img_path, "PNG")
            image_paths.append(img_path)
            
            # Use OpenCV to detect and extract embedded images
            cv_img = cv2.imread(img_path)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for j, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped = cv_img[y:y+h, x:x+w]
                    crop_path = os.path.join(output_dir, f"page_{i+1}_img_{j+1}.png")
                    cv2.imwrite(crop_path, cropped)
                    image_paths.append(crop_path)
        
        return image_paths
    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {e}")
        return []

def extract_tables(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF file.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        List of extracted tables as dictionaries
    """
    try:
        # Use tabula-py to extract tables
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        return [table.to_dict(orient='records') for table in tables]
    except Exception as e:
        logger.error(f"Error extracting tables from {file_path}: {e}")
        return []

def extract_forms(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract form elements (checkboxes, radio buttons, text fields) from a PDF file.
    
    Args:
        file_path: Path to the PDF file
    
    Returns:
        List of extracted form elements as dictionaries
    """
    try:
        # This is a simplified implementation
        # In a real-world scenario, you would use a library like PyPDF2 or pdfrw
        # to extract form fields
        
        form_elements = []
        
        # Open the PDF file
        with open(file_path, 'rb') as file:
            # Create a PDF resource manager and page aggregator
            resource_manager = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(resource_manager, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)
            
            # Process each page
            for page_num, page in enumerate(PDFPage.get_pages(file)):
                interpreter.process_page(page)
                layout = device.get_result()
                
                # Look for potential form elements
                for element in layout:
                    if isinstance(element, LTTextContainer):
                        text = element.get_text().strip().lower()
                        if any(keyword in text for keyword in ['check', 'select', 'choose', 'click']):
                            form_elements.append({
                                'type': 'potential_form_field',
                                'page': page_num + 1,
                                'text': text,
                                'bbox': (element.x0, element.y0, element.x1, element.y1)
                            })
        
        return form_elements
    except Exception as e:
        logger.error(f"Error extracting form elements from {file_path}: {e}")
        return []
