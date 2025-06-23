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
            'handwritten_content': list of extracted handwritten content,
            'metadata': document metadata
        }
    """
    file_path = validate_file(file_path, extension='.pdf')

    logger.info(f"Parsing PDF file: {file_path}")

    # Get document metadata
    try:
        from pdfminer.pdfpage import PDFPage
        with open(file_path, 'rb') as file:
            page_count = len(list(PDFPage.get_pages(file)))
    except Exception as e:
        logger.warning(f"Could not determine page count: {e}")
        page_count = "unknown"

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

    # Create metadata
    file_stats = Path(file_path).stat()
    metadata = {
        "file_name": Path(file_path).name,
        "file_size": file_stats.st_size,
        "page_count": page_count,
        "extraction_method": "pdfminer + tabula",
        "text_length": len(text) if text else 0,
        "tables_found": len(tables),
        "forms_found": len(forms),
        "images_found": len(images),
        "handwritten_items_found": len(handwritten_content)
    }

    return {
        'text': text,
        'images': images,
        'tables': tables,
        'forms': forms,
        'handwritten_content': handwritten_content,
        'metadata': metadata
    }

def extract_text(file_path: Union[str, Path]) -> str:
    """
    Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content (cleaned for JSON compatibility)
    """
    try:
        raw_text = pdfminer_extract_text(file_path)

        # Clean the text for better JSON compatibility
        if raw_text:
            # Replace problematic unicode characters
            cleaned_text = raw_text.replace('\u2013', '-')  # em dash
            cleaned_text = cleaned_text.replace('\u2014', '--')  # en dash
            cleaned_text = cleaned_text.replace('\u2019', "'")  # right single quotation
            cleaned_text = cleaned_text.replace('\u201c', '"')  # left double quotation
            cleaned_text = cleaned_text.replace('\u201d', '"')  # right double quotation
            cleaned_text = cleaned_text.replace('\u00a0', ' ')  # non-breaking space

            # Remove excessive whitespace
            cleaned_text = ' '.join(cleaned_text.split())

            return cleaned_text

        return ""
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

        cleaned_tables = []
        for table in tables:
            # Clean the table data
            # Replace NaN values with None and clean strings
            cleaned_table = table.fillna('')  # Replace NaN with empty string

            # Convert to dictionary and clean further
            table_dict = cleaned_table.to_dict(orient='records')

            # Clean each row
            cleaned_rows = []
            for row in table_dict:
                cleaned_row = {}
                for key, value in row.items():
                    # Clean the key
                    clean_key = str(key).strip() if key is not None else "unknown_column"
                    if not clean_key or clean_key.lower() in ['nan', 'none']:
                        clean_key = "unknown_column"

                    # Clean the value
                    if value is None or (isinstance(value, float) and (value != value)):  # Check for NaN
                        clean_value = None
                    elif isinstance(value, str):
                        clean_value = value.strip()
                        if not clean_value or clean_value.lower() in ['nan', 'none']:
                            clean_value = None
                    else:
                        clean_value = value

                    cleaned_row[clean_key] = clean_value

                # Only add row if it has some meaningful content
                if any(v is not None and str(v).strip() for v in cleaned_row.values()):
                    cleaned_rows.append(cleaned_row)

            if cleaned_rows:  # Only add table if it has content
                cleaned_tables.append({
                    "data": cleaned_rows,
                    "shape": (len(cleaned_rows), len(cleaned_rows[0]) if cleaned_rows else 0),
                    "confidence": 0.8  # Default confidence for tabula extraction
                })

        return cleaned_tables
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
