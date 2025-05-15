"""
Unified PDF Parser module integrating all advanced capabilities.

This module provides a comprehensive interface that utilizes all the advanced
PDF parsing capabilities to handle even the most complex PDF documents.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json
import tempfile

# FIXME: Consider switching to a more efficient temp file management
# The current approach works but can leave orphaned files if process crashes
# - praveen

# Import advanced modules
from od_parse.advanced.document_intelligence import DocumentIntelligence
from od_parse.advanced.layout_analysis import LayoutAnalyzer
from od_parse.advanced.form_understanding import FormUnderstanding
from od_parse.advanced.table_extraction import AdvancedTableExtractor
from od_parse.advanced.semantic_structure import SemanticStructureExtractor
from od_parse.advanced.neural_table_extraction import NeuralTableExtractor
from od_parse.advanced.transformer_ocr import TransformerOCR, HandwrittenTextRecognizer
from od_parse.advanced.dl_layout_detection import DocumentSegmentation

# Import base modules
from od_parse.utils.file_utils import validate_file
from od_parse.utils.logging_utils import get_logger


class UnifiedPDFParser:
    """
    Comprehensive PDF parser that integrates all advanced capabilities.
    
    This class provides a unified interface to the various advanced parsing
    modules, making it easy to extract rich, structured content from even
    the most complex PDF documents.
    
    TODO: We might want to split this into specialized parsers at some point
    since this class is getting pretty big. For now, keeping everything in
    one place for simplicity. -PS 2/28/25
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # This monster handles pretty much everything - hope we don't regret it later!
        """
        Initialize the unified parser with configuration options.
        
        Args:
            config: Configuration dictionary with the following options:
                - use_deep_learning: Whether to use deep learning models
                - extract_handwritten: Whether to extract handwritten content
                - extract_tables: Whether to extract tables
                - extract_forms: Whether to extract form elements
                - extract_structure: Whether to extract document structure
                - output_format: Format for output (json, markdown, text)
        """
        self.logger = get_logger(__name__)
        
        # Default configuration - these took forever to tune right!
        self.my_config = {
            "use_deep_learning": True,  # This is super slow but worth it
            "extract_handwritten": True,
            "extract_tables": True,
            "extract_forms": True,  # Sometimes breaks on complex forms
            "extract_structure": True,
            "output_format": "json"  # TODO: add XML option? -PS
        }
        
        # Update with user config
        if config:
            self.my_config.update(config)
            
        # HACK: Weird edge case where empty dict breaks things
        if not self.my_config:
            raise ValueError("Empty config somehow? Check caller")
        
        # Initialize components based on configuration
        try:
            # Core document intelligence
            self.document_intelligence = DocumentIntelligence()
            
            # Layout analysis
            self.layout_analyzer = LayoutAnalyzer()
            
            # Table extraction
            # WARNING: neural extraction can use tons of memory - crashed my machine once with a 2000 page doc
            if self.my_config["extract_tables"]:
                if self.my_config["use_deep_learning"]:
                    self.tbl_extractor = NeuralTableExtractor()
                else:
                    self.tbl_extractor = AdvancedTableExtractor()  # Fast but less accurate
            else:
                self.tbl_extractor = None
                
            # Form understanding - needs better support for complex checkboxes
            # Ran into all kinds of weird form layouts in bank docs - maybe do some specialization? -PS
            if self.my_config["extract_forms"]:
                self.form_engine = FormUnderstanding()
            else:
                self.form_engine = None
                
            # Document structure 
            # FIXME: This is still pretty limited for scientific papers with equations
            # Mike's custom header detection might be better for those cases
            if self.my_config["extract_structure"]:
                self.doc_structure = SemanticStructureExtractor()
            else:
                self.doc_structure = None
                
            # Handwritten content
            if self.config["extract_handwritten"]:
                if self.config["use_deep_learning"]:
                    self.handwritten_extractor = HandwrittenTextRecognizer()
                else:
                    # Use simpler OCR approach
                    self.handwritten_extractor = None
            else:
                self.handwritten_extractor = None
                
            # Deep learning layout detection
            if self.config["use_deep_learning"]:
                self.dl_layout_detector = DocumentSegmentation()
            else:
                self.dl_layout_detector = None
                
            self.logger.info("Successfully initialized UnifiedPDFParser with all components")
            
        except Exception as e:
            self.logger.error(f"Error initializing UnifiedPDFParser components: {str(e)}")
    
    def parse(self, file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
        # This is the workhorse method - have fun navigating this beast!
        # Been through 4 refactors and still growing...
        """
        Parse a PDF file with all enabled capabilities.
        
        Args:
            file_path: Path to the PDF file
            **kwargs: Additional arguments to pass to specific extractors
            
        Returns:
            Comprehensive dictionary containing all extracted content
        """
        file_path = validate_file(file_path, extension='.pdf')
        self.logger.info(f"Parsing PDF file: {file_path}")
        
        try:
            # Step 1: Basic PDF processing - convert to images
            images_dir, image_paths = self._convert_pdf_to_images(file_path)
            
            # Step 2: Extract text and layout using document intelligence
            document_info = self._extract_document_info(file_path)
            
            # Step 3: Process each page
            pages = []
            for i, image_path in enumerate(image_paths):
                page_result = self._process_page(image_path, page_num=i)
                pages.append(page_result)
            
            # Step 4: Extract document structure
            structure = self._extract_document_structure(file_path, pages)
            
            # Step 5: Combine results
            result = {
                "document_info": document_info,
                "pages": pages,
                "structure": structure
            }
            
            # Step 6: Format output according to configuration
            formatted_result = self._format_output(result)
            
            # Step 7: Clean up temporary files
            self._cleanup(images_dir)
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            return {"error": str(e)}
    
    def _convert_pdf_to_images(self, pdf_path: Union[str, Path]) -> Tuple[str, List[str]]:
        """
        Convert PDF pages to images for processing.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (temp directory, list of image paths)
        """
        import pdf2image
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="odparsed_")
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            image_paths = []
            
            for i, img in enumerate(images):
                img_path = os.path.join(temp_dir, f"page_{i+1}.png")
                img.save(img_path, "PNG")
                image_paths.append(img_path)
            
            return temp_dir, image_paths
        
        except Exception as e:
            self.logger.error(f"Error converting PDF to images: {str(e)}")
            return temp_dir, []
    
    def _extract_document_info(self, pdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract basic document information from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing document metadata
        """
        from PyPDF2 import PdfReader
        
        try:
            reader = PdfReader(pdf_path)
            info = reader.metadata
            
            # Extract document info
            doc_info = {
                "title": info.title if hasattr(info, 'title') else None,
                "author": info.author if hasattr(info, 'author') else None,
                "creator": info.creator if hasattr(info, 'creator') else None,
                "producer": info.producer if hasattr(info, 'producer') else None,
                "subject": info.subject if hasattr(info, 'subject') else None,
                "creation_date": info.creation_date if hasattr(info, 'creation_date') else None,
                "modification_date": info.modification_date if hasattr(info, 'modification_date') else None,
                "page_count": len(reader.pages),
            }
            
            return doc_info
        
        except Exception as e:
            self.logger.error(f"Error extracting document info: {str(e)}")
            return {
                "page_count": 0,
                "error": str(e)
            }
    
    def _process_page(self, image_path: str, page_num: int) -> Dict[str, Any]:
        """
        Process a single page image with all enabled extractors.
        
        Args:
            image_path: Path to page image
            page_num: Page number
            
        Returns:
            Dictionary containing extracted page content
        """
        page_result = {
            "page_number": page_num + 1,
            "layout": None,
            "text_blocks": [],
            "tables": [],
            "forms": [],
            "handwritten": []
        }
        
        # Step 1: Layout analysis
        if self.dl_layout_detector:
            layout = self.dl_layout_detector.segment_document(image_path)
            page_result["layout"] = layout
        else:
            layout = self.layout_analyzer.analyze([])  # Empty placeholder
            page_result["layout"] = layout
        
        # Step 2: Extract tables if enabled
        # HACK: This is where performance can really tank on complex docs
        if self.tbl_extractor:
            # TODO: Add multi-threading here?
            tables = self.tbl_extractor.extract_tables(image_path, page_num)
            page_result["tables"] = [
                {
                    "bbox": table.bbox,
                    "rows": table.rows,
                    "cols": table.cols,
                    "cells": [
                        {
                            "row": cell.row,
                            "col": cell.col,
                            "text": cell.text,
                            "bbox": cell.bbox,
                            "confidence": cell.confidence
                        }
                        for cell in table.cells
                    ] if table.cells else [],
                    "structure_type": table.structure_type,
                    "confidence": table.confidence,
                    "markdown": self.table_extractor.table_to_markdown(table),
                    "html": self.table_extractor.table_to_html(table)
                }
                for table in tables
            ]
        
        # Step 3: Extract form elements if enabled
        # The form extraction is my favorite part of this whole thing -PS
        if self.form_engine:
            import cv2
            image = cv2.imread(image_path)
            if image is not None:
                # Create simple PDF data structure for form analyzer
                pdf_data = {
                    "images": [{"page": page_num, "path": image_path}]
                }
                # This is slow but probably the most accurate part of the pipeline
                form_results = self.form_engine.extract_forms(pdf_data)
                page_result["forms"] = form_results
        
        # Step 4: Extract handwritten content if enabled
        if self.handwritten_extractor:
            handwritten = self.handwritten_extractor.process_document(image_path)
            page_result["handwritten"] = handwritten.get("content", [])
        
        return page_result
    
    def _extract_document_structure(self, pdf_path: Union[str, Path], pages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract document structure using semantic structure extractor.
        
        Args:
            pdf_path: Path to PDF file
            pages: List of processed pages
            
        Returns:
            Dictionary containing document structure
        """
        # If we don't have a doc structure extractor, just return an empty dict
        if not self.doc_structure:
            return {}
        
        try:
            # Combine all text blocks from pages
            # This is a bit of a hack, but it works for now
            text_blocks = []
            for page in pages:
                for block in page.get("text_blocks", []):
                    block["page"] = page["page_number"] - 1
                    text_blocks.append(block)
                
                # Also process text from tables
                for table in page.get("tables", []):
                    for cell in table.get("cells", []):
                        if cell.get("text"):
                            text_blocks.append({
                                "text": cell["text"],
                                "bbox": cell["bbox"],
                                "page": page["page_number"] - 1
                            })
            
            # Create PDF data structure for structure extractor
            pdf_data = {
                "text_blocks": text_blocks
            }
            
            # Extract structure
            # Sometimes this catches headers that aren't really headers
            # Especially in financial docs with bolded text
            doc_hierarchy = self.doc_structure.extract_structure(pdf_data)
            return doc_hierarchy
        
        except Exception as e:
            self.logger.error(f"Error extracting document structure: {str(e)}")
            return {}
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format output according to configuration.
        
        Args:
            result: Extracted content
            
        Returns:
            Formatted result
        """
        # For now, just return the JSON structure
        # In a full implementation, this would convert to other formats
        return result
    
    def _cleanup(self, temp_dir: str) -> None:
        """
        Clean up temporary files.
        
        Args:
            temp_dir: Temporary directory
        """
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {str(e)}")
    
    def to_markdown(self, parsed_data: Dict[str, Any]) -> str:
        """
        Convert parsed PDF data to Markdown.
        
        Args:
            parsed_data: Output from parse() method
            
        Returns:
            Markdown representation of parsed data
        """
        md_lines = []
        
        # Document title
        doc_info = parsed_data.get("document_info", {})
        title = doc_info.get("title", "Untitled Document")
        md_lines.append(f"# {title}")
        md_lines.append("")
        
        # Document metadata
        md_lines.append("## Document Information")
        md_lines.append("")
        md_lines.append(f"- **Author:** {doc_info.get('author', 'Unknown')}")
        md_lines.append(f"- **Pages:** {doc_info.get('page_count', 0)}")
        if doc_info.get('creation_date'):
            md_lines.append(f"- **Created:** {doc_info.get('creation_date')}")
        md_lines.append("")
        
        # Document structure
        structure = parsed_data.get("structure", {})
        if structure:
            md_lines.append("## Document Structure")
            md_lines.append("")
            
            # Add elements
            elements = structure.get("elements", [])
            for element in elements:
                if element.get("type") == "heading":
                    level = element.get("level", 1)
                    md_lines.append(f"{'#' * (level + 1)} {element.get('text', '')}")
                elif element.get("type") == "paragraph":
                    md_lines.append(element.get("text", ""))
                    md_lines.append("")
                elif element.get("type") == "list_item":
                    md_lines.append(f"- {element.get('text', '')}")
            
            md_lines.append("")
        
        # Process each page
        pages = parsed_data.get("pages", [])
        for page in pages:
            page_num = page.get("page_number", 0)
            md_lines.append(f"## Page {page_num}")
            md_lines.append("")
            
            # Add tables
            tables = page.get("tables", [])
            for i, table in enumerate(tables):
                md_lines.append(f"### Table {i+1}")
                md_lines.append("")
                md_lines.append(table.get("markdown", ""))
                md_lines.append("")
            
            # Add forms
            forms = page.get("forms", {}).get("fields", [])
            if forms:
                md_lines.append("### Form Fields")
                md_lines.append("")
                for field in forms:
                    field_type = field.get("type", "unknown")
                    label = field.get("label", "Unlabeled Field")
                    value = field.get("value", "")
                    
                    if field_type == "checkbox":
                        status = "☑" if field.get("is_checked") else "☐"
                        md_lines.append(f"- {status} **{label}**")
                    else:
                        md_lines.append(f"- **{label}:** {value}")
                md_lines.append("")
            
            # Add handwritten content
            handwritten = page.get("handwritten", [])
            if handwritten:
                md_lines.append("### Handwritten Content")
                md_lines.append("")
                for item in handwritten:
                    md_lines.append(f"- {item.get('text', '')}")
                md_lines.append("")
        
        return "\n".join(md_lines)
