"""
Document Intelligence module for understanding document structure and content.

This module provides advanced algorithms for analyzing the semantic structure,
layout, and content of documents beyond simple text extraction.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass


@dataclass
class DocumentRegion:
    """Represents a semantic region within a document."""
    x1: float
    y1: float
    x2: float
    y2: float
    page_number: int
    content_type: str
    confidence: float
    content: Any


class DocumentIntelligence:
    """
    Core class for document understanding that goes beyond basic PDF parsing.
    
    This class implements advanced algorithms to understand the semantic structure,
    content relationships, and hierarchical organization of documents.
    """
    
    def __init__(self, use_deep_learning=True, context_aware=True):
        """
        Initialize the DocumentIntelligence engine.
        
        Args:
            use_deep_learning: Whether to use deep learning models for document understanding
            context_aware: Whether to analyze elements in context with surrounding elements
        """
        self.use_deep_learning = use_deep_learning
        self.context_aware = context_aware
        
        # Load models when they're needed (lazy loading)
        self._layout_model = None
        self._content_classifier = None
        
    def analyze_document(self, parsed_data: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of document structure and content.
        
        Args:
            parsed_data: Basic parsed data from the PDF parser
            
        Returns:
            Enhanced document structure with semantic understanding
        """
        # Extract layout information
        layout_info = self._analyze_layout(parsed_data)
        
        # Classify content types
        content_metadata = self._classify_content(parsed_data)
        
        # Determine document hierarchy
        hierarchy = self._extract_hierarchy(parsed_data, layout_info)
        
        # Identify key-value pairs
        key_values = self._extract_key_values(parsed_data)
        
        # Detect document type
        doc_type, confidence = self._detect_document_type(parsed_data, layout_info)
        
        return {
            "document_type": {
                "type": doc_type,
                "confidence": confidence
            },
            "layout": layout_info,
            "content_metadata": content_metadata,
            "hierarchy": hierarchy,
            "key_values": key_values,
            "original_data": parsed_data
        }
    
    def _analyze_layout(self, parsed_data: Dict) -> Dict[str, Any]:
        """
        Analyze the spatial layout of document elements.
        
        Args:
            parsed_data: Basic parsed data from the PDF parser
            
        Returns:
            Layout analysis results
        """
        # Implement layout analysis logic
        # This would typically involve analyzing positions of text blocks, images, etc.
        layout_regions = []
        
        # Process text blocks
        if "text_blocks" in parsed_data:
            for i, block in enumerate(parsed_data.get("text_blocks", [])):
                region = self._create_region_from_block(block, f"text_{i}", "text")
                layout_regions.append(region)
        
        # Process images
        if "images" in parsed_data:
            for i, image in enumerate(parsed_data.get("images", [])):
                region = self._create_region_from_block(image, f"image_{i}", "image")
                layout_regions.append(region)
        
        # Process tables
        if "tables" in parsed_data:
            for i, table in enumerate(parsed_data.get("tables", [])):
                region = self._create_region_from_block(table, f"table_{i}", "table")
                layout_regions.append(region)
                
        # Analyze columnar structure
        columns = self._detect_columns(layout_regions)
        
        # Analyze reading order
        reading_order = self._determine_reading_order(layout_regions)
        
        return {
            "regions": layout_regions,
            "columns": columns,
            "reading_order": reading_order
        }
    
    def _create_region_from_block(self, block, id, type):
        """Create a region object from a block of content."""
        return {
            "id": id,
            "type": type,
            "x1": block.get("x1", 0),
            "y1": block.get("y1", 0),
            "x2": block.get("x2", 1),
            "y2": block.get("y2", 1),
            "page": block.get("page", 0),
            "content": block.get("content", "")
        }
    
    def _detect_columns(self, regions):
        """Detect column structure in the document."""
        # Simplified column detection
        # Group regions by x-coordinate ranges
        x_ranges = {}
        for region in regions:
            center_x = (region["x1"] + region["x2"]) / 2
            range_key = int(center_x * 10)  # Quantize to reduce noise
            if range_key not in x_ranges:
                x_ranges[range_key] = []
            x_ranges[range_key].append(region)
        
        # Sort and group neighboring ranges
        column_groups = []
        current_group = []
        
        for key in sorted(x_ranges.keys()):
            if not current_group or abs(key - current_group[-1]) <= 1:
                current_group.append(key)
            else:
                column_groups.append(current_group)
                current_group = [key]
        
        if current_group:
            column_groups.append(current_group)
        
        # Convert back to regions
        columns = []
        for group in column_groups:
            column_regions = []
            for key in group:
                column_regions.extend(x_ranges.get(key, []))
            columns.append(column_regions)
        
        return columns
    
    def _determine_reading_order(self, regions):
        """Determine the natural reading order of regions."""
        # Simple reading order: top-to-bottom, left-to-right
        sorted_regions = sorted(
            range(len(regions)), 
            key=lambda i: (regions[i]["page"], regions[i]["y1"], regions[i]["x1"])
        )
        return sorted_regions
    
    def _classify_content(self, parsed_data):
        """Classify the content types in the document."""
        if self._content_classifier is None and self.use_deep_learning:
            # In a real implementation, this would load a trained model
            pass
            
        # Placeholder for content classification
        # In a full implementation, this would use ML to classify content
        content_types = {}
        
        # Simple rule-based classification as fallback
        if "text_blocks" in parsed_data:
            for i, block in enumerate(parsed_data.get("text_blocks", [])):
                content_type = self._classify_text_block(block)
                content_types[f"text_{i}"] = content_type
                
        return content_types
    
    def _classify_text_block(self, block):
        """Classify a text block by its content and style."""
        text = block.get("text", "").lower()
        font_size = block.get("font_size", 0)
        is_bold = block.get("is_bold", False)
        
        # Simple rule-based classification
        if is_bold and font_size > 14:
            return "heading"
        elif any(word in text for word in ["total", "sum", "amount", "$", "€", "£"]):
            return "financial"
        elif any(word in text for word in ["date", "time", "schedule"]):
            return "temporal"
        elif len(text) < 5 and text.isdigit():
            return "numeric"
        else:
            return "body_text"
    
    def _extract_hierarchy(self, parsed_data, layout_info):
        """Extract the hierarchical structure of the document."""
        hierarchy = {
            "root": [],
            "children": {}
        }
        
        # Simple hierarchy extraction based on headings
        current_heading = None
        current_section = None
        
        # Process regions in reading order
        for idx in layout_info["reading_order"]:
            region = layout_info["regions"][idx]
            region_id = region["id"]
            
            # Check if it's a heading
            is_heading = False
            if region["type"] == "text":
                content_type = self._classify_text_block({"text": region["content"]})
                is_heading = (content_type == "heading")
            
            if is_heading:
                # New section
                current_heading = region_id
                current_section = []
                hierarchy["root"].append(current_heading)
                hierarchy["children"][current_heading] = current_section
            elif current_heading is not None:
                # Add to current section
                current_section.append(region_id)
        
        return hierarchy
    
    def _extract_key_values(self, parsed_data):
        """Extract key-value pairs from the document."""
        key_values = []
        
        # Process text blocks for key-value pairs
        if "text_blocks" in parsed_data:
            for block in parsed_data.get("text_blocks", []):
                text = block.get("text", "")
                
                # Look for patterns like "Key: Value" or "Key = Value"
                if ":" in text:
                    parts = text.split(":", 1)
                    key_values.append({
                        "key": parts[0].strip(),
                        "value": parts[1].strip(),
                        "confidence": 0.9
                    })
                elif "=" in text:
                    parts = text.split("=", 1)
                    key_values.append({
                        "key": parts[0].strip(),
                        "value": parts[1].strip(),
                        "confidence": 0.8
                    })
        
        return key_values
    
    def _detect_document_type(self, parsed_data, layout_info):
        """Detect the type of document based on content and layout."""
        # Simple document type detection based on content keywords
        all_text = ""
        if "text_blocks" in parsed_data:
            for block in parsed_data.get("text_blocks", []):
                all_text += block.get("text", "") + " "
        
        all_text = all_text.lower()
        
        # Check for document type indicators
        if "invoice" in all_text or "bill" in all_text:
            return "invoice", 0.8
        elif "agreement" in all_text or "contract" in all_text:
            return "contract", 0.7
        elif "resume" in all_text or "cv" in all_text:
            return "resume", 0.8
        elif "report" in all_text:
            return "report", 0.6
        elif "form" in all_text:
            return "form", 0.7
        else:
            return "generic", 0.5
