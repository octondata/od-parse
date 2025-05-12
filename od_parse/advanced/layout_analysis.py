"""
Layout Analysis module for understanding the spatial arrangement of document elements.

This module provides advanced algorithms for analyzing document layout,
including column detection, reading order determination, and logical grouping.
"""

from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class LayoutElement:
    """Represents a layout element within a document."""
    id: str
    element_type: str
    x1: float
    y1: float
    x2: float
    y2: float
    page: int
    confidence: float = 1.0


class LayoutAnalyzer:
    """
    Analyzes the spatial layout and structure of document elements.
    
    This class implements advanced algorithms to understand document layout,
    including multi-column detection, reading order determination, and 
    logical grouping of elements.
    """
    
    def __init__(self, 
                 detect_columns=True, 
                 detect_reading_order=True,
                 detect_logical_sections=True):
        """
        Initialize the LayoutAnalyzer.
        
        Args:
            detect_columns: Whether to detect column structure
            detect_reading_order: Whether to determine reading order
            detect_logical_sections: Whether to group elements into logical sections
        """
        self.detect_columns = detect_columns
        self.detect_reading_order = detect_reading_order
        self.detect_logical_sections = detect_logical_sections
        
    def analyze(self, elements: List[Dict]) -> Dict[str, Any]:
        """
        Analyze the layout of document elements.
        
        Args:
            elements: List of document elements with position information
            
        Returns:
            Layout analysis results
        """
        layout_elements = self._convert_to_layout_elements(elements)
        
        result = {
            "elements": layout_elements
        }
        
        # Detect columns if enabled
        if self.detect_columns:
            columns = self._detect_columns(layout_elements)
            result["columns"] = columns
        
        # Determine reading order if enabled
        if self.detect_reading_order:
            reading_order = self._determine_reading_order(layout_elements)
            result["reading_order"] = reading_order
        
        # Group elements into logical sections if enabled
        if self.detect_logical_sections:
            sections = self._detect_logical_sections(layout_elements)
            result["logical_sections"] = sections
        
        # Detect page structure
        page_structure = self._analyze_page_structure(layout_elements)
        result["page_structure"] = page_structure
        
        return result
    
    def _convert_to_layout_elements(self, elements: List[Dict]) -> List[LayoutElement]:
        """Convert raw elements to LayoutElement objects."""
        layout_elements = []
        
        for i, elem in enumerate(elements):
            # Handle different input formats
            if isinstance(elem, LayoutElement):
                layout_elements.append(elem)
                continue
                
            # Extract coordinates with fallbacks
            x1 = elem.get("x1", elem.get("bbox", [0, 0, 0, 0])[0] if "bbox" in elem else 0)
            y1 = elem.get("y1", elem.get("bbox", [0, 0, 0, 0])[1] if "bbox" in elem else 0)
            x2 = elem.get("x2", elem.get("bbox", [0, 0, 0, 0])[2] if "bbox" in elem else 1)
            y2 = elem.get("y2", elem.get("bbox", [0, 0, 0, 0])[3] if "bbox" in elem else 1)
            
            # Create layout element
            element_id = elem.get("id", f"elem_{i}")
            element_type = elem.get("type", "unknown")
            page = elem.get("page", 0)
            confidence = elem.get("confidence", 1.0)
            
            layout_element = LayoutElement(
                id=element_id,
                element_type=element_type,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                page=page,
                confidence=confidence
            )
            
            layout_elements.append(layout_element)
            
        return layout_elements
    
    def _detect_columns(self, elements: List[LayoutElement]) -> List[List[str]]:
        """
        Detect column structure in the document.
        
        Uses a density-based approach to identify columns.
        
        Args:
            elements: List of document elements
            
        Returns:
            List of columns, each containing element IDs
        """
        # Skip if no elements
        if not elements:
            return []
            
        # Group elements by page
        elements_by_page = {}
        for elem in elements:
            if elem.page not in elements_by_page:
                elements_by_page[elem.page] = []
            elements_by_page[elem.page].append(elem)
        
        all_columns = []
        
        # Process each page separately
        for page, page_elements in elements_by_page.items():
            # Create x-axis histogram for this page
            histogram = self._create_x_histogram(page_elements)
            
            # Find valleys in histogram (column separators)
            valleys = self._find_histogram_valleys(histogram)
            
            # Convert valleys to column boundaries
            column_boundaries = [(0, valleys[0])] if valleys else [(0, 1.0)]
            for i in range(len(valleys) - 1):
                column_boundaries.append((valleys[i], valleys[i+1]))
            if valleys:
                column_boundaries.append((valleys[-1], 1.0))
            
            # Assign elements to columns
            columns = [[] for _ in range(len(column_boundaries))]
            for elem in page_elements:
                center_x = (elem.x1 + elem.x2) / 2
                
                # Find which column this element belongs to
                for i, (start, end) in enumerate(column_boundaries):
                    if start <= center_x < end:
                        columns[i].append(elem.id)
                        break
            
            all_columns.extend(columns)
        
        return all_columns
    
    def _create_x_histogram(self, elements: List[LayoutElement], bins=100) -> np.ndarray:
        """Create a histogram of element positions along the x-axis."""
        # Create empty histogram
        histogram = np.zeros(bins)
        
        # Add each element to the histogram
        for elem in elements:
            # Convert coordinates to bin indices
            start_bin = min(int(elem.x1 * bins), bins - 1)
            end_bin = min(int(elem.x2 * bins), bins - 1)
            
            # Add to histogram
            histogram[start_bin:end_bin+1] += 1
        
        return histogram
    
    def _find_histogram_valleys(self, histogram: np.ndarray, smoothing=5, threshold=0.3) -> List[float]:
        """Find valleys in a histogram (indicating column separators)."""
        # Smooth histogram
        if smoothing > 1:
            kernel = np.ones(smoothing) / smoothing
            smoothed = np.convolve(histogram, kernel, mode='same')
        else:
            smoothed = histogram
        
        # Normalize
        if np.max(smoothed) > 0:
            smoothed = smoothed / np.max(smoothed)
        
        # Find local minima
        valleys = []
        for i in range(1, len(smoothed) - 1):
            if (smoothed[i] < smoothed[i-1] and 
                smoothed[i] < smoothed[i+1] and 
                smoothed[i] < threshold):
                valleys.append(i / len(smoothed))
        
        return valleys
    
    def _determine_reading_order(self, elements: List[LayoutElement]) -> List[str]:
        """
        Determine the natural reading order of elements.
        
        Uses a combination of top-to-bottom and left-to-right ordering,
        accounting for multi-column layouts.
        
        Args:
            elements: List of document elements
            
        Returns:
            List of element IDs in reading order
        """
        # Group elements by page
        elements_by_page = {}
        for elem in elements:
            if elem.page not in elements_by_page:
                elements_by_page[elem.page] = []
            elements_by_page[elem.page].append(elem)
        
        reading_order = []
        
        # Process each page in order
        for page in sorted(elements_by_page.keys()):
            page_elements = elements_by_page[page]
            
            # If columns are enabled, use column-aware reading order
            if self.detect_columns:
                columns = self._detect_columns(page_elements)
                
                # Process each column in left-to-right order
                for column in columns:
                    # Get elements in this column
                    column_elements = [elem for elem in page_elements if elem.id in column]
                    
                    # Sort elements in this column top-to-bottom
                    sorted_elements = sorted(column_elements, key=lambda e: e.y1)
                    
                    # Add to reading order
                    reading_order.extend([elem.id for elem in sorted_elements])
            else:
                # Simple top-to-bottom, left-to-right ordering
                sorted_elements = sorted(page_elements, key=lambda e: (e.y1, e.x1))
                reading_order.extend([elem.id for elem in sorted_elements])
        
        return reading_order
    
    def _detect_logical_sections(self, elements: List[LayoutElement]) -> Dict[str, List[str]]:
        """
        Group elements into logical sections based on layout.
        
        Uses spatial clustering and size/type information to group elements.
        
        Args:
            elements: List of document elements
            
        Returns:
            Dictionary mapping section IDs to lists of element IDs
        """
        # Group elements by page
        elements_by_page = {}
        for elem in elements:
            if elem.page not in elements_by_page:
                elements_by_page[elem.page] = []
            elements_by_page[elem.page].append(elem)
        
        all_sections = {}
        section_counter = 0
        
        # Process each page
        for page, page_elements in elements_by_page.items():
            # Find potential section headers (larger text elements)
            headers = [elem for elem in page_elements 
                      if elem.element_type == "text" and 
                      (elem.y2 - elem.y1) > 0.02]  # Larger than average text
            
            # If no headers found, treat whole page as one section
            if not headers:
                section_id = f"section_{section_counter}"
                all_sections[section_id] = [elem.id for elem in page_elements]
                section_counter += 1
                continue
            
            # Sort headers by y-position
            headers = sorted(headers, key=lambda e: e.y1)
            
            # Create sections based on headers
            for i, header in enumerate(headers):
                section_id = f"section_{section_counter}"
                section_counter += 1
                
                # Section includes header
                section_elements = [header.id]
                
                # Find y-range for this section
                y_start = header.y2
                y_end = headers[i+1].y1 if i < len(headers) - 1 else 1.0
                
                # Add elements in this y-range to section
                for elem in page_elements:
                    if elem.id != header.id and y_start <= elem.y1 < y_end:
                        section_elements.append(elem.id)
                
                all_sections[section_id] = section_elements
        
        return all_sections
    
    def _analyze_page_structure(self, elements: List[LayoutElement]) -> Dict[int, Dict]:
        """
        Analyze the structure of each page.
        
        Detects headers, footers, margins, and content areas.
        
        Args:
            elements: List of document elements
            
        Returns:
            Dictionary mapping page numbers to page structure information
        """
        # Group elements by page
        elements_by_page = {}
        for elem in elements:
            if elem.page not in elements_by_page:
                elements_by_page[elem.page] = []
            elements_by_page[elem.page].append(elem)
        
        page_structures = {}
        
        # Process each page
        for page, page_elements in elements_by_page.items():
            # Skip if no elements
            if not page_elements:
                continue
                
            # Find element boundaries
            min_x = min(elem.x1 for elem in page_elements)
            min_y = min(elem.y1 for elem in page_elements)
            max_x = max(elem.x2 for elem in page_elements)
            max_y = max(elem.y2 for elem in page_elements)
            
            # Detect header (elements at the top of the page)
            header_height = 0.1  # Top 10% of page
            header_elements = [elem.id for elem in page_elements if elem.y1 < header_height]
            
            # Detect footer (elements at the bottom of the page)
            footer_start = 0.9  # Bottom 10% of page
            footer_elements = [elem.id for elem in page_elements if elem.y2 > footer_start]
            
            # Detect margins
            left_margin = min_x
            right_margin = 1.0 - max_x
            top_margin = min_y
            bottom_margin = 1.0 - max_y
            
            # Create page structure
            page_structures[page] = {
                "header_elements": header_elements,
                "footer_elements": footer_elements,
                "margins": {
                    "left": left_margin,
                    "right": right_margin,
                    "top": top_margin,
                    "bottom": bottom_margin
                },
                "content_area": {
                    "x1": min_x,
                    "y1": min_y,
                    "x2": max_x,
                    "y2": max_y
                }
            }
        
        return page_structures
