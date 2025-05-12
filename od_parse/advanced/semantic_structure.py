"""
Semantic Structure Extraction module for understanding document hierarchy and content relationships.

This module provides advanced capabilities for extracting semantic structure from documents,
including section hierarchies, references, and content relationships.
"""

from typing import Dict, List, Any, Tuple, Optional
import re
from dataclasses import dataclass


@dataclass
class SemanticElement:
    """Represents a semantic element within a document."""
    id: str
    element_type: str  # heading, paragraph, list_item, reference, etc.
    level: int  # Heading level, list nesting level, etc.
    text: str
    page: int
    x1: float
    y1: float
    x2: float
    y2: float
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    attributes: Dict[str, Any] = None
    confidence: float = 1.0


class SemanticStructureExtractor:
    """
    Advanced class for extracting semantic structure from documents.
    
    This class implements sophisticated algorithms for understanding document structure,
    including section hierarchies, cross-references, and content relationships.
    """
    
    def __init__(self, 
                 detect_headings=True,
                 detect_lists=True,
                 detect_references=True,
                 detect_toc=True,
                 detect_footnotes=True,
                 max_heading_levels=6):
        """
        Initialize the SemanticStructureExtractor.
        
        Args:
            detect_headings: Whether to detect section headings
            detect_lists: Whether to detect bulleted and numbered lists
            detect_references: Whether to detect cross-references
            detect_toc: Whether to detect table of contents
            detect_footnotes: Whether to detect footnotes
            max_heading_levels: Maximum number of heading levels to detect
        """
        self.detect_headings = detect_headings
        self.detect_lists = detect_lists
        self.detect_references = detect_references
        self.detect_toc = detect_toc
        self.detect_footnotes = detect_footnotes
        self.max_heading_levels = max_heading_levels
        
    def extract_structure(self, pdf_data: Dict) -> Dict[str, Any]:
        """
        Extract semantic structure from a document.
        
        Args:
            pdf_data: Parsed PDF data including text, images, and content
            
        Returns:
            Dictionary containing extracted semantic structure
        """
        semantic_elements = []
        element_id_counter = 0
        
        # Process text blocks to extract semantic elements
        if "text_blocks" in pdf_data:
            # First pass: identify individual semantic elements
            for block in pdf_data.get("text_blocks", []):
                # Extract basic information
                text = block.get("text", "")
                page = block.get("page", 0)
                bbox = block.get("bbox", [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                
                # Detect headings
                if self.detect_headings:
                    heading_level = self._detect_heading_level(block, pdf_data)
                    if heading_level > 0:
                        element = SemanticElement(
                            id=f"heading_{element_id_counter}",
                            element_type="heading",
                            level=heading_level,
                            text=text,
                            page=page,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            children_ids=[],
                            attributes={"heading_level": heading_level},
                            confidence=0.9
                        )
                        semantic_elements.append(element)
                        element_id_counter += 1
                        continue
                
                # Detect list items
                if self.detect_lists:
                    list_info = self._detect_list_item(block, pdf_data)
                    if list_info["is_list_item"]:
                        element = SemanticElement(
                            id=f"list_item_{element_id_counter}",
                            element_type="list_item",
                            level=list_info["level"],
                            text=text,
                            page=page,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            children_ids=[],
                            attributes={
                                "list_type": list_info["list_type"],
                                "marker": list_info["marker"]
                            },
                            confidence=0.8
                        )
                        semantic_elements.append(element)
                        element_id_counter += 1
                        continue
                
                # Detect footnotes
                if self.detect_footnotes:
                    footnote_info = self._detect_footnote(block, pdf_data)
                    if footnote_info["is_footnote"]:
                        element = SemanticElement(
                            id=f"footnote_{element_id_counter}",
                            element_type="footnote",
                            level=0,
                            text=text,
                            page=page,
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            children_ids=[],
                            attributes={
                                "footnote_number": footnote_info["number"]
                            },
                            confidence=0.8
                        )
                        semantic_elements.append(element)
                        element_id_counter += 1
                        continue
                
                # Default to paragraph
                element = SemanticElement(
                    id=f"paragraph_{element_id_counter}",
                    element_type="paragraph",
                    level=0,
                    text=text,
                    page=page,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    children_ids=[],
                    confidence=0.7
                )
                semantic_elements.append(element)
                element_id_counter += 1
        
        # Second pass: establish hierarchy
        elements_with_hierarchy = self._establish_hierarchy(semantic_elements)
        
        # Third pass: detect references between elements
        if self.detect_references:
            elements_with_references = self._detect_references(elements_with_hierarchy, pdf_data)
        else:
            elements_with_references = elements_with_hierarchy
        
        # Fourth pass: detect table of contents
        if self.detect_toc:
            toc_info = self._detect_toc(elements_with_references, pdf_data)
        else:
            toc_info = None
        
        # Convert to dictionary format
        result = {
            "elements": [self._element_to_dict(elem) for elem in elements_with_references],
            "hierarchy": self._hierarchy_to_dict(elements_with_references)
        }
        
        if toc_info:
            result["toc"] = toc_info
        
        return result
    
    def _element_to_dict(self, element: SemanticElement) -> Dict:
        """Convert a SemanticElement to a dictionary."""
        element_dict = {
            "id": element.id,
            "type": element.element_type,
            "level": element.level,
            "text": element.text,
            "page": element.page,
            "bbox": [element.x1, element.y1, element.x2, element.y2],
            "confidence": element.confidence
        }
        
        if element.parent_id:
            element_dict["parent_id"] = element.parent_id
            
        if element.children_ids:
            element_dict["children_ids"] = element.children_ids
            
        if element.attributes:
            element_dict["attributes"] = element.attributes
            
        return element_dict
    
    def _detect_heading_level(self, block: Dict, pdf_data: Dict) -> int:
        """
        Detect if a text block is a heading and determine its level.
        
        Uses font size, style, and content patterns to identify headings.
        
        Args:
            block: Text block to analyze
            pdf_data: Full PDF data
            
        Returns:
            Heading level (1-6) or 0 if not a heading
        """
        text = block.get("text", "")
        font_size = block.get("font_size", 0)
        is_bold = block.get("is_bold", False)
        
        # Skip empty text
        if not text.strip():
            return 0
        
        # Look for numbered headings like "1.2.3 Heading Text"
        numbered_heading_pattern = r"^(\d+\.)+\d*\s+\w"
        if re.match(numbered_heading_pattern, text):
            # Count the number of dots to determine level
            level = text.count('.') 
            return min(level, self.max_heading_levels)
        
        # Use font size to determine heading level
        # This is a simplified approach - in a real implementation,
        # we would analyze the distribution of font sizes in the document
        
        # If we have font information
        if font_size > 0:
            # Simple heuristic - the larger the font, the lower the heading level
            if font_size > 18 and is_bold:
                return 1
            elif font_size > 16 and is_bold:
                return 2
            elif font_size > 14:
                return 3
            elif font_size > 12 and is_bold:
                return 4
            elif is_bold:
                return 5
        
        # Check if the text is short and followed by more text
        # This is a common pattern for headings
        if len(text) < 100 and text.strip().endswith((':','.')) and is_bold:
            return 6
        
        return 0
    
    def _detect_list_item(self, block: Dict, pdf_data: Dict) -> Dict[str, Any]:
        """
        Detect if a text block is a list item.
        
        Identifies bulleted and numbered lists.
        
        Args:
            block: Text block to analyze
            pdf_data: Full PDF data
            
        Returns:
            Dictionary with list item information
        """
        text = block.get("text", "")
        
        # Default result
        result = {
            "is_list_item": False,
            "level": 0,
            "list_type": "",
            "marker": ""
        }
        
        # Skip empty text
        if not text.strip():
            return result
        
        # Check for bullet points
        bullet_markers = ["•", "◦", "▪", "▫", "⁃", "⦿", "⦾", "⟡", "⟢", "⟣", "⟤", "⟥"]
        for bullet in bullet_markers:
            if text.startswith(bullet):
                result["is_list_item"] = True
                result["list_type"] = "bulleted"
                result["marker"] = bullet
                result["level"] = text.find(bullet) // 2  # Estimate level based on indentation
                return result
        
        # Check for dash bullets
        if text.startswith("-") or text.startswith("–") or text.startswith("—"):
            marker = text[0]
            result["is_list_item"] = True
            result["list_type"] = "bulleted"
            result["marker"] = marker
            result["level"] = text.find(marker) // 2  # Estimate level based on indentation
            return result
        
        # Check for numbered lists (various formats)
        # 1. Item text
        numbered_pattern1 = r"^\s*(\d+)\.\s+"
        # (a) Item text
        numbered_pattern2 = r"^\s*\(([a-zA-Z0-9]+)\)\s+"
        # a) Item text
        numbered_pattern3 = r"^\s*([a-zA-Z0-9]+)\)\s+"
        # a. Item text
        numbered_pattern4 = r"^\s*([a-zA-Z0-9]+)\.\s+"
        
        for pattern in [numbered_pattern1, numbered_pattern2, numbered_pattern3, numbered_pattern4]:
            match = re.match(pattern, text)
            if match:
                result["is_list_item"] = True
                result["list_type"] = "numbered"
                result["marker"] = match.group(0).strip()
                result["level"] = text.find(match.group(0)[0]) // 2  # Estimate level based on indentation
                return result
        
        return result
    
    def _detect_footnote(self, block: Dict, pdf_data: Dict) -> Dict[str, Any]:
        """
        Detect if a text block is a footnote.
        
        Args:
            block: Text block to analyze
            pdf_data: Full PDF data
            
        Returns:
            Dictionary with footnote information
        """
        text = block.get("text", "")
        y_position = block.get("bbox", [0, 0, 0, 0])[1]  # y1 coordinate
        page = block.get("page", 0)
        
        # Default result
        result = {
            "is_footnote": False,
            "number": None
        }
        
        # Skip empty text
        if not text.strip():
            return result
        
        # Check if block is at the bottom of the page
        # In a real implementation, we would compare to the page dimensions
        if y_position > 0.8:  # Assume bottom 20% of page
            # Check for footnote markers at the beginning
            footnote_pattern = r"^[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§]|\[\d+\]|\(\d+\)|^\d+\.|\*+"
            if re.match(footnote_pattern, text):
                result["is_footnote"] = True
                # Extract the footnote number or marker
                marker = re.match(footnote_pattern, text).group(0)
                result["number"] = marker
                return result
            
            # Check for superscript numbers at the beginning
            superscript_pattern = r"^\\u207[0-9]|^\\u208[0-9]"  # Unicode superscript/subscript digits
            if re.match(superscript_pattern, text):
                result["is_footnote"] = True
                marker = re.match(superscript_pattern, text).group(0)
                result["number"] = marker
                return result
        
        return result
    
    def _establish_hierarchy(self, elements: List[SemanticElement]) -> List[SemanticElement]:
        """
        Establish parent-child relationships between semantic elements.
        
        Uses heading levels and spatial relationships to determine hierarchy.
        
        Args:
            elements: List of semantic elements
            
        Returns:
            Elements with hierarchy information
        """
        # Sort elements by page and y-position
        sorted_elements = sorted(elements, key=lambda e: (e.page, e.y1))
        
        # Stack to keep track of current section hierarchy
        # Each item is (heading_level, element_id)
        section_stack = []
        
        # Process elements in order
        processed_elements = []
        
        for element in sorted_elements:
            # Clone the element to avoid modifying the original
            new_element = SemanticElement(
                id=element.id,
                element_type=element.element_type,
                level=element.level,
                text=element.text,
                page=element.page,
                x1=element.x1,
                y1=element.y1,
                x2=element.x2,
                y2=element.y2,
                parent_id=element.parent_id,
                children_ids=element.children_ids if element.children_ids else [],
                attributes=element.attributes,
                confidence=element.confidence
            )
            
            # Handle headings - they define new sections
            if element.element_type == "heading":
                heading_level = element.level
                
                # Pop elements from stack that have >= heading level
                while section_stack and section_stack[-1][0] >= heading_level:
                    section_stack.pop()
                
                # If there are parent sections, set parent ID
                if section_stack:
                    parent_id = section_stack[-1][1]
                    new_element.parent_id = parent_id
                    
                    # Add this element as child of parent
                    for i, e in enumerate(processed_elements):
                        if e.id == parent_id:
                            processed_elements[i].children_ids.append(new_element.id)
                            break
                
                # Add this heading to the stack
                section_stack.append((heading_level, new_element.id))
            
            # Handle list items
            elif element.element_type == "list_item":
                list_level = element.level
                
                # Find the most recent element that could be a parent
                parent_id = None
                
                # First look for a parent list item with one level less
                for e in reversed(processed_elements):
                    if e.element_type == "list_item" and e.level == list_level - 1:
                        parent_id = e.id
                        break
                
                # If no parent list item found, look for the current section heading
                if parent_id is None and section_stack:
                    parent_id = section_stack[-1][1]
                
                if parent_id:
                    new_element.parent_id = parent_id
                    
                    # Add this element as child of parent
                    for i, e in enumerate(processed_elements):
                        if e.id == parent_id:
                            processed_elements[i].children_ids.append(new_element.id)
                            break
            
            # Handle other elements - assign to current section
            else:
                if section_stack:
                    parent_id = section_stack[-1][1]
                    new_element.parent_id = parent_id
                    
                    # Add this element as child of parent
                    for i, e in enumerate(processed_elements):
                        if e.id == parent_id:
                            processed_elements[i].children_ids.append(new_element.id)
                            break
            
            processed_elements.append(new_element)
        
        return processed_elements
    
    def _detect_references(self, elements: List[SemanticElement], pdf_data: Dict) -> List[SemanticElement]:
        """
        Detect references between document elements.
        
        Identifies internal references like "See Section 3.2" or "Figure 1".
        
        Args:
            elements: List of semantic elements
            pdf_data: Full PDF data
            
        Returns:
            Elements with reference information
        """
        # Compile patterns for common reference types
        patterns = {
            "section_ref": r"[Ss]ection\s+(\d+(\.\d+)*)",
            "figure_ref": r"[Ff]igure\s+(\d+(\.\d+)*)",
            "table_ref": r"[Tt]able\s+(\d+(\.\d+)*)",
            "page_ref": r"page\s+(\d+)",
            "footnote_ref": r"[¹²³⁴⁵⁶⁷⁸⁹⁰*†‡§]|\[\d+\]|\(\d+\)"
        }
        
        # Create a mapping of reference targets
        # For example, "Section 3.2" -> element_id of section 3.2
        reference_targets = {}
        
        # First pass: identify potential reference targets
        for element in elements:
            # Sections can be referenced by number
            if element.element_type == "heading":
                # Check if the heading has a number
                match = re.match(r"^(\d+(\.\d+)*)\s+", element.text)
                if match:
                    section_number = match.group(1)
                    reference_targets[f"section_{section_number}"] = element.id
            
            # Tables can be referenced by caption
            if element.element_type == "paragraph" and element.text.lower().startswith("table "):
                match = re.match(r"[Tt]able\s+(\d+(\.\d+)*)", element.text)
                if match:
                    table_number = match.group(1)
                    reference_targets[f"table_{table_number}"] = element.id
            
            # Figures can be referenced by caption
            if element.element_type == "paragraph" and element.text.lower().startswith("figure "):
                match = re.match(r"[Ff]igure\s+(\d+(\.\d+)*)", element.text)
                if match:
                    figure_number = match.group(1)
                    reference_targets[f"figure_{figure_number}"] = element.id
        
        # Second pass: identify references
        processed_elements = []
        
        for element in elements:
            # Clone the element to avoid modifying the original
            new_element = SemanticElement(
                id=element.id,
                element_type=element.element_type,
                level=element.level,
                text=element.text,
                page=element.page,
                x1=element.x1,
                y1=element.y1,
                x2=element.x2,
                y2=element.y2,
                parent_id=element.parent_id,
                children_ids=element.children_ids,
                attributes=element.attributes.copy() if element.attributes else {},
                confidence=element.confidence
            )
            
            # Check for references in the text
            references = []
            
            for ref_type, pattern in patterns.items():
                for match in re.finditer(pattern, element.text):
                    # Extract the reference number
                    if ref_type in ["section_ref", "figure_ref", "table_ref"]:
                        ref_number = match.group(1)
                        ref_key = f"{ref_type.split('_')[0]}_{ref_number}"
                        
                        # Check if we have a target for this reference
                        if ref_key in reference_targets:
                            references.append({
                                "type": ref_type,
                                "text": match.group(0),
                                "target_id": reference_targets[ref_key]
                            })
            
            # Add references to attributes
            if references:
                if not new_element.attributes:
                    new_element.attributes = {}
                new_element.attributes["references"] = references
            
            processed_elements.append(new_element)
        
        return processed_elements
    
    def _detect_toc(self, elements: List[SemanticElement], pdf_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Detect table of contents.
        
        Identifies ToC entries and links them to corresponding sections.
        
        Args:
            elements: List of semantic elements
            pdf_data: Full PDF data
            
        Returns:
            Dictionary with ToC information or None if no ToC found
        """
        # Look for a heading that might indicate a table of contents
        toc_heading = None
        for element in elements:
            if element.element_type == "heading" and any(phrase in element.text.lower() for phrase in ["content", "table of content", "toc"]):
                toc_heading = element
                break
        
        if not toc_heading:
            return None
        
        # Find elements that might be ToC entries
        # These typically follow the ToC heading and precede the first real section
        toc_entries = []
        in_toc = False
        
        for element in elements:
            # Skip until we find the ToC heading
            if not in_toc and element.id != toc_heading.id:
                continue
            
            # Start collecting ToC entries after the heading
            if element.id == toc_heading.id:
                in_toc = True
                continue
            
            # Stop when we hit the first regular section heading
            if element.element_type == "heading" and element.level <= 2:
                break
            
            # Look for ToC entry patterns
            # Typically, these have section numbers, titles, and page numbers
            text = element.text
            
            # Pattern: "3.2 Section Title...............45"
            toc_pattern = r"^(\d+(\.\d+)*)\s+(.*?)\.{2,}\s*(\d+)$"
            match = re.match(toc_pattern, text)
            
            if match:
                section_number = match.group(1)
                section_title = match.group(3).strip()
                page_number = match.group(4)
                
                # Look for a heading with this number and title
                target_id = None
                for e in elements:
                    if e.element_type == "heading" and e.text.startswith(f"{section_number} {section_title}"):
                        target_id = e.id
                        break
                
                toc_entries.append({
                    "number": section_number,
                    "title": section_title,
                    "page": int(page_number),
                    "target_id": target_id
                })
        
        if not toc_entries:
            return None
        
        return {
            "heading_id": toc_heading.id,
            "entries": toc_entries
        }
    
    def _hierarchy_to_dict(self, elements: List[SemanticElement]) -> Dict[str, List[str]]:
        """
        Convert element hierarchy to a dictionary representation.
        
        Args:
            elements: List of semantic elements
            
        Returns:
            Dictionary mapping parent IDs to lists of child IDs
        """
        hierarchy = {}
        
        # Find root elements (those without a parent)
        root_elements = [elem for elem in elements if elem.parent_id is None]
        hierarchy["root"] = [elem.id for elem in root_elements]
        
        # Add children for each element
        for element in elements:
            if element.children_ids:
                hierarchy[element.id] = element.children_ids
        
        return hierarchy
