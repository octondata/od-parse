"""
Form Understanding module for extracting and interpreting form elements from PDFs.

This module provides advanced capabilities for recognizing and understanding
form elements such as checkboxes, radio buttons, text fields, and signatures.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class FormField:
    """Represents a form field within a document."""
    id: str
    field_type: str  # checkbox, radio, text, signature, dropdown
    x1: float
    y1: float
    x2: float
    y2: float
    page: int
    label: Optional[str] = None
    value: Optional[Any] = None
    options: Optional[List[str]] = None
    is_checked: Optional[bool] = None
    confidence: float = 1.0
    group_id: Optional[str] = None


class FormUnderstanding:
    """
    Advanced class for detecting and extracting form elements from PDFs.
    
    This class implements sophisticated algorithms for identifying form fields
    such as checkboxes, radio buttons, text fields, and signatures, along with
    their labels and values.
    """
    
    def __init__(self, 
                 detect_checkboxes=True,
                 detect_radio_buttons=True,
                 detect_text_fields=True,
                 detect_signatures=True,
                 detect_dropdowns=True,
                 ocr_enabled=True):
        """
        Initialize the FormUnderstanding engine.
        
        Args:
            detect_checkboxes: Whether to detect checkbox fields
            detect_radio_buttons: Whether to detect radio button groups
            detect_text_fields: Whether to detect text input fields
            detect_signatures: Whether to detect signature fields
            detect_dropdowns: Whether to detect dropdown fields
            ocr_enabled: Whether to use OCR for field detection
        """
        self.detect_checkboxes = detect_checkboxes
        self.detect_radio_buttons = detect_radio_buttons
        self.detect_text_fields = detect_text_fields
        self.detect_signatures = detect_signatures
        self.detect_dropdowns = detect_dropdowns
        self.ocr_enabled = ocr_enabled
        
    def extract_forms(self, pdf_data: Dict) -> Dict[str, Any]:
        """
        Extract and analyze form elements from a PDF.
        
        Args:
            pdf_data: Parsed PDF data including text, images, and content
            
        Returns:
            Dictionary containing form fields and their attributes
        """
        # Extract raw form fields
        fields = []
        
        # Process based on enabled field types
        if self.detect_checkboxes:
            checkbox_fields = self._detect_checkboxes(pdf_data)
            fields.extend(checkbox_fields)
            
        if self.detect_radio_buttons:
            radio_fields = self._detect_radio_buttons(pdf_data)
            fields.extend(radio_fields)
            
        if self.detect_text_fields:
            text_fields = self._detect_text_fields(pdf_data)
            fields.extend(text_fields)
            
        if self.detect_signatures:
            signature_fields = self._detect_signatures(pdf_data)
            fields.extend(signature_fields)
            
        if self.detect_dropdowns:
            dropdown_fields = self._detect_dropdowns(pdf_data)
            fields.extend(dropdown_fields)
        
        # Match fields with labels
        fields_with_labels = self._match_fields_with_labels(fields, pdf_data)
        
        # Group related fields (e.g., radio button groups)
        grouped_fields = self._group_related_fields(fields_with_labels)
        
        # Extract field values
        fields_with_values = self._extract_field_values(grouped_fields, pdf_data)
        
        return {
            "fields": [self._field_to_dict(field) for field in fields_with_values],
            "groups": self._get_field_groups(fields_with_values),
            "form_structure": self._analyze_form_structure(fields_with_values)
        }
    
    def _field_to_dict(self, field: FormField) -> Dict:
        """Convert a FormField to a dictionary."""
        field_dict = {
            "id": field.id,
            "type": field.field_type,
            "bbox": [field.x1, field.y1, field.x2, field.y2],
            "page": field.page,
            "confidence": field.confidence
        }
        
        if field.label is not None:
            field_dict["label"] = field.label
            
        if field.value is not None:
            field_dict["value"] = field.value
            
        if field.options is not None:
            field_dict["options"] = field.options
            
        if field.is_checked is not None:
            field_dict["is_checked"] = field.is_checked
            
        if field.group_id is not None:
            field_dict["group_id"] = field.group_id
            
        return field_dict
    
    def _detect_checkboxes(self, pdf_data: Dict) -> List[FormField]:
        """
        Detect checkbox fields in a PDF.
        
        Uses image processing and pattern matching to identify checkboxes.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected checkbox fields
        """
        checkboxes = []
        field_id_counter = 0
        
        # Look in annotations if available (for true PDF forms)
        if "annotations" in pdf_data:
            for annot in pdf_data.get("annotations", []):
                if annot.get("subtype") == "Widget" and annot.get("field_type") == "Btn":
                    # This is likely a checkbox
                    x1, y1, x2, y2 = annot.get("rect", [0, 0, 0, 0])
                    page = annot.get("page", 0)
                    is_checked = annot.get("value") == "/Yes"
                    
                    checkbox = FormField(
                        id=f"checkbox_{field_id_counter}",
                        field_type="checkbox",
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page=page,
                        is_checked=is_checked,
                        confidence=0.9
                    )
                    
                    checkboxes.append(checkbox)
                    field_id_counter += 1
        
        # Use image processing to detect checkboxes in images
        if "images" in pdf_data and not checkboxes:
            for i, img_data in enumerate(pdf_data.get("images", [])):
                # In a real implementation, this would use computer vision to detect checkboxes
                # Here we'll use a placeholder that assumes some processing has identified potential checkboxes
                if "detected_checkboxes" in img_data:
                    for j, bbox in enumerate(img_data.get("detected_checkboxes", [])):
                        x1, y1, x2, y2 = bbox
                        page = img_data.get("page", 0)
                        
                        # Analyze the image to determine if checkbox is checked
                        is_checked = self._is_checkbox_checked(img_data, bbox)
                        
                        checkbox = FormField(
                            id=f"checkbox_{field_id_counter}",
                            field_type="checkbox",
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            page=page,
                            is_checked=is_checked,
                            confidence=0.7
                        )
                        
                        checkboxes.append(checkbox)
                        field_id_counter += 1
        
        # If no direct checkbox annotations or images, look for checkbox patterns in text
        if not checkboxes and "text_blocks" in pdf_data:
            for block in pdf_data.get("text_blocks", []):
                text = block.get("text", "")
                
                # Look for checkbox-like patterns in text (□, ☐, ☑, ✓, etc.)
                if "□" in text or "☐" in text or "[ ]" in text or "[x]" in text or "[X]" in text:
                    x1, y1, x2, y2 = block.get("bbox", [0, 0, 0, 0])
                    page = block.get("page", 0)
                    
                    # Determine if checked based on text content
                    is_checked = "☑" in text or "✓" in text or "[x]" in text or "[X]" in text
                    
                    checkbox = FormField(
                        id=f"checkbox_{field_id_counter}",
                        field_type="checkbox",
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page=page,
                        is_checked=is_checked,
                        confidence=0.6
                    )
                    
                    checkboxes.append(checkbox)
                    field_id_counter += 1
        
        return checkboxes
    
    def _is_checkbox_checked(self, img_data: Dict, bbox: List[float]) -> bool:
        """
        Determine if a checkbox is checked based on its image.
        
        Args:
            img_data: Image data
            bbox: Bounding box of the checkbox
            
        Returns:
            True if checkbox appears to be checked, False otherwise
        """
        # In a real implementation, this would analyze the image within the bbox
        # to determine if the checkbox contains a check mark or is filled
        
        # Placeholder logic - in reality would use computer vision
        # For example, check pixel density or use template matching
        
        # Simplistic approach: assume the checkbox is checked if it's in the examples
        return bbox in img_data.get("checked_examples", [])
    
    def _detect_radio_buttons(self, pdf_data: Dict) -> List[FormField]:
        """
        Detect radio button fields in a PDF.
        
        Groups related radio buttons together.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected radio button fields
        """
        radio_buttons = []
        field_id_counter = 0
        
        # Look in annotations if available (for true PDF forms)
        if "annotations" in pdf_data:
            # Group radio buttons by their field names
            radio_groups = {}
            
            for annot in pdf_data.get("annotations", []):
                if annot.get("subtype") == "Widget" and annot.get("field_type") == "Btn" and annot.get("radio", False):
                    # This is likely a radio button
                    x1, y1, x2, y2 = annot.get("rect", [0, 0, 0, 0])
                    page = annot.get("page", 0)
                    field_name = annot.get("field_name", f"unnamed_group_{field_id_counter}")
                    is_selected = annot.get("state") == "selected"
                    option_value = annot.get("option_value", "")
                    
                    # Add to radio group
                    if field_name not in radio_groups:
                        radio_groups[field_name] = []
                        
                    radio_groups[field_name].append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "page": page,
                        "is_selected": is_selected,
                        "option_value": option_value,
                        "option_index": len(radio_groups[field_name])
                    })
            
            # Create FormField objects for each radio button
            for group_name, buttons in radio_groups.items():
                for i, button in enumerate(buttons):
                    radio = FormField(
                        id=f"radio_{field_id_counter}",
                        field_type="radio",
                        x1=button["x1"],
                        y1=button["y1"],
                        x2=button["x2"],
                        y2=button["y2"],
                        page=button["page"],
                        is_checked=button["is_selected"],
                        group_id=group_name,
                        value=button["option_value"] if button["is_selected"] else None,
                        options=[b["option_value"] for b in buttons] if all("option_value" in b for b in buttons) else None,
                        confidence=0.9
                    )
                    
                    radio_buttons.append(radio)
                    field_id_counter += 1
        
        # Use image processing to detect radio buttons in images
        if "images" in pdf_data and not radio_buttons:
            # Similar approach to checkbox detection but for circular radio buttons
            # In a real implementation, this would use computer vision
            # Here we'll use a placeholder
            pass
        
        # If no direct radio button annotations or images, look for radio button patterns in text
        if not radio_buttons and "text_blocks" in pdf_data:
            # Look for patterns like "○" or "●" in text blocks
            # Group nearby radio buttons together
            pass
        
        return radio_buttons
    
    def _detect_text_fields(self, pdf_data: Dict) -> List[FormField]:
        """
        Detect text input fields in a PDF.
        
        Identifies empty spaces for user input.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected text input fields
        """
        text_fields = []
        field_id_counter = 0
        
        # Look in annotations if available (for true PDF forms)
        if "annotations" in pdf_data:
            for annot in pdf_data.get("annotations", []):
                if annot.get("subtype") == "Widget" and annot.get("field_type") in ["Tx", "Ch"]:
                    # This is a text field or choice field
                    x1, y1, x2, y2 = annot.get("rect", [0, 0, 0, 0])
                    page = annot.get("page", 0)
                    value = annot.get("value", "")
                    field_name = annot.get("field_name", f"field_{field_id_counter}")
                    
                    field_type = "text" if annot.get("field_type") == "Tx" else "dropdown"
                    
                    text_field = FormField(
                        id=f"{field_type}_{field_id_counter}",
                        field_type=field_type,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page=page,
                        value=value,
                        label=field_name,
                        options=annot.get("options", []) if field_type == "dropdown" else None,
                        confidence=0.9
                    )
                    
                    text_fields.append(text_field)
                    field_id_counter += 1
        
        # Look for underlines, boxes, or horizontal lines that might indicate text fields
        if "lines" in pdf_data and not text_fields:
            for line in pdf_data.get("lines", []):
                # Check if line is horizontal and might be an underline for a text field
                x1, y1, x2, y2 = line.get("bbox", [0, 0, 0, 0])
                is_horizontal = abs(y2 - y1) < 0.01  # Very small vertical difference
                is_long_enough = (x2 - x1) > 0.1  # At least 10% of page width
                
                if is_horizontal and is_long_enough:
                    page = line.get("page", 0)
                    
                    # Create a text field that spans this line
                    text_field = FormField(
                        id=f"text_{field_id_counter}",
                        field_type="text",
                        x1=x1,
                        y1=y1 - 0.02,  # Add some space above the line for text
                        x2=x2,
                        y2=y1 + 0.005,  # Just past the line
                        page=page,
                        value="",  # Empty by default
                        confidence=0.7
                    )
                    
                    text_fields.append(text_field)
                    field_id_counter += 1
        
        # Look for text field patterns in text blocks
        if "text_blocks" in pdf_data and not text_fields:
            for block in pdf_data.get("text_blocks", []):
                text = block.get("text", "")
                
                # Look for patterns like "______" or "............" that might indicate a field
                if "___" in text or "..." in text or "   " in text:
                    x1, y1, x2, y2 = block.get("bbox", [0, 0, 0, 0])
                    page = block.get("page", 0)
                    
                    text_field = FormField(
                        id=f"text_{field_id_counter}",
                        field_type="text",
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page=page,
                        value="",  # Empty by default
                        confidence=0.6
                    )
                    
                    text_fields.append(text_field)
                    field_id_counter += 1
        
        return text_fields
    
    def _detect_signatures(self, pdf_data: Dict) -> List[FormField]:
        """
        Detect signature fields in a PDF.
        
        Identifies areas meant for signatures.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected signature fields
        """
        signature_fields = []
        field_id_counter = 0
        
        # Look in annotations if available (for true PDF forms)
        if "annotations" in pdf_data:
            for annot in pdf_data.get("annotations", []):
                if annot.get("subtype") == "Widget" and annot.get("field_type") == "Sig":
                    # This is a signature field
                    x1, y1, x2, y2 = annot.get("rect", [0, 0, 0, 0])
                    page = annot.get("page", 0)
                    
                    signature_field = FormField(
                        id=f"signature_{field_id_counter}",
                        field_type="signature",
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        page=page,
                        confidence=0.9
                    )
                    
                    signature_fields.append(signature_field)
                    field_id_counter += 1
        
        # Look for text blocks containing signature-related keywords
        if "text_blocks" in pdf_data:
            for block in pdf_data.get("text_blocks", []):
                text = block.get("text", "").lower()
                
                # Check for signature-related keywords
                if ("signature" in text or "sign here" in text or "sign" in text) and ("x" in text or "_" in text):
                    x1, y1, x2, y2 = block.get("bbox", [0, 0, 0, 0])
                    page = block.get("page", 0)
                    
                    # Create signature field below this text
                    signature_field = FormField(
                        id=f"signature_{field_id_counter}",
                        field_type="signature",
                        x1=x1,
                        y1=y2,  # Start just below the text
                        x2=x2,
                        y2=y2 + 0.05,  # Add enough space for a signature
                        page=page,
                        label=text.strip(),
                        confidence=0.7
                    )
                    
                    signature_fields.append(signature_field)
                    field_id_counter += 1
        
        return signature_fields
    
    def _detect_dropdowns(self, pdf_data: Dict) -> List[FormField]:
        """
        Detect dropdown fields in a PDF.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected dropdown fields
        """
        # Most dropdown fields would be detected in the text field detection
        # as they appear as choice fields in PDF annotations
        # This method would implement additional heuristics for detecting dropdowns
        # that aren't explicitly defined in the PDF annotations
        
        return []  # For now, rely on text field detection
    
    def _match_fields_with_labels(self, fields: List[FormField], pdf_data: Dict) -> List[FormField]:
        """
        Match form fields with their corresponding labels.
        
        Uses spatial proximity and text analysis to find labels.
        
        Args:
            fields: List of detected form fields
            pdf_data: Parsed PDF data
            
        Returns:
            Fields with labels attached
        """
        fields_with_labels = []
        
        # For each field, look for nearby text that might be a label
        for field in fields:
            # Skip if field already has a label
            if field.label is not None:
                fields_with_labels.append(field)
                continue
            
            # Look for text blocks near this field
            if "text_blocks" in pdf_data:
                closest_text = None
                min_distance = float('inf')
                
                for block in pdf_data.get("text_blocks", []):
                    # Skip if not on the same page
                    if block.get("page", 0) != field.page:
                        continue
                    
                    # Calculate distance between field and text block
                    block_x1, block_y1, block_x2, block_y2 = block.get("bbox", [0, 0, 0, 0])
                    
                    # Check if text is to the left or above the field
                    is_left = block_x2 < field.x1
                    is_above = block_y2 < field.y1
                    
                    if is_left or is_above:
                        # Calculate center points
                        field_center_x = (field.x1 + field.x2) / 2
                        field_center_y = (field.y1 + field.y2) / 2
                        block_center_x = (block_x1 + block_x2) / 2
                        block_center_y = (block_y1 + block_y2) / 2
                        
                        # Calculate Euclidean distance
                        distance = ((field_center_x - block_center_x) ** 2 + (field_center_y - block_center_y) ** 2) ** 0.5
                        
                        # Update closest text if this is closer
                        if distance < min_distance and distance < 0.2:  # Within 20% of the page size
                            min_distance = distance
                            closest_text = block.get("text", "").strip()
            
                # If we found a closest text, use it as the label
                if closest_text:
                    # Clean up label (remove trailing colons, etc.)
                    label = closest_text.rstrip(':').rstrip()
                    field = FormField(
                        id=field.id,
                        field_type=field.field_type,
                        x1=field.x1,
                        y1=field.y1,
                        x2=field.x2,
                        y2=field.y2,
                        page=field.page,
                        label=label,
                        value=field.value,
                        options=field.options,
                        is_checked=field.is_checked,
                        confidence=field.confidence,
                        group_id=field.group_id
                    )
            
            fields_with_labels.append(field)
        
        return fields_with_labels
    
    def _group_related_fields(self, fields: List[FormField]) -> List[FormField]:
        """
        Group related form fields together.
        
        Identifies fields that belong to the same logical group.
        
        Args:
            fields: List of form fields
            
        Returns:
            Fields with group information
        """
        # This is mainly for radio buttons and checkboxes that might be related
        
        # Radio buttons should already be grouped
        
        # Look for checkbox groups with similar labels
        label_groups = {}
        
        for field in fields:
            if field.field_type == "checkbox" and field.label:
                # Extract the main part of the label (before any specific option text)
                label_parts = field.label.split('[')
                main_label = label_parts[0].strip()
                
                # Add to group
                if main_label not in label_groups:
                    label_groups[main_label] = []
                    
                label_groups[main_label].append(field)
        
        # Assign group IDs to checkbox groups
        grouped_fields = []
        
        for field in fields:
            # If this field is part of a checkbox group with multiple members, add group ID
            if field.field_type == "checkbox" and field.label:
                label_parts = field.label.split('[')
                main_label = label_parts[0].strip()
                
                if main_label in label_groups and len(label_groups[main_label]) > 1:
                    # This is part of a group with multiple checkboxes
                    field = FormField(
                        id=field.id,
                        field_type=field.field_type,
                        x1=field.x1,
                        y1=field.y1,
                        x2=field.x2,
                        y2=field.y2,
                        page=field.page,
                        label=field.label,
                        value=field.value,
                        options=field.options,
                        is_checked=field.is_checked,
                        confidence=field.confidence,
                        group_id=f"checkbox_group_{main_label}"
                    )
            
            grouped_fields.append(field)
        
        return grouped_fields
    
    def _extract_field_values(self, fields: List[FormField], pdf_data: Dict) -> List[FormField]:
        """
        Extract values from form fields.
        
        For fields like checkboxes and radio buttons, determines if they are checked.
        For text fields, extracts any filled-in text.
        
        Args:
            fields: List of form fields
            pdf_data: Parsed PDF data
            
        Returns:
            Fields with extracted values
        """
        # Most values should already be extracted during detection
        # This method would implement additional processing for cases
        # where values weren't extracted during detection
        
        return fields
    
    def _get_field_groups(self, fields: List[FormField]) -> Dict[str, List[str]]:
        """
        Extract field groups from the form fields.
        
        Args:
            fields: List of form fields
            
        Returns:
            Dictionary mapping group IDs to lists of field IDs
        """
        groups = {}
        
        for field in fields:
            if field.group_id:
                if field.group_id not in groups:
                    groups[field.group_id] = []
                    
                groups[field.group_id].append(field.id)
        
        return groups
    
    def _analyze_form_structure(self, fields: List[FormField]) -> Dict[str, Any]:
        """
        Analyze the overall structure of the form.
        
        Args:
            fields: List of form fields
            
        Returns:
            Dictionary containing form structure information
        """
        # Count fields by type
        field_counts = {}
        for field in fields:
            if field.field_type not in field_counts:
                field_counts[field.field_type] = 0
                
            field_counts[field.field_type] += 1
        
        # Check if form has a signature field
        has_signature = any(field.field_type == "signature" for field in fields)
        
        # Analyze field density by page
        fields_by_page = {}
        for field in fields:
            if field.page not in fields_by_page:
                fields_by_page[field.page] = 0
                
            fields_by_page[field.page] += 1
        
        # Determine form complexity
        if len(fields) < 5:
            complexity = "simple"
        elif len(fields) < 20:
            complexity = "moderate"
        else:
            complexity = "complex"
        
        return {
            "field_counts": field_counts,
            "has_signature": has_signature,
            "fields_by_page": fields_by_page,
            "total_fields": len(fields),
            "complexity": complexity
        }
