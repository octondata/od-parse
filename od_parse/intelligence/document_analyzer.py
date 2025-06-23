"""
Intelligent Document Analyzer

This module provides advanced document understanding and structured extraction
for complex documents like tax forms, contracts, invoices, etc.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class DocumentType(Enum):
    """Document type classification."""
    TAX_FORM_1040 = "tax_form_1040"
    TAX_FORM_W2 = "tax_form_w2"
    INVOICE = "invoice"
    CONTRACT = "contract"
    BANK_STATEMENT = "bank_statement"
    MEDICAL_RECORD = "medical_record"
    LEGAL_DOCUMENT = "legal_document"
    FINANCIAL_STATEMENT = "financial_statement"
    UNKNOWN = "unknown"


@dataclass
class FieldMapping:
    """Mapping for document fields."""
    field_name: str
    patterns: List[str]
    data_type: str
    required: bool = False
    validation_regex: Optional[str] = None


@dataclass
class StructuredField:
    """Structured field with metadata."""
    name: str
    value: Any
    confidence: float
    data_type: str
    source_location: Optional[str] = None
    validation_status: str = "unknown"


class DocumentAnalyzer:
    """
    Intelligent document analyzer that understands document structure
    and extracts meaningful, structured information.
    """
    
    def __init__(self):
        """Initialize the document analyzer."""
        self.logger = get_logger(__name__)
        self._load_field_mappings()
    
    def _load_field_mappings(self):
        """Load field mappings for different document types."""
        self.field_mappings = {
            DocumentType.TAX_FORM_1040: self._get_1040_field_mappings(),
            DocumentType.TAX_FORM_W2: self._get_w2_field_mappings(),
            DocumentType.INVOICE: self._get_invoice_field_mappings(),
            DocumentType.BANK_STATEMENT: self._get_bank_statement_mappings(),
        }
    
    def analyze_document(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a document and extract structured information.
        
        Args:
            parsed_data: Raw parsed data from PDF
            
        Returns:
            Structured document analysis
        """
        try:
            # Step 1: Classify document type
            doc_type = self._classify_document(parsed_data)
            
            # Step 2: Extract structured fields
            structured_fields = self._extract_structured_fields(parsed_data, doc_type)
            
            # Step 3: Analyze tables with context
            structured_tables = self._analyze_tables(parsed_data.get('tables', []), doc_type)
            
            # Step 4: Extract key-value pairs
            key_value_pairs = self._extract_key_value_pairs(parsed_data.get('text', ''))
            
            # Step 5: Validate extracted data
            validation_results = self._validate_extracted_data(structured_fields, doc_type)
            
            # Step 6: Generate document summary
            document_summary = self._generate_document_summary(
                structured_fields, structured_tables, doc_type
            )
            
            return {
                "document_intelligence": {
                    "document_type": doc_type.value,
                    "confidence": self._calculate_classification_confidence(parsed_data, doc_type),
                    "processing_method": "intelligent_analysis"
                },
                "structured_fields": {
                    field.name: {
                        "value": field.value,
                        "confidence": field.confidence,
                        "data_type": field.data_type,
                        "validation_status": field.validation_status,
                        "source_location": field.source_location
                    }
                    for field in structured_fields
                },
                "structured_tables": structured_tables,
                "key_value_pairs": key_value_pairs,
                "validation_results": validation_results,
                "document_summary": document_summary,
                "extraction_metadata": {
                    "total_fields_extracted": len(structured_fields),
                    "total_tables_analyzed": len(structured_tables),
                    "total_key_value_pairs": len(key_value_pairs),
                    "overall_confidence": self._calculate_overall_confidence(structured_fields)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Document analysis failed: {e}")
            return {
                "document_intelligence": {
                    "document_type": DocumentType.UNKNOWN.value,
                    "confidence": 0.0,
                    "processing_method": "fallback",
                    "error": str(e)
                }
            }
    
    def _classify_document(self, parsed_data: Dict[str, Any]) -> DocumentType:
        """Classify the document type based on content."""
        text = parsed_data.get('text', '').lower()
        
        # Tax form patterns
        if any(pattern in text for pattern in [
            'form 1040', 'u.s. individual income tax return', 'irs use only',
            'department of the treasury', 'internal revenue service'
        ]):
            return DocumentType.TAX_FORM_1040
        
        if any(pattern in text for pattern in [
            'form w-2', 'wage and tax statement', 'employer identification number'
        ]):
            return DocumentType.TAX_FORM_W2
        
        # Invoice patterns
        if any(pattern in text for pattern in [
            'invoice', 'bill to', 'invoice number', 'due date', 'amount due'
        ]):
            return DocumentType.INVOICE
        
        # Bank statement patterns
        if any(pattern in text for pattern in [
            'bank statement', 'account summary', 'beginning balance', 'ending balance'
        ]):
            return DocumentType.BANK_STATEMENT
        
        return DocumentType.UNKNOWN
    
    def _extract_structured_fields(
        self, 
        parsed_data: Dict[str, Any], 
        doc_type: DocumentType
    ) -> List[StructuredField]:
        """Extract structured fields based on document type."""
        structured_fields = []
        text = parsed_data.get('text', '')
        
        if doc_type not in self.field_mappings:
            return structured_fields
        
        field_mappings = self.field_mappings[doc_type]
        
        for field_mapping in field_mappings:
            value, confidence, location = self._extract_field_value(
                text, field_mapping
            )
            
            if value is not None:
                validation_status = self._validate_field_value(
                    value, field_mapping
                )
                
                structured_field = StructuredField(
                    name=field_mapping.field_name,
                    value=value,
                    confidence=confidence,
                    data_type=field_mapping.data_type,
                    source_location=location,
                    validation_status=validation_status
                )
                structured_fields.append(structured_field)
        
        return structured_fields
    
    def _extract_field_value(
        self, 
        text: str, 
        field_mapping: FieldMapping
    ) -> Tuple[Any, float, Optional[str]]:
        """Extract a specific field value from text."""
        for pattern in field_mapping.patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                if match.groups():
                    value = match.group(1).strip()
                    confidence = 0.9  # High confidence for regex matches
                    location = f"Position {match.start()}-{match.end()}"
                    
                    # Convert to appropriate data type
                    converted_value = self._convert_data_type(
                        value, field_mapping.data_type
                    )
                    
                    return converted_value, confidence, location
        
        return None, 0.0, None
    
    def _convert_data_type(self, value: str, data_type: str) -> Any:
        """Convert string value to appropriate data type."""
        try:
            if data_type == "float":
                # Remove commas and dollar signs
                cleaned = re.sub(r'[,$]', '', value)
                return float(cleaned)
            elif data_type == "int":
                cleaned = re.sub(r'[,]', '', value)
                return int(cleaned)
            elif data_type == "ssn":
                # Format SSN consistently
                digits = re.sub(r'[^\d]', '', value)
                if len(digits) == 9:
                    return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
            elif data_type == "date":
                # Standardize date format
                return self._standardize_date(value)
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    def _standardize_date(self, date_str: str) -> str:
        """Standardize date format."""
        # Simple date standardization - can be enhanced
        date_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',
            r'(\d{1,2})-(\d{1,2})-(\d{4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                month, day, year = match.groups()
                return f"{month.zfill(2)}/{day.zfill(2)}/{year}"
        
        return date_str
    
    def _validate_field_value(self, value: Any, field_mapping: FieldMapping) -> str:
        """Validate extracted field value."""
        if field_mapping.validation_regex:
            if isinstance(value, str) and re.match(field_mapping.validation_regex, value):
                return "valid"
            else:
                return "invalid"
        
        # Basic validation based on data type
        if field_mapping.data_type == "ssn":
            if isinstance(value, str) and re.match(r'^\d{3}-\d{2}-\d{4}$', value):
                return "valid"
            else:
                return "invalid"
        elif field_mapping.data_type in ["float", "int"]:
            if isinstance(value, (int, float)):
                return "valid"
            else:
                return "invalid"
        
        return "unknown"
    
    def _analyze_tables(self, tables: List[Dict], doc_type: DocumentType) -> List[Dict[str, Any]]:
        """Analyze tables with document context."""
        structured_tables = []
        
        for i, table in enumerate(tables):
            table_data = table.get('data', [])
            if not table_data:
                continue
            
            # Analyze table structure
            table_analysis = {
                "table_id": f"table_{i+1}",
                "table_type": self._classify_table_type(table_data, doc_type),
                "dimensions": {
                    "rows": len(table_data),
                    "columns": len(table_data[0]) if table_data else 0
                },
                "headers": self._extract_table_headers(table_data),
                "structured_data": self._structure_table_data(table_data, doc_type),
                "confidence": table.get('confidence', 0.8),
                "summary": self._generate_table_summary(table_data)
            }
            
            structured_tables.append(table_analysis)
        
        return structured_tables
    
    def _classify_table_type(self, table_data: List[List], doc_type: DocumentType) -> str:
        """Classify the type of table based on content."""
        if not table_data:
            return "unknown"
        
        # Convert table to text for analysis
        table_text = ' '.join([' '.join(str(cell) for cell in row) for row in table_data]).lower()
        
        if doc_type == DocumentType.TAX_FORM_1040:
            if any(keyword in table_text for keyword in ['income', 'wages', 'salary']):
                return "income_table"
            elif any(keyword in table_text for keyword in ['deduction', 'expense']):
                return "deduction_table"
            elif any(keyword in table_text for keyword in ['tax', 'withholding']):
                return "tax_calculation_table"
        
        # Generic table types
        if any(keyword in table_text for keyword in ['amount', 'total', 'sum']):
            return "financial_table"
        elif any(keyword in table_text for keyword in ['date', 'time']):
            return "temporal_table"
        
        return "data_table"
    
    def _extract_table_headers(self, table_data: List[List]) -> List[str]:
        """Extract and clean table headers."""
        if not table_data:
            return []
        
        # Assume first row contains headers
        headers = []
        for cell in table_data[0]:
            if cell and str(cell).strip():
                headers.append(str(cell).strip())
            else:
                headers.append(f"Column_{len(headers)+1}")
        
        return headers
    
    def _structure_table_data(self, table_data: List[List], doc_type: DocumentType) -> List[Dict]:
        """Structure table data with proper field mapping."""
        if len(table_data) < 2:  # Need at least headers + 1 data row
            return []
        
        headers = self._extract_table_headers(table_data)
        structured_rows = []
        
        for row_data in table_data[1:]:  # Skip header row
            if not any(cell for cell in row_data):  # Skip empty rows
                continue
            
            row_dict = {}
            for i, cell in enumerate(row_data):
                header = headers[i] if i < len(headers) else f"Column_{i+1}"
                
                # Clean and type the cell value
                cleaned_value = self._clean_cell_value(cell)
                row_dict[header] = cleaned_value
            
            structured_rows.append(row_dict)
        
        return structured_rows
    
    def _clean_cell_value(self, cell: Any) -> Any:
        """Clean and type cell values."""
        if cell is None or cell == '':
            return None
        
        cell_str = str(cell).strip()
        
        # Try to convert to number
        if re.match(r'^-?\$?[\d,]+\.?\d*$', cell_str):
            try:
                # Remove currency symbols and commas
                cleaned = re.sub(r'[,$]', '', cell_str)
                if '.' in cleaned:
                    return float(cleaned)
                else:
                    return int(cleaned)
            except ValueError:
                pass
        
        return cell_str
    
    def _generate_table_summary(self, table_data: List[List]) -> Dict[str, Any]:
        """Generate a summary of table contents."""
        if not table_data:
            return {}
        
        total_cells = sum(len(row) for row in table_data)
        non_empty_cells = sum(1 for row in table_data for cell in row if cell and str(cell).strip())
        
        return {
            "total_rows": len(table_data),
            "total_columns": len(table_data[0]) if table_data else 0,
            "total_cells": total_cells,
            "non_empty_cells": non_empty_cells,
            "data_density": non_empty_cells / total_cells if total_cells > 0 else 0
        }
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from text using patterns."""
        key_value_pairs = {}
        
        # Common key-value patterns
        patterns = [
            r'([A-Za-z\s]+):\s*([^\n\r]+)',  # "Key: Value"
            r'([A-Za-z\s]+)\s+([0-9,.$]+)',  # "Key 123.45"
            r'([A-Za-z\s]{3,})\s*\.\s*\.\s*\.\s*([^\n\r]+)',  # "Key ... Value"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                key = match.group(1).strip()
                value = match.group(2).strip()
                
                # Filter out noise
                if len(key) > 2 and len(value) > 0 and len(key) < 100:
                    # Clean the value
                    cleaned_value = self._clean_cell_value(value)
                    key_value_pairs[key] = cleaned_value
        
        return key_value_pairs
    
    def _validate_extracted_data(
        self, 
        structured_fields: List[StructuredField], 
        doc_type: DocumentType
    ) -> Dict[str, Any]:
        """Validate the extracted structured data."""
        validation_results = {
            "total_fields": len(structured_fields),
            "valid_fields": 0,
            "invalid_fields": 0,
            "unknown_fields": 0,
            "validation_errors": [],
            "completeness_score": 0.0
        }
        
        for field in structured_fields:
            if field.validation_status == "valid":
                validation_results["valid_fields"] += 1
            elif field.validation_status == "invalid":
                validation_results["invalid_fields"] += 1
                validation_results["validation_errors"].append(
                    f"Field '{field.name}' has invalid value: {field.value}"
                )
            else:
                validation_results["unknown_fields"] += 1
        
        # Calculate completeness score
        if doc_type in self.field_mappings:
            required_fields = [
                fm for fm in self.field_mappings[doc_type] if fm.required
            ]
            found_required = sum(
                1 for field in structured_fields 
                if any(rf.field_name == field.name for rf in required_fields)
            )
            validation_results["completeness_score"] = (
                found_required / len(required_fields) if required_fields else 1.0
            )
        
        return validation_results
    
    def _generate_document_summary(
        self, 
        structured_fields: List[StructuredField],
        structured_tables: List[Dict],
        doc_type: DocumentType
    ) -> Dict[str, Any]:
        """Generate an intelligent summary of the document."""
        summary = {
            "document_type": doc_type.value,
            "key_information": {},
            "financial_summary": {},
            "data_quality": {},
            "recommendations": []
        }
        
        # Extract key information based on document type
        if doc_type == DocumentType.TAX_FORM_1040:
            summary["key_information"] = self._summarize_tax_form(structured_fields)
            summary["financial_summary"] = self._summarize_tax_financials(structured_fields)
        
        # Data quality assessment
        total_confidence = sum(field.confidence for field in structured_fields)
        avg_confidence = total_confidence / len(structured_fields) if structured_fields else 0
        
        summary["data_quality"] = {
            "average_confidence": avg_confidence,
            "total_fields_extracted": len(structured_fields),
            "total_tables_analyzed": len(structured_tables),
            "extraction_completeness": self._calculate_completeness(structured_fields, doc_type)
        }
        
        # Generate recommendations
        summary["recommendations"] = self._generate_recommendations(
            structured_fields, structured_tables, doc_type
        )
        
        return summary
    
    def _summarize_tax_form(self, structured_fields: List[StructuredField]) -> Dict[str, Any]:
        """Summarize tax form key information."""
        key_info = {}
        
        for field in structured_fields:
            if field.name in ["taxpayer_name", "spouse_name", "ssn", "filing_status"]:
                key_info[field.name] = field.value
        
        return key_info
    
    def _summarize_tax_financials(self, structured_fields: List[StructuredField]) -> Dict[str, Any]:
        """Summarize tax form financial information."""
        financials = {}
        
        for field in structured_fields:
            if field.name in ["total_income", "adjusted_gross_income", "taxable_income", 
                            "total_tax", "amount_owed", "refund_amount"]:
                financials[field.name] = field.value
        
        return financials
    
    def _calculate_completeness(
        self, 
        structured_fields: List[StructuredField], 
        doc_type: DocumentType
    ) -> float:
        """Calculate extraction completeness score."""
        if doc_type not in self.field_mappings:
            return 0.0
        
        expected_fields = len(self.field_mappings[doc_type])
        extracted_fields = len(structured_fields)
        
        return min(extracted_fields / expected_fields, 1.0) if expected_fields > 0 else 0.0
    
    def _generate_recommendations(
        self, 
        structured_fields: List[StructuredField],
        structured_tables: List[Dict],
        doc_type: DocumentType
    ) -> List[str]:
        """Generate recommendations for improving extraction."""
        recommendations = []
        
        # Check for low confidence fields
        low_confidence_fields = [
            field for field in structured_fields if field.confidence < 0.7
        ]
        if low_confidence_fields:
            recommendations.append(
                f"Review {len(low_confidence_fields)} fields with low confidence scores"
            )
        
        # Check for validation errors
        invalid_fields = [
            field for field in structured_fields if field.validation_status == "invalid"
        ]
        if invalid_fields:
            recommendations.append(
                f"Correct {len(invalid_fields)} fields with validation errors"
            )
        
        # Check completeness
        completeness = self._calculate_completeness(structured_fields, doc_type)
        if completeness < 0.8:
            recommendations.append(
                "Consider manual review - extraction completeness is below 80%"
            )
        
        return recommendations
    
    def _calculate_overall_confidence(self, structured_fields: List[StructuredField]) -> float:
        """Calculate overall confidence score."""
        if not structured_fields:
            return 0.0
        
        total_confidence = sum(field.confidence for field in structured_fields)
        return total_confidence / len(structured_fields)
    
    def _calculate_classification_confidence(
        self, 
        parsed_data: Dict[str, Any], 
        doc_type: DocumentType
    ) -> float:
        """Calculate confidence in document classification."""
        # Simple confidence based on keyword matches
        text = parsed_data.get('text', '').lower()
        
        if doc_type == DocumentType.TAX_FORM_1040:
            keywords = ['form 1040', 'internal revenue service', 'tax return']
            matches = sum(1 for keyword in keywords if keyword in text)
            return min(matches / len(keywords), 1.0)
        
        return 0.5  # Default confidence
    
    def _get_1040_field_mappings(self) -> List[FieldMapping]:
        """Get field mappings for Form 1040."""
        return [
            FieldMapping(
                field_name="taxpayer_name",
                patterns=[
                    r"Your first name and middle initial\s+([A-Za-z\s]+)",
                    r"Name\(s\) shown on return\s+([A-Za-z\s]+)"
                ],
                data_type="string",
                required=True
            ),
            FieldMapping(
                field_name="ssn",
                patterns=[
                    r"Your social security number\s+(\d{3}[-\s]?\d{2}[-\s]?\d{4})",
                    r"SSN\s+(\d{3}[-\s]?\d{2}[-\s]?\d{4})"
                ],
                data_type="ssn",
                required=True,
                validation_regex=r"^\d{3}-\d{2}-\d{4}$"
            ),
            FieldMapping(
                field_name="filing_status",
                patterns=[
                    r"Filing Status.*?(Single|Married filing jointly|Married filing separately|Head of household|Qualifying surviving spouse)"
                ],
                data_type="string",
                required=True
            ),
            FieldMapping(
                field_name="total_income",
                patterns=[
                    r"Total income.*?(\d{1,3}(?:,\d{3})*)",
                    r"Line 9.*?(\d{1,3}(?:,\d{3})*)"
                ],
                data_type="float",
                required=True
            ),
            FieldMapping(
                field_name="adjusted_gross_income",
                patterns=[
                    r"Adjusted gross income.*?(\d{1,3}(?:,\d{3})*)",
                    r"Line 11.*?(\d{1,3}(?:,\d{3})*)"
                ],
                data_type="float",
                required=True
            ),
            FieldMapping(
                field_name="taxable_income",
                patterns=[
                    r"Taxable income.*?(\d{1,3}(?:,\d{3})*)",
                    r"Line 15.*?(\d{1,3}(?:,\d{3})*)"
                ],
                data_type="float",
                required=True
            ),
            FieldMapping(
                field_name="total_tax",
                patterns=[
                    r"Total tax.*?(\d{1,3}(?:,\d{3})*)",
                    r"Line 24.*?(\d{1,3}(?:,\d{3})*)"
                ],
                data_type="float",
                required=True
            )
        ]
    
    def _get_w2_field_mappings(self) -> List[FieldMapping]:
        """Get field mappings for W-2 forms."""
        return [
            FieldMapping(
                field_name="employer_name",
                patterns=[r"Employer's name.*?([A-Za-z\s&,\.]+)"],
                data_type="string",
                required=True
            ),
            FieldMapping(
                field_name="employee_name",
                patterns=[r"Employee's name.*?([A-Za-z\s,\.]+)"],
                data_type="string",
                required=True
            ),
            FieldMapping(
                field_name="wages",
                patterns=[r"Wages, tips, other compensation.*?(\d{1,3}(?:,\d{3})*)"],
                data_type="float",
                required=True
            )
        ]
    
    def _get_invoice_field_mappings(self) -> List[FieldMapping]:
        """Get field mappings for invoices."""
        return [
            FieldMapping(
                field_name="invoice_number",
                patterns=[r"Invoice\s*#?\s*:?\s*([A-Za-z0-9-]+)"],
                data_type="string",
                required=True
            ),
            FieldMapping(
                field_name="total_amount",
                patterns=[r"Total.*?(\$?\d{1,3}(?:,\d{3})*\.?\d*)"],
                data_type="float",
                required=True
            )
        ]
    
    def _get_bank_statement_mappings(self) -> List[FieldMapping]:
        """Get field mappings for bank statements."""
        return [
            FieldMapping(
                field_name="account_number",
                patterns=[r"Account\s*#?\s*:?\s*([0-9-]+)"],
                data_type="string",
                required=True
            ),
            FieldMapping(
                field_name="beginning_balance",
                patterns=[r"Beginning balance.*?(\$?\d{1,3}(?:,\d{3})*\.?\d*)"],
                data_type="float",
                required=True
            )
        ]
