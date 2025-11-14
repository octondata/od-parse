"""
Text normalization utilities for fixing OCR spacing and formatting issues.

This module provides functions to normalize OCR-extracted text by:
- Removing extra spaces within words and numbers
- Fixing date formatting issues
- Normalizing document numbers and codes
- Cleaning up field values
"""

import re
from typing import Optional, Dict, Any
from datetime import datetime


def normalize_ocr_spacing(text: str) -> str:
    """
    Remove extra spaces within words, numbers, and dates that OCR often introduces.

    Examples:
        "L S" -> "LS"
        "2 D" -> "2D"
        "0 1 / 1 5 / 2 0 2 5" -> "01/15/2025"
        "S I N G A P O R E" -> "SINGAPORE"

    Args:
        text: Input text with OCR spacing issues

    Returns:
        Normalized text with extra spaces removed
    """
    if not text:
        return text

    # Remove spaces between single characters (common OCR issue)
    # Pattern: single char + space + single char (repeated)
    # "A B C D" -> "ABCD"
    normalized = re.sub(r"\b([A-Z])\s+(?=[A-Z]\b)", r"\1", text)

    # Remove spaces within numbers
    # "1 2 3 4" -> "1234"
    normalized = re.sub(r"(\d)\s+(?=\d)", r"\1", normalized)

    # Fix date patterns with spaces
    # "01 / 15 / 2025" -> "01/15/2025"
    # "01 - 15 - 2025" -> "01-15-2025"
    normalized = re.sub(
        r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2,4})", r"\1/\2/\3", normalized
    )
    normalized = re.sub(
        r"(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{2,4})", r"\1-\2-\3", normalized
    )

    # Fix alphanumeric codes with spaces
    # "A 1 2 3" -> "A123"
    normalized = re.sub(r"([A-Z])\s+(\d)", r"\1\2", normalized)
    normalized = re.sub(r"(\d)\s+([A-Z])", r"\1\2", normalized)

    return normalized.strip()


def extract_date(text: str, field_name: str = "") -> Optional[str]:
    """
    Extract and normalize dates from OCR text with various formats.

    Handles common OCR issues:
    - Extra spaces: "01 / 15 / 2025"
    - Missing leading zeros: "1/5/2025"
    - Various separators: "/" or "-"
    - Different formats: MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD

    Args:
        text: Input text containing a date
        field_name: Optional field name for context (e.g., "birth_date")

    Returns:
        Normalized date string in ISO format (YYYY-MM-DD) or None if no valid date found
    """
    if not text:
        return None

    # First normalize spacing
    normalized = normalize_ocr_spacing(text)

    # Try various date patterns
    date_patterns = [
        # MM/DD/YYYY or M/D/YYYY
        (r"(\d{1,2})/(\d{1,2})/(\d{4})", "mdy"),
        # DD-MM-YYYY or D-M-YYYY
        (r"(\d{1,2})-(\d{1,2})-(\d{4})", "dmy"),
        # YYYY-MM-DD or YYYY/MM/DD
        (r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", "ymd"),
        # MM/DD/YY or M/D/YY (2-digit year)
        (r"(\d{1,2})/(\d{1,2})/(\d{2})", "mdy_short"),
        # MMDDYYYY (no separators)
        (r"(\d{2})(\d{2})(\d{4})", "mdy_nosep"),
    ]

    for pattern, format_type in date_patterns:
        match = re.search(pattern, normalized)
        if match:
            try:
                if format_type == "mdy":
                    month, day, year = match.groups()
                    date_obj = datetime(int(year), int(month), int(day))
                elif format_type == "dmy":
                    day, month, year = match.groups()
                    date_obj = datetime(int(year), int(month), int(day))
                elif format_type == "ymd":
                    year, month, day = match.groups()
                    date_obj = datetime(int(year), int(month), int(day))
                elif format_type == "mdy_short":
                    month, day, year = match.groups()
                    # Assume 20xx for years 00-50, 19xx for 51-99
                    full_year = int(year)
                    if full_year <= 50:
                        full_year += 2000
                    else:
                        full_year += 1900
                    date_obj = datetime(full_year, int(month), int(day))
                elif format_type == "mdy_nosep":
                    month, day, year = match.groups()
                    date_obj = datetime(int(year), int(month), int(day))

                # Return in ISO format
                return date_obj.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                # Invalid date, try next pattern
                continue

    return None


def normalize_document_number(text: str) -> Optional[str]:
    """
    Normalize document numbers by removing extra spaces.

    Examples:
        "A 1 2 3 4 5 6 7" -> "A1234567"
        "2 D" -> "2D"
        "I - 9 4" -> "I-94"

    Args:
        text: Input text containing a document number

    Returns:
        Normalized document number or None if empty
    """
    if not text:
        return None

    # Normalize spacing
    normalized = normalize_ocr_spacing(text)

    # Remove any remaining spaces
    normalized = normalized.replace(" ", "")

    return normalized if normalized else None


def normalize_class_of_admission(text: str) -> Optional[str]:
    """
    Normalize class of admission codes (visa types).

    Examples:
        "L S" -> "LS"
        "H 1 B" -> "H1B"
        "F - 1" -> "F-1"

    Args:
        text: Input text containing class of admission

    Returns:
        Normalized class code or None if empty
    """
    if not text:
        return None

    # Normalize spacing
    normalized = normalize_ocr_spacing(text)

    # Common visa types should have specific formats
    # H-1B, F-1, J-1, L-1, etc. should have hyphen
    # But OCR might extract as "H 1 B" or "H1B"

    # If it's a letter followed by digit(s) and optional letter, add hyphen
    # "H1B" -> "H-1B", "F1" -> "F-1"
    normalized = re.sub(r"^([A-Z])(\d+)([A-Z]?)$", r"\1-\2\3", normalized)

    return normalized if normalized else None


def normalize_country_name(text: str) -> Optional[str]:
    """
    Normalize country names by removing extra spaces.

    Examples:
        "S I N G A P O R E" -> "SINGAPORE"
        "U N I T E D   S T A T E S" -> "UNITED STATES"

    Args:
        text: Input text containing a country name

    Returns:
        Normalized country name or None if empty
    """
    if not text:
        return None

    # Normalize spacing
    normalized = normalize_ocr_spacing(text)

    # For multi-word countries, preserve single spaces between words
    # but remove extra spaces
    normalized = " ".join(normalized.split())

    return normalized.upper() if normalized else None


def normalize_name(text: str) -> Optional[str]:
    """
    Normalize person names by removing extra spaces and fixing capitalization.

    Examples:
        "M U K H E R J E E" -> "MUKHERJEE"
        "A A N K I T A" -> "AANKITA"

    Args:
        text: Input text containing a name

    Returns:
        Normalized name or None if empty
    """
    if not text:
        return None

    # Normalize spacing
    normalized = normalize_ocr_spacing(text)

    # Remove extra spaces
    normalized = " ".join(normalized.split())

    # Capitalize properly
    normalized = normalized.upper()

    return normalized if normalized else None


def normalize_i94_fields(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize all fields in an I-94 form extraction result.

    This function applies appropriate normalization to each field type:
    - Dates: Extract and format as YYYY-MM-DD
    - Names: Remove extra spaces, capitalize
    - Document numbers: Remove spaces
    - Class of admission: Format visa codes properly
    - Countries: Normalize spacing

    Args:
        extracted_data: Dictionary with raw extracted field values

    Returns:
        Dictionary with normalized field values
    """
    normalized = extracted_data.copy()

    # Normalize date fields
    date_fields = [
        "birth_date",
        "date_of_birth",
        "dob",
        "arrival_date",
        "date_of_arrival",
        "admit_until_date",
        "expiration_date",
        "processing_date",
        "issue_date",
    ]

    for field in date_fields:
        if field in normalized and normalized[field]:
            date_value = extract_date(str(normalized[field]), field_name=field)
            normalized[field] = date_value

    # Normalize name fields
    name_fields = [
        "first_name",
        "last_name",
        "full_name",
        "given_name",
        "surname",
        "middle_name",
    ]

    for field in name_fields:
        if field in normalized and normalized[field]:
            normalized[field] = normalize_name(str(normalized[field]))

    # Normalize document number fields
    doc_fields = [
        "document_number",
        "i94_number",
        "i94_record_number",
        "admission_number",
        "passport_number",
    ]

    for field in doc_fields:
        if field in normalized and normalized[field]:
            normalized[field] = normalize_document_number(str(normalized[field]))

    # Normalize class of admission
    if "class_of_admission" in normalized and normalized["class_of_admission"]:
        normalized["class_of_admission"] = normalize_class_of_admission(
            str(normalized["class_of_admission"])
        )

    # Normalize country fields
    country_fields = [
        "country_of_citizenship",
        "country",
        "nationality",
        "passport_country",
        "country_of_issuance",
    ]

    for field in country_fields:
        if field in normalized and normalized[field]:
            normalized[field] = normalize_country_name(str(normalized[field]))

    return normalized
