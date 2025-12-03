"""Utility modules for the od-parse library."""
from __future__ import annotations

from od_parse.utils.file_utils import validate_file
from od_parse.utils.logging_utils import get_logger
from od_parse.utils.text_cleaner import (
    calculate_alpha_ratio,
    clean_cid_codes,
    clean_extracted_text,
    clean_ocr_artifacts,
    extract_valid_names,
    get_text_quality_score,
    is_garbage_text,
)
from od_parse.utils.text_normalizer import (
    extract_date,
    normalize_class_of_admission,
    normalize_country_name,
    normalize_document_number,
    normalize_i94_fields,
    normalize_name,
    normalize_ocr_spacing,
)

__all__ = [
    "validate_file",
    "get_logger",
    "clean_extracted_text",
    "extract_valid_names",
    "get_text_quality_score",
    "calculate_alpha_ratio",
    "is_garbage_text",
    "clean_cid_codes",
    "clean_ocr_artifacts",
    "normalize_ocr_spacing",
    "extract_date",
    "normalize_document_number",
    "normalize_class_of_admission",
    "normalize_country_name",
    "normalize_name",
    "normalize_i94_fields",
]
