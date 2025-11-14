"""
Utility modules for the od-parse library.
"""

from od_parse.utils.file_utils import validate_file
from od_parse.utils.logging_utils import get_logger
from od_parse.utils.text_cleaner import (
    clean_extracted_text,
    extract_valid_names,
    get_text_quality_score,
    calculate_alpha_ratio,
    is_garbage_text,
    clean_cid_codes,
    clean_ocr_artifacts,
)
from od_parse.utils.text_normalizer import (
    normalize_ocr_spacing,
    extract_date,
    normalize_document_number,
    normalize_class_of_admission,
    normalize_country_name,
    normalize_name,
    normalize_i94_fields,
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
