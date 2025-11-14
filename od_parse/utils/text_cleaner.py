"""
Text cleaning utilities for od-parse.

This module provides functions to clean and validate extracted text,
removing garbage characters and improving text quality.
"""

import re
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def calculate_alpha_ratio(text: str) -> float:
    """
    Calculate the ratio of alphabetic characters in text.

    Args:
        text: Input text

    Returns:
        Ratio of alphabetic characters (0.0 to 1.0)
    """
    if not text:
        return 0.0

    alpha_count = sum(c.isalpha() for c in text)
    return alpha_count / len(text)


def is_garbage_text(text: str, min_alpha_ratio: float = 0.5) -> bool:
    """
    Check if text appears to be garbage (low alphabetic ratio).

    Args:
        text: Input text
        min_alpha_ratio: Minimum alphabetic ratio (default: 0.5 = 50%)

    Returns:
        True if text appears to be garbage
    """
    if not text or len(text.strip()) < 3:
        return True

    alpha_ratio = calculate_alpha_ratio(text)
    return alpha_ratio < min_alpha_ratio


def clean_cid_codes(text: str) -> str:
    """
    Remove CID codes from text.

    CID codes appear as (cid:123) and indicate font encoding issues.

    Args:
        text: Input text with potential CID codes

    Returns:
        Text with CID codes removed
    """
    # Remove CID codes like (cid:123)
    cleaned = re.sub(r"\(cid:\d+\)", "", text)

    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifacts and noise.

    Args:
        text: Input text with potential OCR artifacts

    Returns:
        Cleaned text
    """
    # Remove common OCR noise patterns
    patterns = [
        r"[|]{2,}",  # Multiple pipes
        r"[_]{3,}",  # Multiple underscores
        r"[~]{2,}",  # Multiple tildes
        r"[`]{2,}",  # Multiple backticks
        r"[\^]{2,}",  # Multiple carets
        r"[\\]{2,}",  # Multiple backslashes
    ]

    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)

    # Remove isolated special characters (not part of words)
    cleaned = re.sub(r"\s+[^\w\s]\s+", " ", cleaned)

    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned


def clean_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove or normalize special characters.

    Args:
        text: Input text
        keep_punctuation: Whether to keep common punctuation (.,!?;:)

    Returns:
        Cleaned text
    """
    if keep_punctuation:
        # Keep alphanumeric, spaces, and common punctuation
        cleaned = re.sub(r'[^a-zA-Z0-9\s.,!?;:\-\'"()\[\]{}]', "", text)
    else:
        # Keep only alphanumeric and spaces
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove excessive whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.

    Args:
        text: Input text

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces with single space
    text = re.sub(r" +", " ", text)

    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def clean_extracted_text(
    text: str,
    remove_cid: bool = True,
    remove_ocr_artifacts: bool = True,
    remove_special_chars: bool = False,
    min_alpha_ratio: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Clean extracted text with multiple cleaning strategies.

    Args:
        text: Input text to clean
        remove_cid: Remove CID codes (default: True)
        remove_ocr_artifacts: Remove OCR artifacts (default: True)
        remove_special_chars: Remove special characters (default: False)
        min_alpha_ratio: Minimum alpha ratio to consider text valid (default: None)

    Returns:
        Dictionary with cleaned text and metadata:
        {
            'original_text': str,
            'cleaned_text': str,
            'is_garbage': bool,
            'alpha_ratio': float,
            'cid_count': int,
            'cleaning_applied': list
        }
    """
    original_text = text
    cleaned = text
    cleaning_applied = []

    # Count CID codes before cleaning
    cid_count = text.count("(cid:")

    # Remove CID codes
    if remove_cid and cid_count > 0:
        cleaned = clean_cid_codes(cleaned)
        cleaning_applied.append("cid_removal")
        logger.debug(f"Removed {cid_count} CID codes")

    # Remove OCR artifacts
    if remove_ocr_artifacts:
        before_len = len(cleaned)
        cleaned = clean_ocr_artifacts(cleaned)
        if len(cleaned) < before_len:
            cleaning_applied.append("ocr_artifact_removal")
            logger.debug(
                f"Removed OCR artifacts (reduced by {before_len - len(cleaned)} chars)"
            )

    # Remove special characters
    if remove_special_chars:
        before_len = len(cleaned)
        cleaned = clean_special_characters(cleaned, keep_punctuation=True)
        if len(cleaned) < before_len:
            cleaning_applied.append("special_char_removal")
            logger.debug(
                f"Removed special characters (reduced by {before_len - len(cleaned)} chars)"
            )

    # Normalize whitespace
    cleaned = normalize_whitespace(cleaned)

    # Calculate alpha ratio
    alpha_ratio = calculate_alpha_ratio(cleaned)

    # Check if garbage
    is_garbage = False
    if min_alpha_ratio is not None:
        is_garbage = is_garbage_text(cleaned, min_alpha_ratio)
        if is_garbage:
            logger.warning(
                f"Text appears to be garbage (alpha ratio: {alpha_ratio:.2f})"
            )

    return {
        "original_text": original_text,
        "cleaned_text": cleaned,
        "is_garbage": is_garbage,
        "alpha_ratio": alpha_ratio,
        "cid_count": cid_count,
        "cleaning_applied": cleaning_applied,
        "original_length": len(original_text),
        "cleaned_length": len(cleaned),
        "reduction_percent": (
            ((len(original_text) - len(cleaned)) / len(original_text) * 100)
            if original_text
            else 0
        ),
    }


def extract_valid_names(text: str, min_alpha_ratio: float = 0.7) -> Optional[str]:
    """
    Extract valid person names from text, filtering out garbage.

    Args:
        text: Input text that may contain a name
        min_alpha_ratio: Minimum alphabetic ratio for valid names (default: 0.7 = 70%)

    Returns:
        Cleaned name if valid, None if garbage
    """
    if not text:
        return None

    # Clean the text
    cleaned = clean_extracted_text(
        text,
        remove_cid=True,
        remove_ocr_artifacts=True,
        remove_special_chars=True,
        min_alpha_ratio=min_alpha_ratio,
    )

    # Check if garbage
    if cleaned["is_garbage"]:
        logger.warning(
            f"Name appears to be garbage (alpha ratio: {cleaned['alpha_ratio']:.2f}): {text[:50]}"
        )
        return None

    # Additional name-specific validation
    name = cleaned["cleaned_text"].strip()

    # Names should have at least 2 characters
    if len(name) < 2:
        return None

    # Names should not be all numbers
    if name.replace(" ", "").isdigit():
        return None

    # Names should not have excessive numbers
    digit_ratio = sum(c.isdigit() for c in name) / len(name)
    if digit_ratio > 0.3:  # More than 30% digits
        logger.warning(f"Name has too many digits ({digit_ratio:.0%}): {name}")
        return None

    return name


def get_text_quality_score(text: str) -> Dict[str, Any]:
    """
    Calculate comprehensive text quality score.

    Args:
        text: Input text

    Returns:
        Dictionary with quality metrics:
        {
            'overall_score': float (0.0 to 1.0),
            'alpha_ratio': float,
            'digit_ratio': float,
            'space_ratio': float,
            'special_char_ratio': float,
            'avg_word_length': float,
            'cid_count': int,
            'quality_level': str ('excellent', 'good', 'fair', 'poor')
        }
    """
    if not text:
        return {"overall_score": 0.0, "quality_level": "poor", "error": "Empty text"}

    # Calculate character ratios
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    space_ratio = sum(c.isspace() for c in text) / len(text)
    special_char_ratio = 1.0 - alpha_ratio - digit_ratio - space_ratio

    # Calculate word statistics
    words = text.split()
    avg_word_length = sum(len(w) for w in words) / len(words) if words else 0

    # Count CID codes
    cid_count = text.count("(cid:")

    # Calculate overall score
    # Good text should have:
    # - High alpha ratio (70-90%)
    # - Reasonable space ratio (10-20%)
    # - Low special char ratio (<10%)
    # - Average word length 3-8 characters
    # - No CID codes

    alpha_score = min(alpha_ratio / 0.8, 1.0)  # Target 80% alpha
    space_score = (
        min(space_ratio / 0.15, 1.0) if space_ratio < 0.3 else 0.5
    )  # Target 15% spaces
    special_score = 1.0 - min(
        special_char_ratio / 0.1, 1.0
    )  # Penalize >10% special chars
    word_length_score = 1.0 if 3 <= avg_word_length <= 8 else 0.5
    cid_score = 1.0 if cid_count == 0 else max(0.0, 1.0 - (cid_count / 100))

    overall_score = (
        alpha_score + space_score + special_score + word_length_score + cid_score
    ) / 5

    # Determine quality level
    if overall_score >= 0.8:
        quality_level = "excellent"
    elif overall_score >= 0.6:
        quality_level = "good"
    elif overall_score >= 0.4:
        quality_level = "fair"
    else:
        quality_level = "poor"

    return {
        "overall_score": overall_score,
        "alpha_ratio": alpha_ratio,
        "digit_ratio": digit_ratio,
        "space_ratio": space_ratio,
        "special_char_ratio": special_char_ratio,
        "avg_word_length": avg_word_length,
        "cid_count": cid_count,
        "quality_level": quality_level,
    }
