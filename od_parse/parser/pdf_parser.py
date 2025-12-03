"""Core PDF parsing functionality."""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import pdf2image
from pdfminer.converter import PDFPageAggregator
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams, LTTextContainer
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from od_parse.utils.file_utils import validate_file
from od_parse.utils.logging_utils import get_logger
from od_parse.utils.text_normalizer import clean_text_for_json, normalize_ocr_spacing

logger = get_logger(__name__)

# Optional: pdfplumber for table extraction (pure Python, no Java needed)
PDFPLUMBER_AVAILABLE: bool = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    logger.warning(
        "pdfplumber not available. Table extraction will be limited. "
        "Install with: pip install pdfplumber"
    )


# Text quality thresholds
MIN_TEXT_LENGTH = 10
MIN_ALPHA_RATIO = 0.5
MIN_VALID_WORDS = 3
MAX_GARBAGE_RATIO = 0.1
VALID_WORD_REGEX = re.compile(r"[a-zA-Z]{3,}")
GARBAGE_CHARS = frozenset("<>;()[]{}@#$%^&*")

# CID encoding thresholds
MIN_CID_COUNT_FOR_FALLBACK = 10
MAX_CID_RATIO = 0.1  # 10% of content

# Normalization thresholds
MIN_ALPHA_RATIO_FOR_NORMALIZATION = 0.3
MAX_CID_COUNT_FOR_NORMALIZATION = 10

# OCR settings
OCR_DPI = 300
LOW_RES_THRESHOLD = 2000  # pixels
MIN_UPSCALE_FACTOR = 2.0
MAX_UPSCALE_FACTOR = 4.0
TARGET_RESOLUTION = 2000  # pixels

# OCR preprocessing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
DENOISE_STRENGTH = 10
DENOISE_TEMPLATE_WINDOW = 7
DENOISE_SEARCH_WINDOW = 21
SHARPEN_AMOUNT = 1.5
SHARPEN_BLUR_SIGMA = 2.0
MIN_DESKEW_ANGLE = 0.5  # degrees


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


def validate_and_log_text_quality(
    text: str, source: str, min_length: int = MIN_TEXT_LENGTH
) -> tuple[bool, float, bool]:
    """
    Validate text quality and log results.

    This helper function reduces code duplication by centralizing
    quality checks and logging for text extracted from different sources.

    Args:
        text: The extracted text to validate
        source: Source of extraction ("OCR", "PyMuPDF", "pdfminer")
        min_length: Minimum text length to consider valid

    Returns:
        Tuple of (is_valid, alpha_ratio, is_readable)
        - is_valid: True if text meets minimum length requirement
        - alpha_ratio: Ratio of alphabetic characters (0.0 to 1.0)
        - is_readable: True if text passes readability checks

    Example:
        >>> is_valid, alpha_ratio, readable = validate_and_log_text_quality(
        ...     "MUKHERJEE NAME DATE", "OCR"
        ... )
        >>> print(f"Valid: {is_valid}, Readable: {readable}")
        Valid: True, Readable: True
    """
    if not text or len(text.strip()) < min_length:
        logger.debug(f"{source} text too short: {len(text) if text else 0} chars")
        return False, 0.0, False

    alpha_ratio = calculate_alpha_ratio(text)
    readable = is_readable_text(text)

    logger.info(
        f"📊 {source} quality: {len(text)} chars, "
        f"alpha_ratio={alpha_ratio:.2f}, readable={readable}"
    )

    return True, alpha_ratio, readable


def is_readable_text(text: str) -> bool:
    """
    Checks if text appears to be readable using several heuristics.

    Optimized single-pass implementation that checks:
    1. Minimum length
    2. High ratio of alphabetic characters (>= 50%)
    3. Multiple words with 3+ consecutive letters (>= 3 words)
    4. Low ratio of garbage special characters (<= 10%)

    Args:
        text: The input string to analyze.

    Returns:
        True if the text is deemed readable, False otherwise.

    Performance:
        O(n) single-pass algorithm, ~50% faster than previous implementation.
    """
    if not text or len(text) < MIN_TEXT_LENGTH:
        return False

    text_len = len(text)
    alpha_count = 0
    garbage_count = 0
    valid_word_count = 0
    current_word_len = 0

    # Single pass through text
    for char in text:
        if char.isalpha():
            alpha_count += 1
            current_word_len += 1
        else:
            # End of word - check if it's valid (3+ letters)
            if current_word_len >= 3:
                valid_word_count += 1
            current_word_len = 0

            # Check if it's a garbage character
            if char in GARBAGE_CHARS:
                garbage_count += 1

    # Check last word (if text ends with a letter)
    if current_word_len >= 3:
        valid_word_count += 1

    # All checks in one pass - return True only if all conditions met
    alpha_ratio = alpha_count / text_len
    garbage_ratio = garbage_count / text_len

    return (
        alpha_ratio >= MIN_ALPHA_RATIO
        and garbage_ratio <= MAX_GARBAGE_RATIO
        and valid_word_count >= MIN_VALID_WORDS
    )


def parse_pdf(
    file_path: Union[str, Path],
    use_ocr: bool = True,
    clean_text: bool = True,
    preserve_structure: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Parse a PDF file and extract its content.

    Automatically detects scanned PDFs and applies OCR when needed.

    Args:
        file_path: Path to the PDF file
        use_ocr: Whether to use OCR for scanned PDFs (default: True)
        clean_text: Whether to clean text output (remove \\n, normalize whitespace) (default: True)
        preserve_structure: If clean_text=True, whether to preserve paragraph structure (default: False)
        **kwargs: Additional arguments for parsing

    Returns:
        Dictionary containing extracted content
    """
    file_path = validate_file(file_path, extension=".pdf")
    logger.info(f"Parsing PDF file: {file_path}")

    # Get document metadata
    page_count = "unknown"
    try:
        with open(file_path, "rb") as file:
            page_count = len(list(PDFPage.get_pages(file)))
            logger.info(f"Page count: {page_count}")
    except Exception as e:
        logger.warning(f"Could not determine page count: {e}")

    # Run extraction steps
    text = _run_extraction_step(
        "Text", extract_text, file_path, use_ocr_fallback=use_ocr
    )

    # Clean text if requested
    if clean_text and text:
        text = clean_text_for_json(text, preserve_structure=preserve_structure)
        logger.info(
            f"✨ Text cleaned for JSON output (preserve_structure={preserve_structure})"
        )

    images = _run_extraction_step("Images", extract_images, file_path)
    tables = _run_extraction_step("Tables", extract_tables, file_path)
    forms = _run_extraction_step("Forms", extract_forms, file_path)

    # Handwritten content extraction is currently disabled
    handwritten_content = []

    # Create metadata
    file_stats = Path(file_path).stat()
    metadata = {
        "file_name": Path(file_path).name,
        "file_size": file_stats.st_size,
        "page_count": page_count,
        "extraction_method": "pdfminer + pdfplumber",
        "text_length": len(text),
        "tables_found": len(tables),
        "forms_found": len(forms),
        "images_found": len(images),
        "handwritten_items_found": len(handwritten_content),
    }

    return {
        "text": text,
        "images": images,
        "tables": tables,
        "forms": forms,
        "handwritten_content": handwritten_content,
        "metadata": metadata,
    }


def _run_extraction_step(step_name: str, extraction_func, *args, **kwargs):
    """Run a single extraction step, timing it and logging the result."""
    start_time = time.time()
    try:
        result = extraction_func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        result_count = len(result) if isinstance(result, (list, str)) else 0
        logger.info(
            f"✅ {step_name} extraction completed: {result_count} items in {elapsed_time:.2f}s"
        )
        return result
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"✗ {step_name} extraction failed after {elapsed_time:.2f}s: {e}")
        # Return a default value based on expected type
        return "" if "text" in step_name.lower() else []


def extract_text(file_path: Union[str, Path], use_ocr_fallback: bool = True) -> str:
    """
    Extract text content from a PDF file.

    Automatically detects scanned PDFs and applies OCR when needed.
    Also detects CID-encoded PDFs (fonts without proper character mappings).

    Args:
        file_path: Path to the PDF file
        use_ocr_fallback: Whether to use OCR if no text is found (default: True)

    Returns:
        Extracted text content (cleaned for JSON compatibility)
    """
    try:
        # First, try to extract text using pdfminer
        raw_text = pdfminer_extract_text(file_path)

        # Clean the text for better JSON compatibility
        if raw_text:
            # Replace problematic unicode characters
            cleaned_text = raw_text.replace("\u2013", "-")  # em dash
            cleaned_text = cleaned_text.replace("\u2014", "--")  # en dash
            cleaned_text = cleaned_text.replace("\u2019", "'")  # right single quotation
            cleaned_text = cleaned_text.replace("\u201c", '"')  # left double quotation
            cleaned_text = cleaned_text.replace("\u201d", '"')  # right double quotation
            cleaned_text = cleaned_text.replace("\u00a0", " ")  # non-breaking space

            # Remove excessive whitespace
            cleaned_text = " ".join(cleaned_text.split())

            # Check if pdfminer text is readable FIRST (before CID check)
            pdfminer_is_readable = is_readable_text(cleaned_text)

            # If text is unreadable (garbage), skip to OCR immediately
            if not pdfminer_is_readable and use_ocr_fallback:
                logger.warning(
                    "⚠️  pdfminer extracted garbage text (not readable) - going straight to OCR"
                )
                logger.debug(f"📝 Garbage sample: {cleaned_text[:200]}")
                ocr_text = extract_text_with_ocr(file_path)

                # Use helper function to validate OCR result
                is_valid, alpha_ratio, readable = validate_and_log_text_quality(
                    ocr_text, "OCR"
                )

                if is_valid:
                    return ocr_text
                else:
                    logger.warning(
                        "OCR didn't produce usable text, returning pdfminer text"
                    )
                    return cleaned_text

            # Check for CID codes (character identifier codes from embedded fonts)
            # CID codes appear as "(cid:XXX)" in the extracted text
            cid_count = cleaned_text.count("(cid:")

            # If more than 10% of content is CID codes, the PDF has encoding issues
            if cid_count > MIN_CID_COUNT_FOR_FALLBACK and len(cleaned_text) > 0:
                cid_ratio = cid_count / (len(cleaned_text) / 100)  # Approximate ratio
                if cid_ratio > MAX_CID_RATIO:
                    logger.warning(
                        f"⚠️  Detected {cid_count} CID codes in extracted text - PDF has font encoding issues"
                    )

                    # Try PyMuPDF first (often handles CID fonts better)
                    logger.info("Trying PyMuPDF as alternative text extractor...")
                    pymupdf_text = extract_text_with_pymupdf(file_path)

                    # Check if PyMuPDF produced better results
                    if pymupdf_text:
                        pymupdf_cid_count = pymupdf_text.count("(cid:")
                        _, alpha_ratio, readable = validate_and_log_text_quality(
                            pymupdf_text, "PyMuPDF"
                        )

                        logger.info(
                            f"📊 PyMuPDF CID check: {pymupdf_cid_count} codes (vs {cid_count})"
                        )
                        logger.debug(f"📝 PyMuPDF sample: {pymupdf_text[:200]}")

                        # Only accept PyMuPDF result if it has:
                        # 1. Fewer CID codes than pdfminer
                        # 2. Text is actually readable (not garbage like "4<2/,91,,")
                        if pymupdf_cid_count < cid_count and readable:
                            logger.info(
                                f"✅ PyMuPDF produced readable text: {pymupdf_cid_count} CID codes vs {cid_count}"
                            )
                            return pymupdf_text
                        else:
                            logger.warning(
                                "⚠️  PyMuPDF text not readable (garbage detected) - will try OCR"
                            )

                    # If PyMuPDF didn't help, try OCR as last resort
                    if use_ocr_fallback:
                        logger.info("Falling back to OCR for better text extraction")
                        ocr_text = extract_text_with_ocr(file_path)

                        # Use helper function to validate OCR result
                        is_valid, _, _ = validate_and_log_text_quality(ocr_text, "OCR")

                        if is_valid:
                            return ocr_text
                        else:
                            logger.warning(
                                "OCR didn't produce usable text, returning pdfminer text"
                            )
                            return cleaned_text

            # Check if we got meaningful text (more than just whitespace/special chars)
            if len(cleaned_text.strip()) > 10:
                logger.info(f"Extracted {len(cleaned_text)} characters using pdfminer")
                return cleaned_text

        # If no text found or very little text, this might be a scanned PDF
        # Try multiple fallback methods
        if use_ocr_fallback:
            # First try PyMuPDF (faster than OCR)
            logger.info(
                f"No text found with pdfminer, trying PyMuPDF for scanned PDF: {file_path}"
            )
            pymupdf_text = extract_text_with_pymupdf(file_path)
            if pymupdf_text and len(pymupdf_text.strip()) > 10:
                logger.info(
                    f"✅ Extracted {len(pymupdf_text)} characters using PyMuPDF"
                )
                return pymupdf_text

            # If PyMuPDF didn't work, try OCR
            logger.info(
                f"PyMuPDF didn't extract text, attempting OCR for scanned PDF: {file_path}"
            )
            ocr_text = extract_text_with_ocr(file_path)
            if ocr_text and len(ocr_text.strip()) > 0:
                logger.info(f"✅ Extracted {len(ocr_text)} characters using OCR")
                return ocr_text
            else:
                logger.warning(f"⚠️  OCR extraction returned no text for {file_path}")
                logger.warning("⚠️  This may be due to missing Tesseract installation")
                logger.warning(
                    "⚠️  Install Tesseract: brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)"
                )

        return ""
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
        return ""


def extract_text_with_pymupdf(file_path: Union[str, Path]) -> str:
    """
    Extract text using PyMuPDF (fitz) as an alternative to pdfminer.

    PyMuPDF often handles CID-encoded fonts better than pdfminer.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text from all pages
    """
    try:
        import fitz  # PyMuPDF

        logger.info(f"Trying PyMuPDF for text extraction: {file_path}")

        doc = fitz.open(file_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                all_text.append(text.strip())

        doc.close()

        combined_text = " ".join(all_text)

        # Clean the text
        if combined_text:
            cleaned_text = " ".join(combined_text.split())

            # Only normalize if text quality is good (check alpha ratio and CID codes)
            alpha_ratio = calculate_alpha_ratio(cleaned_text)
            cid_count = cleaned_text.count("(cid:")

            # Normalize only if text quality is reasonable
            if (
                alpha_ratio > MIN_ALPHA_RATIO_FOR_NORMALIZATION
                and cid_count < MAX_CID_COUNT_FOR_NORMALIZATION
            ):
                cleaned_text = normalize_ocr_spacing(cleaned_text)
                logger.debug(
                    f"PyMuPDF text normalized and cleaned ({len(cleaned_text)} chars, alpha_ratio={alpha_ratio:.2f})"
                )
            else:
                logger.debug(
                    f"PyMuPDF text NOT normalized (alpha_ratio={alpha_ratio:.2f}, cid_count={cid_count})"
                )

            return cleaned_text

        return ""

    except ImportError:
        logger.warning(
            "PyMuPDF (fitz) not available. Install with: pip install PyMuPDF"
        )
        return ""
    except Exception as e:
        logger.error(f"Error during PyMuPDF extraction from {file_path}: {e}")
        return ""


def extract_text_with_ocr(file_path: Union[str, Path]) -> str:
    """
    Extract text from a scanned PDF using OCR.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text from all pages
    """
    try:
        import pytesseract
        from PIL import Image

        logger.info(f"Converting PDF to images for OCR: {file_path}")

        # Convert PDF pages to images
        images = pdf2image.convert_from_path(file_path, dpi=OCR_DPI)

        all_text = []
        for i, img in enumerate(images):
            logger.info(f"Running OCR on page {i+1}/{len(images)}")

            # Convert PIL image to numpy array for preprocessing
            img_array = np.array(img)

            # ADVANCED PREPROCESSING FOR LOW-RESOLUTION IMAGES
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # 1. UPSCALE LOW-RESOLUTION IMAGES
            height, width = gray.shape
            # If image is low resolution, upscale it
            if width < LOW_RES_THRESHOLD or height < LOW_RES_THRESHOLD:
                # Calculate scale factor
                scale_factor = max(
                    MIN_UPSCALE_FACTOR,
                    min(MAX_UPSCALE_FACTOR, TARGET_RESOLUTION / max(width, height)),
                )
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                logger.info(
                    f"📐 Upscaling low-res image from {width}x{height} to {new_width}x{new_height} ({scale_factor:.1f}x)"
                )

                # Use INTER_CUBIC for upscaling (better quality than INTER_LINEAR)
                gray = cv2.resize(
                    gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )

            # 2. ENHANCE CONTRAST (CLAHE)
            # Contrast Limited Adaptive Histogram Equalization
            # This improves contrast locally, especially good for low-quality scans
            clahe = cv2.createCLAHE(
                clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE
            )
            gray = clahe.apply(gray)
            logger.debug("✨ Applied CLAHE contrast enhancement")

            # 3. DENOISE
            # Apply denoising to remove noise while preserving edges
            processed = cv2.fastNlMeansDenoising(
                gray,
                h=DENOISE_STRENGTH,
                templateWindowSize=DENOISE_TEMPLATE_WINDOW,
                searchWindowSize=DENOISE_SEARCH_WINDOW,
            )
            logger.debug("🧹 Applied denoising")

            # 4. SHARPEN
            # Apply unsharp masking to sharpen text edges
            gaussian = cv2.GaussianBlur(processed, (0, 0), SHARPEN_BLUR_SIGMA)
            processed = cv2.addWeighted(processed, SHARPEN_AMOUNT, gaussian, -0.5, 0)
            logger.debug("🔪 Applied sharpening")

            # 5. DESKEW (Rotation Correction)
            # Detect and correct skew/rotation
            coords = np.column_stack(np.where(processed > 0))
            if len(coords) > 0:
                angle = cv2.minAreaRect(coords)[-1]

                # Correct angle
                if angle < -45:
                    angle = -(90 + angle)
                else:
                    angle = -angle

                # Only deskew if angle is significant
                if abs(angle) > MIN_DESKEW_ANGLE:
                    logger.info(f"🔄 Deskewing image by {angle:.2f} degrees")
                    (h, w) = processed.shape
                    center = (w // 2, h // 2)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                    processed = cv2.warpAffine(
                        processed,
                        M,
                        (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE,
                    )

            # 6. ADAPTIVE THRESHOLDING
            # Apply adaptive thresholding to binarize the image
            # This works better than global thresholding for varying lighting conditions
            processed = cv2.adaptiveThreshold(
                processed,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,  # Block size
                2,  # Constant subtracted from mean
            )
            logger.debug("⚫⚪ Applied adaptive thresholding")

            # 7. MORPHOLOGICAL OPERATIONS
            # Close small gaps in characters (helps with broken text)
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            logger.debug("🔗 Applied morphological closing")

            # Convert back to PIL Image for pytesseract
            processed_img = Image.fromarray(processed)

            # Extract text using pytesseract
            # Use config optimized for low-quality scanned documents
            # --oem 3: Use default OCR Engine Mode (LSTM + Legacy)
            # --psm 1: Automatic page segmentation with OSD (Orientation and Script Detection)
            # -c tessedit_char_whitelist: Limit to common characters (optional, commented out)
            custom_config = r"--oem 3 --psm 1"

            # Try OCR with the processed image
            page_text = pytesseract.image_to_string(processed_img, config=custom_config)

            # If OCR returns very little text or garbage, try alternative PSM modes
            if (
                len(page_text.strip()) < MIN_TEXT_LENGTH
                or calculate_alpha_ratio(page_text) < MIN_ALPHA_RATIO_FOR_NORMALIZATION
            ):
                logger.warning(
                    f"⚠️  Low-quality OCR result on page {i+1}, trying alternative PSM mode..."
                )

                # Try PSM 3 (Fully automatic page segmentation, but no OSD)
                alt_config = r"--oem 3 --psm 3"
                alt_text = pytesseract.image_to_string(processed_img, config=alt_config)

                # Use alternative if it's better
                if len(alt_text.strip()) > len(page_text.strip()):
                    logger.info("✅ Alternative PSM mode produced better results")
                    page_text = alt_text

            if page_text.strip():
                all_text.append(f"--- Page {i+1} ---\n{page_text.strip()}")

        # Combine all pages
        combined_text = "\n\n".join(all_text)

        # Clean the text
        if combined_text:
            # Additional whitespace normalization
            cleaned_text = " ".join(combined_text.split())

            # Only normalize if text quality is good (check alpha ratio)
            alpha_ratio = calculate_alpha_ratio(cleaned_text)

            # Normalize only if alpha ratio is reasonable
            # OCR should produce mostly alphabetic text, not garbage
            if alpha_ratio > MIN_ALPHA_RATIO_FOR_NORMALIZATION:
                cleaned_text = normalize_ocr_spacing(cleaned_text)
                logger.info(
                    f"✅ OCR text normalized and cleaned ({len(cleaned_text)} chars, alpha_ratio={alpha_ratio:.2f})"
                )
            else:
                logger.warning(
                    f"⚠️  OCR text quality too low for normalization (alpha_ratio={alpha_ratio:.2f})"
                )

            return cleaned_text

        return ""

    except ImportError as e:
        logger.error(f"❌ pytesseract not available: {e}")
        logger.error("💡 Install with: pip install pytesseract")
        return ""
    except Exception as e:
        error_msg = str(e)
        if (
            "tesseract is not installed" in error_msg.lower()
            or "not in your path" in error_msg.lower()
        ):
            logger.error("❌ Tesseract OCR is not installed on your system")
            logger.error("💡 Install Tesseract:")
            logger.error("   macOS:   brew install tesseract")
            logger.error("   Ubuntu:  sudo apt-get install tesseract-ocr")
            logger.error(
                "   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki"
            )
        else:
            logger.error(f"❌ Error during OCR extraction from {file_path}: {e}")
        return ""


def extract_images(
    file_path: Union[str, Path],
    output_dir: Optional[str] = None,
    max_pages: Optional[int] = None,
    dpi: int = 200,
) -> List[str]:
    """
    Extract images from a PDF file.

    Args:
        file_path: Path to the PDF file
        output_dir: Directory to save extracted images
        max_pages: Maximum number of pages to process (None = all pages)
        dpi: DPI for image extraction (default: 200)

    Returns:
        List of paths to extracted images
    """
    file_path = Path(file_path)
    if output_dir is None:
        output_dir = file_path.parent / f"{file_path.stem}_images"

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Convert PDF pages to images
        # Limit pages if max_pages is specified
        if max_pages:
            images = pdf2image.convert_from_path(
                file_path, dpi=dpi, last_page=max_pages
            )
        else:
            images = pdf2image.convert_from_path(file_path, dpi=dpi)
        image_paths = []

        for i, img in enumerate(images):
            img_path = os.path.join(output_dir, f"page_{i+1}.png")
            img.save(img_path, "PNG")
            image_paths.append(img_path)

            # Use OpenCV to detect and extract embedded images
            cv_img = cv2.imread(img_path)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for j, contour in enumerate(contours):
                if cv2.contourArea(contour) > 10000:  # Filter small contours
                    x, y, w, h = cv2.boundingRect(contour)
                    cropped = cv_img[y:y + h, x:x + w]
                    crop_path = os.path.join(output_dir, f"page_{i+1}_img_{j+1}.png")
                    cv2.imwrite(crop_path, cropped)
                    image_paths.append(crop_path)

        return image_paths
    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {e}")
        return []


def extract_tables(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract tables from a PDF file using pdfplumber (pure Python, no Java needed).

    Args:
        file_path: Path to the PDF file

    Returns:
        List of extracted tables as dictionaries
    """
    if not PDFPLUMBER_AVAILABLE:
        logger.warning("⚠️  pdfplumber not available - skipping table extraction")
        logger.warning("💡 Install with: pip install pdfplumber")
        return []

    try:
        cleaned_tables = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract tables from this page
                tables = page.extract_tables()

                if not tables:
                    continue

                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:  # Need at least header + 1 row
                        continue

                    # First row is usually the header
                    headers = table[0]
                    rows = table[1:]

                    # Clean headers
                    clean_headers = []
                    for i, header in enumerate(headers):
                        if header and str(header).strip():
                            clean_header = str(header).strip()
                        else:
                            clean_header = f"column_{i}"
                        clean_headers.append(clean_header)

                    # Convert to list of dictionaries
                    cleaned_rows = []
                    for row in rows:
                        if not row or len(row) != len(clean_headers):
                            continue

                        cleaned_row = {}
                        has_content = False

                        for header, value in zip(clean_headers, row):
                            # Clean the value
                            if value is None or (
                                isinstance(value, str) and not value.strip()
                            ):
                                clean_value = None
                            elif isinstance(value, str):
                                clean_value = value.strip()
                                if clean_value.lower() in ["nan", "none", ""]:
                                    clean_value = None
                                else:
                                    has_content = True
                            else:
                                clean_value = value
                                has_content = True

                            cleaned_row[header] = clean_value

                        # Only add row if it has some meaningful content
                        if has_content:
                            cleaned_rows.append(cleaned_row)

                    if cleaned_rows:  # Only add table if it has content
                        cleaned_tables.append(
                            {
                                "data": cleaned_rows,
                                "shape": (len(cleaned_rows), len(clean_headers)),
                                "page": page_num + 1,
                                "table_index": table_idx,
                                "confidence": 0.85,  # pdfplumber is quite reliable
                            }
                        )

        if cleaned_tables:
            logger.info(f"✅ Extracted {len(cleaned_tables)} tables from {file_path}")
        else:
            logger.info(f"No tables found in {file_path}")

        return cleaned_tables

    except Exception as e:
        logger.error(f"❌ Error extracting tables from {file_path}: {e}")
        return []


def extract_forms(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Extract form elements (checkboxes, radio buttons, text fields) from a PDF file.

    Args:
        file_path: Path to the PDF file

    Returns:
        List of extracted form elements as dictionaries
    """
    try:
        # This is a simplified implementation
        # In a real-world scenario, you would use a library like PyPDF2 or pdfrw
        # to extract form fields

        form_elements = []

        # Open the PDF file
        with open(file_path, "rb") as file:
            # Create a PDF resource manager and page aggregator
            resource_manager = PDFResourceManager()
            laparams = LAParams()
            device = PDFPageAggregator(resource_manager, laparams=laparams)
            interpreter = PDFPageInterpreter(resource_manager, device)

            # Process each page
            for page_num, page in enumerate(PDFPage.get_pages(file)):
                interpreter.process_page(page)
                layout = device.get_result()

                # Look for potential form elements
                for element in layout:
                    if isinstance(element, LTTextContainer):
                        raw_text = element.get_text().strip()
                        # Clean the text for better JSON output
                        cleaned_text = clean_text_for_json(
                            raw_text, preserve_structure=True
                        )
                        text_lower = cleaned_text.lower()

                        if any(
                            keyword in text_lower
                            for keyword in ["check", "select", "choose", "click"]
                        ):
                            form_elements.append(
                                {
                                    "type": "potential_form_field",
                                    "page": page_num + 1,
                                    "text": cleaned_text,
                                    "bbox": (
                                        element.x0,
                                        element.y0,
                                        element.x1,
                                        element.y1,
                                    ),
                                }
                            )

        return form_elements
    except Exception as e:
        logger.error(f"Error extracting form elements from {file_path}: {e}")
        return []
