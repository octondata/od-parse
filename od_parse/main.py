"""
OctonData Parse - Main Interface Module

This module provides an easy-to-use interface to the PDF parsing capabilities
of the od-parse library, allowing users to extract rich content from documents.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pdf2image
import pdfplumber
import PyPDF2

from od_parse.chunking.document_segmenter import DocumentSegmenter
from od_parse.config import get_advanced_config
from od_parse.config.llm_config import get_llm_config
from od_parse.converter import convert_to_markdown
from od_parse.parser import core_parse_pdf
from od_parse.quality import assess_document_quality
from od_parse.utils.logging_utils import configure_logging, get_logger


def parse_pdf(
    file_path: Union[str, Path],
    output_format: str = "raw",
    output_file: Optional[str] = None,
    pipeline_type: str = "default",
    use_deep_learning: bool = True,
    llm_model: Optional[str] = None,
    require_llm: bool = True,
    for_embeddings: bool = False,
) -> Dict[str, Any]:
    """
    Parse a PDF file using LLM-powered advanced document understanding.
    Orchestrates the parsing pipeline by calling a series of helper functions.
    """
    start_time = time.time()

    file_path = _validate_and_log_inputs(
        file_path, pipeline_type, use_deep_learning, llm_model, require_llm
    )

    if require_llm:
        _check_llm_availability()

    _configure_advanced_features(use_deep_learning)

    parsed_data = _run_core_parsing(file_path)

    parsed_data = _run_enhancements(
        parsed_data, file_path, use_deep_learning, require_llm, llm_model, pipeline_type
    )

    result = _build_final_result(
        parsed_data, file_path, start_time, pipeline_type, use_deep_learning
    )

    result = _handle_output(result, output_format, output_file, for_embeddings)

    return result


def _validate_and_log_inputs(
    file_path, pipeline_type, use_deep_learning, llm_model, require_llm
):
    """Validate inputs and log initial parameters."""
    logger = get_logger(__name__)
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    logger.info(f"Processing document: {file_path}")
    logger.info(f"Pipeline type: {pipeline_type}")
    logger.info(f"Use advanced features: {use_deep_learning}")
    logger.info(f"LLM model: {llm_model or 'auto-select'}")
    logger.info(f"Require LLM: {require_llm}")
    return file_path


def _check_llm_availability():
    """Check if required LLM API keys are available."""
    try:
        llm_config = get_llm_config()
        available_models = llm_config.get_available_models()
        if not available_models:
            raise ValueError(
                "No LLM API keys found. od-parse requires LLM access for document parsing.\n"
                "Please set one of the following environment variables:\n"
                "  OPENAI_API_KEY for OpenAI models (recommended)\n"
                "  ANTHROPIC_API_KEY for Claude models\n"
                "  GOOGLE_API_KEY for Gemini models\n"
                "  AZURE_OPENAI_API_KEY for Azure OpenAI\n"
                "See README.md for detailed setup instructions."
            )
        get_logger(__name__).info(f"Found {len(available_models)} available LLM models")
    except ImportError:
        raise ValueError(
            "LLM configuration not available. Please install required dependencies."
        )


def _configure_advanced_features(use_deep_learning):
    """Enable advanced features based on configuration."""
    config = get_advanced_config()
    if use_deep_learning:
        config.enable_feature("trocr", check_dependencies=False)
        config.enable_feature("table_transformer", check_dependencies=False)
        config.enable_feature("quality_assessment", check_dependencies=False)
        config.enable_feature("multilingual", check_dependencies=False)
    return config


def _run_core_parsing(file_path: Path) -> Dict[str, Any]:
    """Run the core PDF parsing logic with error handling."""
    logger = get_logger(__name__)
    try:
        return core_parse_pdf(str(file_path))
    except Exception as e:
        logger.error(f"Core parsing failed: {e}")
        return {
            "text": "",
            "tables": [],
            "forms": [],
            "images": [],
            "metadata": {"error": str(e)},
        }


def _run_enhancements(
    parsed_data, file_path, use_deep_learning, require_llm, llm_model, pipeline_type
):
    """Apply post-processing and data enhancement."""
    if use_deep_learning and require_llm:
        return _enhance_with_llm_processing(parsed_data, file_path, llm_model)
    elif use_deep_learning:
        return _enhance_with_advanced_features(parsed_data, file_path, pipeline_type)
    return parsed_data


def _build_final_result(
    parsed_data, file_path, start_time, pipeline_type, use_deep_learning
):
    """Assemble the final result dictionary with metadata and summary."""
    processing_time = time.time() - start_time
    file_stats = file_path.stat()
    return {
        "parsed_data": parsed_data,
        "metadata": {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_stats.st_size,
            "processing_time_seconds": processing_time,
            "pipeline_type": pipeline_type,
            "use_deep_learning": use_deep_learning,
            "timestamp": time.time(),
        },
        "summary": _create_summary(parsed_data, file_path, processing_time),
    }


def _handle_output(result, output_format, output_file, for_embeddings):
    """Manage output formatting, optimization, and file writing."""
    logger = get_logger(__name__)
    if output_format == "markdown":
        try:
            logger.info("Starting markdown conversion...")
            markdown_content = convert_to_markdown(result["parsed_data"])
            result["markdown_output"] = markdown_content
            logger.info("Markdown conversion completed successfully")
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error converting to markdown: {e}\n{error_details}")
            result["markdown_output"] = (
                f"# {result['metadata']['file_name']}\n\nMarkdown conversion failed: {e}\n\n"
                f"Error details have been logged.\n\n{error_details}"
            )

    if for_embeddings or output_format == "raw":
        result = _optimize_for_embeddings(result)

    if output_file:
        _write_output_file(result, output_file, output_format, logger)

    return result


def _optimize_for_embeddings(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optimize parsed result for embedding generation in document processing pipelines.

    This function extracts and structures the most relevant content for embedding generation,
    removing metadata and formatting that isn't useful for semantic understanding.
    """
    parsed_data = result.get("parsed_data", {})

    # Extract core content for embeddings
    embedding_optimized = {
        # Core document content
        "text": parsed_data.get("text", ""),
        "document_type": parsed_data.get("document_classification", {}).get(
            "document_type", "unknown"
        ),
        "confidence": parsed_data.get("document_classification", {}).get(
            "confidence", 0.0
        ),
        # Structured data from LLM analysis
        "extracted_data": parsed_data.get("llm_analysis", {}).get("extracted_data", {}),
        # Key entities and indicators
        "key_indicators": parsed_data.get("document_classification", {}).get(
            "key_indicators", {}
        ),
        # Structured content
        "tables": _extract_table_content_for_embeddings(parsed_data.get("tables", [])),
        "forms": _extract_form_content_for_embeddings(parsed_data.get("forms", [])),
        # Metadata for context
        "processing_metadata": {
            "document_type": parsed_data.get("document_classification", {}).get(
                "document_type", "unknown"
            ),
            "confidence": parsed_data.get("document_classification", {}).get(
                "confidence", 0.0
            ),
            "text_length": len(parsed_data.get("text", "")),
            "has_tables": len(parsed_data.get("tables", [])) > 0,
            "has_forms": len(parsed_data.get("forms", [])) > 0,
            "llm_processed": parsed_data.get("llm_analysis", {}).get(
                "processing_success", False
            ),
        },
    }

    return {"embedding_data": embedding_optimized}


def _extract_table_content_for_embeddings(tables: List[Dict]) -> List[Dict]:
    """Extract table content optimized for embeddings."""
    embedding_tables = []

    for table in tables:
        if "data" in table:
            # Convert table data to text representation
            table_text = []
            for row in table["data"]:
                if isinstance(row, dict):
                    row_text = " | ".join(
                        [f"{k}: {v}" for k, v in row.items() if v is not None]
                    )
                    if row_text:
                        table_text.append(row_text)

            embedding_tables.append(
                {
                    "content": "\n".join(table_text),
                    "row_count": len(table["data"]) if "data" in table else 0,
                }
            )

    return embedding_tables


def _extract_form_content_for_embeddings(forms: List[Dict]) -> List[Dict]:
    """Extract form content optimized for embeddings."""
    embedding_forms = []

    for form in forms:
        form_content = {}

        # Extract form fields and values
        if "fields" in form:
            for field in form["fields"]:
                if isinstance(field, dict):
                    field_name = field.get("name", field.get("label", "unknown_field"))
                    field_value = field.get("value", field.get("text", ""))
                    if field_value:
                        form_content[field_name] = field_value

        if form_content:
            embedding_forms.append(
                {"fields": form_content, "field_count": len(form_content)}
            )

    return embedding_forms


def _enhance_with_llm_processing(
    parsed_data: Dict[str, Any], file_path: Path, llm_model: Optional[str] = None
) -> Dict[str, Any]:
    """Enhance parsed data with LLM-powered document understanding."""
    logger = get_logger(__name__)

    try:
        from od_parse.llm import LLMDocumentProcessor
        from pdf2image import convert_from_path

        # Initialize LLM processor
        processor = LLMDocumentProcessor(model_id=llm_model)

        # Convert PDF to images for vision models
        try:
            images = convert_from_path(
                str(file_path), first_page=1, last_page=3
            )  # First 3 pages
            logger.info(f"Converted {len(images)} pages to images for LLM processing")
        except Exception as e:
            logger.warning(f"Could not convert PDF to images: {e}")
            images = None

        # Process with LLM
        enhanced_data = processor.process_document(parsed_data, images)

        logger.info("LLM processing completed successfully")
        return enhanced_data

    except ImportError as e:
        logger.error(f"LLM processing dependencies not available: {e}")
        logger.info("Falling back to traditional advanced features")
        return _enhance_with_advanced_features(parsed_data, file_path, "default")
    except Exception as e:
        logger.error(f"LLM processing failed: {e}")
        logger.info("Falling back to traditional advanced features")
        return _enhance_with_advanced_features(parsed_data, file_path, "default")


def _enhance_with_advanced_features(
    parsed_data: Dict[str, Any], file_path: Path, pipeline_type: str
) -> Dict[str, Any]:
    """Enhance parsed data with advanced features if available."""
    logger = get_logger(__name__)

    # Try to enhance with smart document classification
    try:
        from od_parse.intelligence import DocumentClassifier

        classifier = DocumentClassifier()
        classification_result = classifier.classify_document(parsed_data)

        # Convert to dict for JSON serialization
        parsed_data["document_classification"] = {
            "document_type": classification_result.document_type.value,
            "confidence": classification_result.confidence,
            "detected_patterns": classification_result.detected_patterns,
            "key_indicators": classification_result.key_indicators,
            "metadata": classification_result.metadata,
            "suggestions": classification_result.suggestions,
        }

        logger.info(
            f"Document classified as: {classification_result.document_type.value} "
            f"(confidence: {classification_result.confidence:.2f})"
        )

    except ImportError:
        logger.debug("Document classification not available")
    except Exception as e:
        logger.warning(f"Document classification failed: {e}")

    # Try to enhance with quality assessment
    try:
        from od_parse.quality import assess_document_quality

        quality_result = assess_document_quality(parsed_data)
        parsed_data["quality_assessment"] = quality_result
        logger.info(
            f"Quality assessment completed. Overall score: {quality_result.get('overall_score', 0):.2f}"
        )
    except ImportError:
        logger.debug("Quality assessment not available")
    except Exception as e:
        logger.warning(f"Quality assessment failed: {e}")

    # Try to enhance with TrOCR if focusing on text
    if pipeline_type in ["default", "full", "fast"]:
        try:
            from od_parse.ocr import TrOCREngine

            engine = TrOCREngine()
            if engine.is_available():
                logger.info("TrOCR enhancement available but requires image input")
        except ImportError:
            logger.debug("TrOCR not available")
        except Exception as e:
            logger.warning(f"TrOCR enhancement failed: {e}")

    # Try to enhance with multilingual processing
    if "text" in parsed_data and parsed_data["text"]:
        try:
            from od_parse.multilingual import detect_document_language

            text_content = parsed_data["text"]
            if isinstance(text_content, dict):
                text_content = text_content.get("content", "")

            if text_content:
                language_result = detect_document_language(text_content)
                parsed_data["language_detection"] = language_result
                logger.info(
                    f"Detected language: {language_result.get('language', 'unknown')}"
                )
        except ImportError:
            logger.debug("Multilingual processing not available")
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

    return parsed_data


def _create_summary(
    parsed_data: Dict[str, Any], file_path: Path, processing_time: float
) -> Dict[str, Any]:
    """Create a summary of the parsing results."""
    # Count extracted elements
    tables_count = len(parsed_data.get("tables", []))
    forms_count = len(parsed_data.get("forms", []))
    images_count = len(parsed_data.get("images", []))

    # Estimate text length
    text_content = parsed_data.get("text", "")
    if isinstance(text_content, dict):
        text_content = text_content.get("content", "")
    text_length = len(str(text_content))

    # Get quality score if available
    quality_score = None
    if "quality_assessment" in parsed_data:
        quality_score = parsed_data["quality_assessment"].get("overall_score")

    # Get detected language if available
    detected_language = None
    if "language_detection" in parsed_data:
        detected_language = parsed_data["language_detection"].get("language")

    return {
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
        "page_count": parsed_data.get("metadata", {}).get("page_count", "unknown"),
        "processing_time_seconds": processing_time,
        "extraction_statistics": {
            "text_length": text_length,
            "tables_extracted": tables_count,
            "form_fields_extracted": forms_count,
            "images_extracted": images_count,
            "handwritten_items_extracted": 0,  # Not implemented yet
            "structure_elements_extracted": 0,  # Not implemented yet
        },
        "quality_score": quality_score,
        "detected_language": detected_language,
        "has_advanced_features": any(
            ["quality_assessment" in parsed_data, "language_detection" in parsed_data]
        ),
    }


def _clean_for_json(obj):
    """Clean data structure for valid JSON serialization."""

    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            # Clean the key
            clean_key = str(key).strip()
            if not clean_key:
                clean_key = "unknown_key"

            # Clean the value
            cleaned[clean_key] = _clean_for_json(value)
        return cleaned

    elif isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]

    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    elif obj is None:
        return None

    elif isinstance(obj, str):
        # Handle unicode characters and clean up
        try:
            # Replace common problematic characters
            cleaned = obj.replace("\u2013", "-")  # em dash
            cleaned = cleaned.replace("\u2014", "--")  # en dash
            cleaned = cleaned.replace("\u2019", "'")  # right single quotation
            cleaned = cleaned.replace("\u201c", '"')  # left double quotation
            cleaned = cleaned.replace("\u201d", '"')  # right double quotation
            cleaned = cleaned.strip()
            return cleaned if cleaned else None
        except:
            return str(obj)

    else:
        try:
            # Try to convert to string, handle any encoding issues
            return str(obj)
        except:
            return None


def _write_output_file(
    result: Dict[str, Any], output_file: str, output_format: str, logger
) -> None:
    """Write the result to an output file."""
    output_path = Path(output_file)

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write output based on format
        if output_format == "json":
            # Clean the result for valid JSON
            cleaned_result = _clean_for_json(result)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_result, f, indent=2, ensure_ascii=False)
        elif output_format == "markdown":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.get("markdown_output", ""))
        elif output_format == "summary":
            # Clean the summary for valid JSON
            cleaned_summary = _clean_for_json(result.get("summary", {}))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Output written to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        raise


def parse_segmented(file_path: Union[str, Path], **kwargs) -> List[Dict[str, Any]]:
    """
    Adaptively parses a PDF, segmenting it if it contains multiple documents.

    Performs a fast pre-check to see if segmentation is needed. If so, it
    processes each detected document chunk individually.

    Args:
        file_path: Path to the PDF file.
        **kwargs: Additional arguments to pass to the underlying parser.

    Returns:
        A list of parsed result dictionaries, one for each detected document.
    """
    logger = get_logger(__name__)
    file_path = str(file_path)

    if not _is_likely_mixed_document(file_path):
        # Fast Path: Treat as a single document
        logger.info("Document appears uniform. Processing as a single file.")
        result = parse_pdf(file_path, **kwargs)
        return [result]

    # Intelligent Segmentation Path
    logger.info("Document appears to be mixed. Starting segmentation...")
    segmenter = DocumentSegmenter()
    chunks = segmenter.segment(file_path)

    if len(chunks) <= 1:
        logger.info("Segmentation resulted in a single chunk. Processing as one file.")
        result = parse_pdf(file_path, **kwargs)
        return [result]

    logger.info(
        f"Detected {len(chunks)} distinct document chunks. Processing each individually."
    )

    results = []
    reader = PyPDF2.PdfReader(file_path)

    for i, page_nums in enumerate(chunks):
        writer = PyPDF2.PdfWriter()
        logger.info(
            f"Processing chunk {i+1}/{len(chunks)} (Pages: {page_nums[0]}-{page_nums[-1]})..."
        )

        if not page_nums:
            continue

        for page_num in page_nums:
            # Page numbers in PyPDF2 are 0-indexed
            writer.add_page(reader.pages[page_num - 1])

        # Save the chunk to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            temp_path = temp_pdf.name
            writer.write(temp_path)

        try:
            # Parse the temporary PDF chunk
            chunk_result = parse_pdf(temp_path, **kwargs)
            # Add chunk info to the metadata
            chunk_result["metadata"]["chunk_info"] = {
                "chunk_index": i + 1,
                "total_chunks": len(chunks),
                "pages": page_nums,
            }
            results.append(chunk_result)
        finally:
            # Clean up the temporary file
            os.remove(temp_path)

    return results


def _is_likely_mixed_document(
    file_path: str, sample_size: int = 10, variance_threshold: float = 0.05
) -> bool:
    """
    Performs a fast pre-check on a sample of pages to determine if a PDF
    is likely to contain multiple different documents.

    Args:
        file_path: Path to the PDF file.
        sample_size: The number of pages to sample.
        variance_threshold: The threshold above which the document is considered mixed.

    Returns:
        True if the document is likely mixed, False otherwise.
    """
    logger = get_logger(__name__)
    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            if page_count <= sample_size or page_count <= 2:
                logger.info("Document is too short to require segmentation pre-check.")
                return False

            # Use stratified sampling to select pages
            indices = np.linspace(0, page_count - 1, sample_size, dtype=int)

            logger.info(f"Pre-check sampling pages: {list(p + 1 for p in indices)}")

            segmenter = DocumentSegmenter()
            fingerprints = []
            for i in indices:
                page = pdf.pages[i]
                # Extract single page image to conserve memory
                image = pdf2image.convert_from_path(
                    file_path, first_page=i + 1, last_page=i + 1
                )[0]
                text = page.extract_text() or ""
                fp = segmenter._get_page_fingerprint(np.array(image), text)
                fingerprints.append(fp)

            # Calculate variance of pairwise visual similarity
            if len(fingerprints) < 2:
                return False

            distances = []
            for i in range(len(fingerprints)):
                for j in range(i + 1, len(fingerprints)):
                    hash1 = fingerprints[i]["visual_hash"]
                    hash2 = fingerprints[j]["visual_hash"]
                    distance = hash1.compare(hash2)
                    distances.append(distance)

            variance = np.var(distances) / 64.0  # Normalize by hash length
            logger.info(
                f"Layout variance score: {variance:.4f} (Threshold: {variance_threshold})"
            )

            return variance > variance_threshold

    except Exception as e:
        logger.error(f"Failed to perform segmentation pre-check: {e}")
        # Err on the side of caution: if pre-check fails, assume segmentation is needed.
        return True


def main():
    """Main entry point for the command-line interface."""
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="OctonData Parse - Advanced PDF Parser"
    )

    parser.add_argument("file", help="Path to the PDF file to parse")

    parser.add_argument(
        "--output-format",
        choices=["json", "markdown", "summary"],
        default="json",
        help="Format for output",
    )

    parser.add_argument("--output-file", help="Path to output file")

    parser.add_argument(
        "--pipeline",
        choices=["default", "full", "fast", "tables", "forms", "structure"],
        default="default",
        help="Type of pipeline to use",
    )

    parser.add_argument(
        "--deep-learning",
        action="store_true",
        help="Use deep learning models for enhanced extraction",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Parse arguments
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    configure_logging(level=log_level)

    try:
        # Parse the PDF
        result = parse_pdf(
            file_path=args.file,
            output_format=args.output_format,
            output_file=args.output_file,
            pipeline_type=args.pipeline,
            use_deep_learning=args.deep_learning,
        )
        # Print summary if no output file
        if not args.output_file:
            if args.output_format == "json":
                # Clean the result for valid JSON output
                cleaned_result = _clean_for_json(result)
                print(json.dumps(cleaned_result, indent=2, ensure_ascii=False))
            elif args.output_format == "markdown":
                print(result.get("markdown_output", ""))
            elif args.output_format == "summary":
                summary = result.get("summary", {})
                print("\nDOCUMENT SUMMARY")
                print("================")
                print(f"File: {summary.get('file_name', 'Unknown')}")
                print(f"Size: {summary.get('file_size', 0)} bytes")
                print(f"Pages: {summary.get('page_count', 'Unknown')}")
                print("\nExtraction Statistics:")
                stats = summary.get("extraction_statistics", {})
                print(f"- Text Length: {stats.get('text_length', 0)} characters")
                print(f"- Tables: {stats.get('tables_extracted', 0)}")
                print(f"- Form Fields: {stats.get('form_fields_extracted', 0)}")
                print(f"- Images: {stats.get('images_extracted', 0)}")
                print(
                    f"- Handwritten Items: {stats.get('handwritten_items_extracted', 0)}"
                )
                print(
                    f"- Structure Elements: {stats.get('structure_elements_extracted', 0)}"
                )

                # Show quality score if available
                if summary.get("quality_score") is not None:
                    print(f"\nQuality Score: {summary.get('quality_score', 0):.2f}")

                # Show detected language if available
                if summary.get("detected_language"):
                    print(f"Detected Language: {summary.get('detected_language')}")

                print(
                    f"\nProcessing Time: {summary.get('processing_time_seconds', 0):.2f} seconds"
                )

        return 0

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error parsing PDF: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
