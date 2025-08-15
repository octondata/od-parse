"""
OctonData Parse - Main Interface Module

This module provides an easy-to-use interface to the PDF parsing capabilities
of the od-parse library, allowing users to extract rich content from documents.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core parsing functionality
try:
    from od_parse.parser import parse_pdf as core_parse_pdf
    from od_parse.converter import convert_to_markdown
    from od_parse.config import get_advanced_config
    from od_parse.utils.logging_utils import get_logger, configure_logging
except ImportError:
    # Fallback for when running from within od_parse directory
    from parser import parse_pdf as core_parse_pdf
    from converter import convert_to_markdown
    from config import get_advanced_config
    from utils.logging_utils import get_logger, configure_logging


def parse_pdf(
    file_path: Union[str, Path],
    output_format: str = "json",
    output_file: Optional[str] = None,
    pipeline_type: str = "default",
    use_deep_learning: bool = True
) -> Dict[str, Any]:
    """
    Parse a PDF file using the available parsing capabilities.

    Args:
        file_path: Path to the PDF file
        output_format: Format for output (json, markdown, text, summary)
        output_file: Path to output file (if None, no file is written)
        pipeline_type: Type of processing to use (default, full, fast, tables, forms, structure)
        use_deep_learning: Whether to use advanced features (if available)

    Returns:
        Dictionary containing parsed content
    """
    import time
    start_time = time.time()

    logger = get_logger(__name__)

    # Validate file path
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix.lower() == '.pdf':
        raise ValueError(f"File is not a PDF: {file_path}")

    logger.info(f"Processing document: {file_path}")
    logger.info(f"Pipeline type: {pipeline_type}")
    logger.info(f"Use advanced features: {use_deep_learning}")

    # Configure advanced features if requested
    config = get_advanced_config()
    if use_deep_learning:
        # Enable available advanced features
        config.enable_feature('trocr', check_dependencies=False)
        config.enable_feature('table_transformer', check_dependencies=False)
        config.enable_feature('quality_assessment', check_dependencies=False)
        config.enable_feature('multilingual', check_dependencies=False)

    # Parse the PDF using core functionality
    try:
        parsed_data = core_parse_pdf(str(file_path))
    except Exception as e:
        logger.error(f"Core parsing failed: {e}")
        # Create minimal result structure
        parsed_data = {
            "text": "",
            "tables": [],
            "forms": [],
            "images": [],
            "metadata": {"error": str(e)}
        }

    # Enhance with advanced features if available and requested
    if use_deep_learning:
        parsed_data = _enhance_with_advanced_features(parsed_data, file_path, pipeline_type)

    # Add processing metadata
    processing_time = time.time() - start_time
    file_stats = file_path.stat()

    # Create comprehensive result
    result = {
        "parsed_data": parsed_data,
        "metadata": {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size": file_stats.st_size,
            "processing_time_seconds": processing_time,
            "pipeline_type": pipeline_type,
            "use_deep_learning": use_deep_learning,
            "timestamp": time.time()
        },
        "summary": _create_summary(parsed_data, file_path, processing_time)
    }

    # Convert to requested format
    if output_format == "markdown":
        try:
            markdown_content = convert_to_markdown(parsed_data)
            result["markdown_output"] = markdown_content
        except Exception as e:
            logger.warning(f"Markdown conversion failed: {e}")
            result["markdown_output"] = f"# {file_path.name}\n\nMarkdown conversion failed: {e}"

    # Write output to file if specified
    if output_file:
        _write_output_file(result, output_file, output_format, logger)

    return result


def _enhance_with_advanced_features(parsed_data: Dict[str, Any], file_path: Path, pipeline_type: str) -> Dict[str, Any]:
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
            "suggestions": classification_result.suggestions
        }

        logger.info(f"Document classified as: {classification_result.document_type.value} "
                   f"(confidence: {classification_result.confidence:.2f})")

    except ImportError:
        logger.debug("Document classification not available")
    except Exception as e:
        logger.warning(f"Document classification failed: {e}")

    # Try to enhance with quality assessment
    try:
        from od_parse.quality import assess_document_quality
        quality_result = assess_document_quality(parsed_data)
        parsed_data["quality_assessment"] = quality_result
        logger.info(f"Quality assessment completed. Overall score: {quality_result.get('overall_score', 0):.2f}")
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
                logger.info(f"Detected language: {language_result.get('language', 'unknown')}")
        except ImportError:
            logger.debug("Multilingual processing not available")
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")

    return parsed_data


def _create_summary(parsed_data: Dict[str, Any], file_path: Path, processing_time: float) -> Dict[str, Any]:
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
            "structure_elements_extracted": 0   # Not implemented yet
        },
        "quality_score": quality_score,
        "detected_language": detected_language,
        "has_advanced_features": any([
            "quality_assessment" in parsed_data,
            "language_detection" in parsed_data
        ])
    }


def _clean_for_json(obj):
    """Clean data structure for valid JSON serialization."""
    import math

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
            cleaned = obj.replace('\u2013', '-')  # em dash
            cleaned = cleaned.replace('\u2014', '--')  # en dash
            cleaned = cleaned.replace('\u2019', "'")  # right single quotation
            cleaned = cleaned.replace('\u201c', '"')  # left double quotation
            cleaned = cleaned.replace('\u201d', '"')  # right double quotation
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


def _write_output_file(result: Dict[str, Any], output_file: str, output_format: str, logger) -> None:
    """Write the result to an output file."""
    output_path = Path(output_file)

    # Create directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Write output based on format
        if output_format == "json":
            # Clean the result for valid JSON
            cleaned_result = _clean_for_json(result)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_result, f, indent=2, ensure_ascii=False)
        elif output_format == "markdown":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result.get("markdown_output", ""))
        elif output_format == "summary":
            # Clean the summary for valid JSON
            cleaned_summary = _clean_for_json(result.get("summary", {}))
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Output written to: {output_path}")

    except Exception as e:
        logger.error(f"Failed to write output file: {e}")
        raise


def main():
    """Main entry point for the command-line interface."""
    # Configure argument parser
    parser = argparse.ArgumentParser(description="OctonData Parse - Advanced PDF Parser")
    
    parser.add_argument("file", help="Path to the PDF file to parse")
    
    parser.add_argument(
        "--output-format", 
        choices=["json", "markdown", "summary"], 
        default="json",
        help="Format for output"
    )
    
    parser.add_argument(
        "--output-file", 
        help="Path to output file"
    )
    
    parser.add_argument(
        "--pipeline", 
        choices=["default", "full", "fast", "tables", "forms", "structure"], 
        default="default",
        help="Type of pipeline to use"
    )
    
    parser.add_argument(
        "--deep-learning",
        action="store_true",
        help="Use deep learning models for enhanced extraction"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
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
            use_deep_learning=args.deep_learning
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
                print(f"- Handwritten Items: {stats.get('handwritten_items_extracted', 0)}")
                print(f"- Structure Elements: {stats.get('structure_elements_extracted', 0)}")

                # Show quality score if available
                if summary.get('quality_score') is not None:
                    print(f"\nQuality Score: {summary.get('quality_score', 0):.2f}")

                # Show detected language if available
                if summary.get('detected_language'):
                    print(f"Detected Language: {summary.get('detected_language')}")

                print(f"\nProcessing Time: {summary.get('processing_time_seconds', 0):.2f} seconds")
        
        return 0
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error parsing PDF: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
