"""
OctonData Parse - Main Interface Module

This module provides an easy-to-use interface to the advanced PDF parsing capabilities
of the od-parse library, allowing users to extract rich content from complex documents.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

from od_parse.advanced.pipeline import (
    AdvancedParsingStage,
    BasicParsingStage,
    DocumentStructureStage,
    FormExtractionStage,
    HandwrittenContentStage,
    LoadDocumentStage,
    OutputFormattingStage,
    PDFPipeline,
    TableExtractionStage,
)
from od_parse.utils.logging_utils import configure_logging, get_logger


def parse_pdf(
    file_path: Union[str, Path],
    output_format: str = "json",
    output_file: Optional[str] = None,
    pipeline_type: str = "default",
    use_deep_learning: bool = True,
) -> Dict[str, Any]:
    """
    Parse a PDF file using the unified parser.

    Args:
        file_path: Path to the PDF file
        output_format: Format for output (json, markdown, text, summary)
        output_file: Path to output file (if None, no file is written)
        pipeline_type: Type of pipeline to use (default, full, fast, tables, forms, structure)
        use_deep_learning: Whether to use deep learning models

    Returns:
        Dictionary containing parsed content
    """
    logger = get_logger(__name__)

    # Validate file path
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.suffix.lower() == ".pdf":
        raise ValueError(f"File is not a PDF: {file_path}")

    # Select pipeline based on type
    logger.info(f"Using pipeline type: {pipeline_type}")

    if pipeline_type == "full":
        pipeline = PDFPipeline.create_full_pipeline()
    elif pipeline_type == "fast":
        pipeline = PDFPipeline.create_fast_pipeline()
    elif pipeline_type == "tables":
        pipeline = PDFPipeline.create_tables_pipeline()
    elif pipeline_type == "forms":
        pipeline = PDFPipeline.create_forms_pipeline()
    elif pipeline_type == "structure":
        pipeline = PDFPipeline.create_structure_pipeline()
    else:
        # Custom default pipeline
        pipeline = PDFPipeline()
        pipeline.add_stage(LoadDocumentStage())

        if use_deep_learning:
            pipeline.add_stage(AdvancedParsingStage())
        else:
            pipeline.add_stage(BasicParsingStage())

        pipeline.add_stage(TableExtractionStage({"use_neural": use_deep_learning}))

        if use_deep_learning:
            pipeline.add_stage(FormExtractionStage())
            pipeline.add_stage(HandwrittenContentStage())
            pipeline.add_stage(DocumentStructureStage())

        pipeline.add_stage(OutputFormattingStage({"format": output_format}))

    # Process the document
    logger.info(f"Processing document: {file_path}")
    result = pipeline.process(file_path)

    # Write output to file if specified
    if output_file:
        output_path = Path(output_file)

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write output based on format
        if output_format == "json" or output_format == "summary":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
        elif output_format == "markdown":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.get("markdown_output", ""))

        logger.info(f"Output written to: {output_path}")

    return result


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
                print(json.dumps(result, indent=2))
            elif args.output_format == "markdown":
                print(result.get("markdown_output", ""))
            elif args.output_format == "summary":
                summary = result.get("summary", {})
                print("\nDOCUMENT SUMMARY")
                print("================")
                print(f"File: {summary.get('file_name')}")
                print(f"Size: {summary.get('file_size')} bytes")
                print(f"Pages: {summary.get('page_count')}")
                print("\nExtraction Statistics:")
                stats = summary.get("extraction_statistics", {})
                print(f"- Tables: {stats.get('tables_extracted', 0)}")
                print(f"- Form Fields: {stats.get('form_fields_extracted', 0)}")
                print(
                    f"- Handwritten Items: {stats.get('handwritten_items_extracted', 0)}"
                )
                print(
                    f"- Structure Elements: {stats.get('structure_elements_extracted', 0)}"
                )
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
