"""
Advanced PDF Parsing Example

This script demonstrates the advanced PDF parsing capabilities of the od-parse library,
showing how to extract rich content from complex documents using various pipeline configurations.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.main import parse_pdf
from od_parse.advanced.pipeline import (
    PDFPipeline,
    LoadDocumentStage,
    BasicParsingStage,
    AdvancedParsingStage,
    TableExtractionStage,
    FormExtractionStage,
    HandwrittenContentStage,
    DocumentStructureStage,
    OutputFormattingStage,
)
from od_parse.utils.logging_utils import configure_logging


def simple_example(pdf_path):
    """
    Simple example of parsing a PDF using the default pipeline.

    Args:
        pdf_path: Path to the PDF file
    """
    print("\n=== SIMPLE EXAMPLE ===")
    print(f"Parsing PDF: {pdf_path}")

    # Parse the PDF using default settings
    result = parse_pdf(
        file_path=pdf_path,
        output_format="summary",
        pipeline_type="default",
        use_deep_learning=False,  # Use faster extraction without deep learning
    )

    # Print summary information
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
    print(f"- Handwritten Items: {stats.get('handwritten_items_extracted', 0)}")
    print(f"- Structure Elements: {stats.get('structure_elements_extracted', 0)}")

    print(f"\nProcessing Time: {summary.get('processing_time_seconds', 0):.2f} seconds")


def table_extraction_example(pdf_path):
    """
    Example of extracting tables from a PDF.

    Args:
        pdf_path: Path to the PDF file
    """
    print("\n=== TABLE EXTRACTION EXAMPLE ===")
    print(f"Extracting tables from PDF: {pdf_path}")

    # Create a custom pipeline for table extraction
    pipeline = PDFPipeline()
    pipeline.add_stage(LoadDocumentStage())
    pipeline.add_stage(
        TableExtractionStage({"use_neural": True})
    )  # Use neural network-based extraction
    pipeline.add_stage(OutputFormattingStage({"format": "json"}))

    # Process the PDF
    result = pipeline.process(pdf_path)

    # Print table information
    tables = result.get("tables", [])
    print(f"\nExtracted {len(tables)} tables")

    for i, table in enumerate(tables):
        print(f"\nTable {i+1} (Page {table.get('page_number')})")
        print(f"Rows: {table.get('rows')}, Columns: {table.get('cols')}")
        print(f"Confidence: {table.get('confidence', 0):.2f}")

        # Print markdown representation of the table
        print("\nMarkdown Table:")
        print(table.get("markdown", ""))


def form_extraction_example(pdf_path):
    """
    Example of extracting form fields from a PDF.

    Args:
        pdf_path: Path to the PDF file
    """
    print("\n=== FORM EXTRACTION EXAMPLE ===")
    print(f"Extracting form fields from PDF: {pdf_path}")

    # Parse the PDF using the forms pipeline
    result = parse_pdf(
        file_path=pdf_path, pipeline_type="forms", use_deep_learning=True
    )

    # Print form field information
    forms = result.get("forms", [])
    print(f"\nExtracted {len(forms)} form fields")

    for i, field in enumerate(forms):
        field_type = field.get("type", "unknown")
        label = field.get("label", "Unlabeled Field")
        value = field.get("value", "")
        page = field.get("page_number", 0)

        print(f"\nField {i+1} (Page {page})")
        print(f"Type: {field_type}")
        print(f"Label: {label}")

        if field_type == "checkbox":
            status = "Checked" if field.get("is_checked") else "Unchecked"
            print(f"Status: {status}")
        else:
            print(f"Value: {value}")


def document_structure_example(pdf_path):
    """
    Example of extracting document structure from a PDF.

    Args:
        pdf_path: Path to the PDF file
    """
    print("\n=== DOCUMENT STRUCTURE EXAMPLE ===")
    print(f"Extracting structure from PDF: {pdf_path}")

    # Parse the PDF using the structure pipeline
    result = parse_pdf(
        file_path=pdf_path, pipeline_type="structure", use_deep_learning=True
    )

    # Print structure information
    structure = result.get("structure", {})
    elements = structure.get("elements", [])
    print(f"\nExtracted {len(elements)} structural elements")

    for i, element in enumerate(elements):
        elem_type = element.get("type", "unknown")
        text = element.get("text", "")

        if elem_type == "heading":
            level = element.get("level", 1)
            print(f"\nHeading (Level {level}): {text}")
        elif elem_type == "paragraph":
            if len(text) > 100:
                text = text[:100] + "..."
            print(f"\nParagraph: {text}")
        elif elem_type == "list_item":
            print(f"\nList Item: {text}")
        else:
            print(f"\n{elem_type.capitalize()}: {text}")


def full_extraction_example(pdf_path, output_path=None):
    """
    Example of full PDF extraction with all capabilities.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the output
    """
    print("\n=== FULL EXTRACTION EXAMPLE ===")
    print(f"Performing full extraction on PDF: {pdf_path}")

    # Parse the PDF using the full pipeline
    result = parse_pdf(
        file_path=pdf_path,
        output_format="markdown",
        output_file=output_path,
        pipeline_type="full",
        use_deep_learning=True,
    )

    print("\nFull extraction completed")

    if output_path:
        print(f"Results saved to: {output_path}")
    else:
        # Print the first 500 characters of the markdown output
        markdown = result.get("markdown_output", "")
        preview = markdown[:500] + "..." if len(markdown) > 500 else markdown
        print(f"\nMarkdown Preview:\n\n{preview}")


def main():
    """Main function to run the examples."""
    # Configure logging
    configure_logging()

    # Check if a PDF file path is provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Please provide a path to a PDF file.")
        print("Usage: python advanced_pdf_parsing.py <pdf_path> [output_path]")
        return 1

    # Check if output path is provided
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    # Run the examples
    try:
        simple_example(pdf_path)
        table_extraction_example(pdf_path)
        form_extraction_example(pdf_path)
        document_structure_example(pdf_path)
        full_extraction_example(pdf_path, output_path)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
