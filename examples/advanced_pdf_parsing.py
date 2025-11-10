"""
Advanced PDF Parsing Example

This script demonstrates the advanced PDF parsing capabilities of the od-parse library,
showing how to extract rich content from complex documents using various configurations.

Note: od-parse requires LLM API keys. Set GOOGLE_API_KEY environment variable
or pass api_keys parameter.
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse import parse_pdf
from od_parse.utils.logging_utils import configure_logging


def simple_example(pdf_path, api_key=None):
    """
    Simple example of parsing a PDF using the default pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
    """
    print("\n=== SIMPLE EXAMPLE ===")
    print(f"Parsing PDF: {pdf_path}")
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY environment variable")
        return
    
    # Parse the PDF using default settings
    result = parse_pdf(
        file_path=pdf_path,
        llm_model="gemini-2.0-flash",
        api_keys={"google": api_key},
        output_format="summary",
        pipeline_type="default",
        use_deep_learning=False  # Use faster extraction without deep learning
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


def table_extraction_example(pdf_path, api_key=None):
    """
    Example of extracting tables from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
    """
    print("\n=== TABLE EXTRACTION EXAMPLE ===")
    print(f"Extracting tables from PDF: {pdf_path}")
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY environment variable")
        return
    
    # Parse PDF with focus on table extraction
    result = parse_pdf(
        file_path=pdf_path,
        llm_model="gemini-2.0-flash",
        api_keys={"google": api_key},
        output_format="json",
        pipeline_type="default",
        use_deep_learning=True  # Use deep learning for better table extraction
    )
    
    # Print table information
    parsed_data = result.get("parsed_data", {})
    tables = parsed_data.get("tables", [])
    print(f"\nExtracted {len(tables)} tables")
    
    for i, table in enumerate(tables):
        page = table.get('page', table.get('page_number', 'unknown'))
        print(f"\nTable {i+1} (Page {page})")
        rows = table.get('rows', table.get('num_rows', 'unknown'))
        cols = table.get('cols', table.get('num_cols', 'unknown'))
        print(f"Rows: {rows}, Columns: {cols}")
        if 'confidence' in table:
            print(f"Confidence: {table.get('confidence', 0):.2f}")
        
        # Print table data if available
        if 'data' in table:
            print("\nTable Data (first few rows):")
            for row in table['data'][:3]:
                print(f"  {row}")


def form_extraction_example(pdf_path, api_key=None):
    """
    Example of extracting form fields from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
    """
    print("\n=== FORM EXTRACTION EXAMPLE ===")
    print(f"Extracting form fields from PDF: {pdf_path}")
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY environment variable")
        return
    
    # Parse the PDF using the forms pipeline
    result = parse_pdf(
        file_path=pdf_path,
        llm_model="gemini-2.0-flash",
        api_keys={"google": api_key},
        pipeline_type="forms",
        use_deep_learning=True
    )
    
    # Print form field information
    parsed_data = result.get("parsed_data", {})
    forms = parsed_data.get("forms", [])
    print(f"\nExtracted {len(forms)} forms")
    
    for i, form in enumerate(forms):
        form_id = form.get("form_id", f"form_{i+1}")
        page = form.get("page", form.get("page_number", 0))
        fields = form.get("fields", [])
        
        print(f"\nForm {i+1} (ID: {form_id}, Page {page})")
        print(f"Fields: {len(fields)}")
        
        for j, field in enumerate(fields[:5]):  # Show first 5 fields
            field_type = field.get("type", "unknown")
            name = field.get("name", field.get("label", "Unlabeled Field"))
            value = field.get("value", "")
            
            print(f"  Field {j+1}: {name} ({field_type})")
            if field_type == "checkbox":
                status = "Checked" if field.get("is_checked", field.get("value")) else "Unchecked"
                print(f"    Status: {status}")
            elif value:
                print(f"    Value: {value}")
        
        if len(fields) > 5:
            print(f"  ... and {len(fields) - 5} more fields")


def document_structure_example(pdf_path, api_key=None):
    """
    Example of extracting document structure from a PDF.
    
    Args:
        pdf_path: Path to the PDF file
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
    """
    print("\n=== DOCUMENT STRUCTURE EXAMPLE ===")
    print(f"Extracting structure from PDF: {pdf_path}")
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY environment variable")
        return
    
    # Parse the PDF using the structure pipeline
    result = parse_pdf(
        file_path=pdf_path,
        llm_model="gemini-2.0-flash",
        api_keys={"google": api_key},
        pipeline_type="structure",
        use_deep_learning=True
    )
    
    # Print structure information
    parsed_data = result.get("parsed_data", {})
    structure = parsed_data.get("structure", {})
    elements = structure.get("elements", [])
    
    if not elements:
        # Try to get structure from LLM analysis
        llm_analysis = parsed_data.get("llm_analysis", {})
        if llm_analysis:
            print(f"\nDocument Type: {llm_analysis.get('document_type', 'unknown')}")
            print(f"Key Information: {llm_analysis.get('key_information', {})}")
    
    print(f"\nExtracted {len(elements)} structural elements")
    
    for i, element in enumerate(elements[:10]):  # Show first 10 elements
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
    
    if len(elements) > 10:
        print(f"\n... and {len(elements) - 10} more elements")


def full_extraction_example(pdf_path, output_path=None, api_key=None):
    """
    Example of full PDF extraction with all capabilities.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the output
        api_key: Optional API key (uses GOOGLE_API_KEY env var if not provided)
    """
    print("\n=== FULL EXTRACTION EXAMPLE ===")
    print(f"Performing full extraction on PDF: {pdf_path}")
    
    # Get API key from parameter or environment
    api_key = api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY environment variable")
        return
    
    # Parse the PDF using the full pipeline
    result = parse_pdf(
        file_path=pdf_path,
        llm_model="gemini-2.0-flash",
        api_keys={"google": api_key},
        output_format="markdown",
        output_file=output_path,
        pipeline_type="full",
        use_deep_learning=True
    )
    
    print("\nFull extraction completed")
    
    if output_path:
        print(f"Results saved to: {output_path}")
    else:
        # Print summary information
        parsed_data = result.get("parsed_data", {})
        print(f"\nExtracted Content:")
        print(f"- Text: {len(parsed_data.get('text', ''))} characters")
        print(f"- Tables: {len(parsed_data.get('tables', []))} tables")
        print(f"- Forms: {len(parsed_data.get('forms', []))} forms")
        print(f"- Images: {len(parsed_data.get('images', []))} images")


def main():
    """Main function to run the examples."""
    # Configure logging
    configure_logging()
    
    # Check if a PDF file path is provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        print("Please provide a path to a PDF file.")
        print("Usage: python advanced_pdf_parsing.py <pdf_path> [output_path] [api_key]")
        print("\nNote: API key can also be set via GOOGLE_API_KEY environment variable")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return 1
    
    # Check if output path is provided
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Check if API key is provided
    api_key = None
    if len(sys.argv) > 3:
        api_key = sys.argv[3]
    elif not os.getenv('GOOGLE_API_KEY'):
        print("Warning: No API key provided. Set GOOGLE_API_KEY environment variable or pass as third argument")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
    
    # Run the examples
    try:
        simple_example(pdf_path, api_key)
        table_extraction_example(pdf_path, api_key)
        form_extraction_example(pdf_path, api_key)
        document_structure_example(pdf_path, api_key)
        full_extraction_example(pdf_path, output_path, api_key)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
