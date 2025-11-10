#!/usr/bin/env python3
"""
Basic usage example for the od-parse library.

This example demonstrates basic PDF parsing and conversion to Markdown.
Note: od-parse requires LLM API keys. Set GOOGLE_API_KEY environment variable
or pass api_keys parameter.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse import parse_pdf, convert_to_markdown

def main():
    """
    Parse a PDF file and convert it to Markdown.
    """
    parser = argparse.ArgumentParser(description='Parse a PDF file and convert it to Markdown.')
    parser.add_argument('pdf_file', help='Path to the PDF file to parse')
    parser.add_argument('--output', '-o', help='Path to the output Markdown file')
    parser.add_argument('--api-key', help='Google API key (or set GOOGLE_API_KEY env var)')
    parser.add_argument('--no-images', action='store_true', help='Exclude images from the output')
    parser.add_argument('--no-tables', action='store_true', help='Exclude tables from the output')
    parser.add_argument('--no-forms', action='store_true', help='Exclude form elements from the output')
    parser.add_argument('--no-handwritten', action='store_true', help='Exclude handwritten content from the output')
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output:
        pdf_path = Path(args.pdf_file)
        args.output = pdf_path.with_suffix('.md')
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Error: API key required. Set GOOGLE_API_KEY environment variable or use --api-key")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return 1
    
    print(f"Parsing PDF file: {args.pdf_file}")
    
    # Parse PDF with API key
    result = parse_pdf(
        args.pdf_file,
        llm_model="gemini-2.0-flash",  # Google Gemini model
        api_keys={"google": api_key},
        output_format="raw"  # Get raw data structure
    )
    
    # Extract parsed_data from result
    parsed_data = result.get('parsed_data', {})
    
    print(f"Converting to Markdown: {args.output}")
    markdown = convert_to_markdown(
        parsed_data,
        output_file=args.output,
        include_images=not args.no_images,
        include_tables=not args.no_tables,
        include_forms=not args.no_forms,
        include_handwritten=not args.no_handwritten
    )
    
    print(f"Markdown saved to: {args.output}")
    
    # Print a summary of extracted content
    print("\nExtracted content summary:")
    print(f"- Text: {len(parsed_data.get('text', ''))} characters")
    print(f"- Images: {len(parsed_data.get('images', []))} images")
    print(f"- Tables: {len(parsed_data.get('tables', []))} tables")
    print(f"- Form elements: {len(parsed_data.get('forms', []))} form elements")
    print(f"- Handwritten content: {len(parsed_data.get('handwritten_content', []))} items")
    
    return 0

if __name__ == '__main__':
    main()
