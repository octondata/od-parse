"""
Example: Parsing a PDF with both text and images using Google Gemini models.

This example demonstrates how to use od-parse to parse PDFs containing both
text and images, leveraging Google Gemini's vision capabilities.

The library automatically:
1. Extracts text from the PDF
2. Converts PDF pages to images
3. Passes both text and images to Google Gemini (which supports vision)
4. Returns enhanced structured data

All Google Gemini models support vision:
- gemini-2.0-flash (default, fast)
- gemini-2.0-flash-exp (experimental)
- gemini-1.5-pro (more powerful)
- gemini-1.5-flash (fast alternative)
"""

from od_parse import parse_pdf

def parse_pdf_with_images_google(pdf_path: str, google_api_key: str):
    """
    Parse a PDF with text and images using Google Gemini models.
    
    The library automatically:
    1. Extracts text from the PDF
    2. Converts PDF pages to images
    3. Passes both text and images to Google Gemini (which supports vision)
    4. Returns structured data with enhanced understanding
    
    Args:
        pdf_path: Path to the PDF file
        google_api_key: Your Google API key (Gemini API key)
    """
    
    # Parse PDF with Google Gemini model
    # The library automatically handles images for vision-capable models
    result = parse_pdf(
        pdf_path,
        llm_model="gemini-2.0-flash",  # Google Gemini model with vision support
        api_keys={"google": google_api_key},
        output_format="json",  # Get structured JSON output
        require_llm=True  # Ensure LLM processing is used
    )
    
    return result


def parse_pdf_with_custom_output(pdf_path: str, google_api_key: str, output_file: str):
    """
    Parse PDF and save to file.
    
    Args:
        pdf_path: Path to the PDF file
        google_api_key: Your Google API key
        output_file: Path to save the output JSON file
    """
    
    result = parse_pdf(
        pdf_path,
        llm_model="gemini-2.0-flash",  # Google Gemini with vision
        api_keys={"google": google_api_key},
        output_format="json",
        output_file=output_file,  # Save to file
        require_llm=True
    )
    
    print(f"Results saved to: {output_file}")
    return result


def parse_pdf_with_different_google_models(pdf_path: str, google_api_key: str):
    """
    Example showing different Google Gemini models you can use.
    
    All Google Gemini models support vision (images).
    """
    
    models = [
        "gemini-2.0-flash",      # Fast, default model
        "gemini-2.0-flash-exp",  # Experimental version
        "gemini-1.5-pro",        # More powerful, slower
        "gemini-1.5-flash"       # Fast alternative
    ]
    
    results = {}
    
    for model in models:
        print(f"\nParsing with {model}...")
        try:
            result = parse_pdf(
                pdf_path,
                llm_model=model,
                api_keys={"google": google_api_key},
                require_llm=True
            )
            results[model] = result
            print(f"✓ Successfully parsed with {model}")
        except Exception as e:
            print(f"✗ Error with {model}: {e}")
            results[model] = None
    
    return results


def example_basic_usage():
    """Basic usage example."""
    
    # Your Google API key
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
    
    # Path to your PDF file
    pdf_file = "path/to/your/document.pdf"
    
    # Parse the PDF
    result = parse_pdf_with_images_google(pdf_file, GOOGLE_API_KEY)
    
    # Access the results
    print("\n=== Parsed Results ===")
    print(f"Text extracted: {len(result.get('parsed_data', {}).get('text', ''))} characters")
    print(f"Pages: {result.get('metadata', {}).get('num_pages', 0)}")
    print(f"Tables found: {len(result.get('parsed_data', {}).get('tables', []))}")
    print(f"Images found: {len(result.get('parsed_data', {}).get('images', []))}")
    
    # Access enhanced LLM analysis
    if 'llm_analysis' in result.get('parsed_data', {}):
        llm_analysis = result['parsed_data']['llm_analysis']
        print(f"\n=== LLM Analysis ===")
        print(f"Document type: {llm_analysis.get('document_type', 'unknown')}")
        print(f"Key information: {llm_analysis.get('key_information', {})}")
    
    return result


def example_with_environment_variable():
    """
    Example using environment variable instead of passing API key directly.
    
    Set environment variable:
    export GOOGLE_API_KEY="your-api-key-here"
    """
    
    import os
    
    # The library will automatically use GOOGLE_API_KEY from environment
    pdf_file = "path/to/your/document.pdf"
    
    result = parse_pdf(
        pdf_file,
        llm_model="gemini-2.0-flash",  # Google Gemini with vision
        # No api_keys parameter - uses GOOGLE_API_KEY from environment
        output_format="json",
        require_llm=True
    )
    
    return result


def example_advanced_usage():
    """Advanced usage with custom options."""
    
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"
    pdf_file = "path/to/your/document.pdf"
    
    result = parse_pdf(
        pdf_file,
        llm_model="gemini-2.0-flash",  # Google Gemini with vision
        api_keys={"google": GOOGLE_API_KEY},
        output_format="json",
        output_file="output.json",  # Save to file
        use_deep_learning=True,  # Use advanced features
        require_llm=True,  # Ensure LLM processing
        for_embeddings=False  # Not optimizing for embeddings
    )
    
    return result


if __name__ == "__main__":
    # Example 1: Basic usage
    print("=== Example 1: Basic Usage ===")
    # Uncomment and set your API key and PDF path:
    # example_basic_usage()
    
    # Example 2: Using environment variable
    print("\n=== Example 2: Using Environment Variable ===")
    # Set GOOGLE_API_KEY environment variable first:
    # export GOOGLE_API_KEY="your-api-key"
    # example_with_environment_variable()
    
    # Example 3: Advanced usage
    print("\n=== Example 3: Advanced Usage ===")
    # example_advanced_usage()
    
    print("\n=== Quick Start ===")
    print("""
    To use this library with a PDF containing text and images:
    
    1. Get your Google API key from: https://makersuite.google.com/app/apikey
    
    2. Use the library:
    
       from od_parse import parse_pdf
       
       result = parse_pdf(
           "your_document.pdf",
           llm_model="gemini-2.0-flash",  # Google Gemini with vision
           api_keys={"google": "YOUR_API_KEY"},
           output_format="json"
       )
    
    3. The library automatically:
       - Extracts text from the PDF
       - Converts pages to images
       - Passes both to Google Gemini (which supports vision)
       - Returns enhanced structured data
    
    That's it! The library handles everything automatically.
    """)

