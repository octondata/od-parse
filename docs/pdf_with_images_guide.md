# Parsing PDFs with Text and Images Using Google Models

This guide explains how to use od-parse to parse PDFs containing both text and images using Google Gemini models.

## Quick Start

```python
from od_parse import parse_pdf

# Parse PDF with text and images using Google Gemini
result = parse_pdf(
    "your_document.pdf",
    llm_model="gemini-2.0-flash",  # Google Gemini with vision support
    api_keys={"google": "YOUR_GOOGLE_API_KEY"},
    output_format="json"
)

# Access results
print(f"Text: {result['parsed_data']['text']}")
print(f"Tables: {len(result['parsed_data']['tables'])}")
print(f"LLM Analysis: {result['parsed_data']['llm_analysis']}")
```

## How It Works

The library automatically handles PDFs with both text and images:

1. **Text Extraction**: Extracts text from the PDF using multiple methods
2. **Image Conversion**: Converts PDF pages to images for vision models
3. **Vision Processing**: Passes both text and images to Google Gemini (which supports vision)
4. **Enhanced Understanding**: Returns structured data with enhanced analysis

## Google Gemini Models with Vision

All Google Gemini models support vision (images):

| Model | Speed | Best For | Vision Support |
|-------|-------|----------|----------------|
| `gemini-2.0-flash` | Fast | General documents (default) | ✅ Yes |
| `gemini-2.0-flash-exp` | Fast | Experimental features | ✅ Yes |
| `gemini-1.5-pro` | Slower | Complex documents | ✅ Yes |
| `gemini-1.5-flash` | Fast | Quick processing | ✅ Yes |

## Examples

### Example 1: Basic Usage

```python
from od_parse import parse_pdf

# Your Google API key
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY_HERE"

# Parse PDF with images
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",
    api_keys={"google": GOOGLE_API_KEY},
    output_format="json"
)

# Access the results
print(f"Pages: {result['metadata']['num_pages']}")
print(f"Text length: {len(result['parsed_data']['text'])}")
print(f"Tables: {len(result['parsed_data']['tables'])}")
```

### Example 2: Using Environment Variable

```python
from od_parse import parse_pdf
import os

# Set environment variable first:
# export GOOGLE_API_KEY="your-api-key"

# The library automatically uses GOOGLE_API_KEY from environment
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",
    # No api_keys parameter - uses GOOGLE_API_KEY from environment
    output_format="json"
)
```

### Example 3: Save to File

```python
from od_parse import parse_pdf

result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",
    api_keys={"google": "YOUR_API_KEY"},
    output_format="json",
    output_file="output.json"  # Save to file
)
```

### Example 4: Different Google Models

```python
from od_parse import parse_pdf

models = [
    "gemini-2.0-flash",      # Fast, default
    "gemini-2.0-flash-exp",  # Experimental
    "gemini-1.5-pro",        # More powerful
    "gemini-1.5-flash"       # Fast alternative
]

for model in models:
    result = parse_pdf(
        "document.pdf",
        llm_model=model,
        api_keys={"google": "YOUR_API_KEY"},
        output_format="json"
    )
    print(f"{model}: {result['parsed_data']['llm_analysis']['document_type']}")
```

### Example 5: Accessing Enhanced Analysis

```python
from od_parse import parse_pdf

result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",
    api_keys={"google": "YOUR_API_KEY"},
    output_format="json"
)

# Access enhanced LLM analysis
llm_analysis = result['parsed_data']['llm_analysis']
print(f"Document type: {llm_analysis['document_type']}")
print(f"Key information: {llm_analysis['key_information']}")
print(f"Structured data: {llm_analysis['structured_data']}")
```

## What Gets Extracted

When parsing a PDF with text and images, the library extracts:

### Text Content
- All text from the PDF
- Structured text with formatting
- OCR text (if needed)

### Images
- Embedded images in the PDF
- Page images (for vision processing)
- Image metadata

### Tables
- All tables in the document
- Table structure and data
- Table relationships

### Forms
- Form fields and values
- Form structure
- Form metadata

### Enhanced Analysis (via LLM)
- Document type classification
- Key information extraction
- Structured data extraction
- Document understanding

## API Key Setup

### Option 1: Pass as Parameter (Recommended)

```python
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",
    api_keys={"google": "YOUR_GOOGLE_API_KEY"},
    output_format="json"
)
```

### Option 2: Environment Variable

```bash
# Set environment variable
export GOOGLE_API_KEY="your-api-key-here"
```

```python
# Use in code (library automatically uses environment variable)
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",
    output_format="json"
)
```

### Getting Your Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key

## Output Format

The library returns a structured dictionary:

```python
{
    "parsed_data": {
        "text": "Extracted text content...",
        "tables": [...],
        "forms": [...],
        "images": [...],
        "llm_analysis": {
            "document_type": "invoice",
            "key_information": {...},
            "structured_data": {...}
        }
    },
    "metadata": {
        "num_pages": 5,
        "file_size": 123456,
        "processing_time": 2.5
    },
    "summary": {
        "total_text_length": 5000,
        "tables_extracted": 2,
        "forms_extracted": 1
    }
}
```

## Best Practices

### 1. Use the Right Model

- **Fast processing**: `gemini-2.0-flash` or `gemini-1.5-flash`
- **Complex documents**: `gemini-1.5-pro`
- **Experimental features**: `gemini-2.0-flash-exp`

### 2. Handle Large PDFs

For large PDFs, consider:

```python
# Process in segments
from od_parse import parse_segmented

results = parse_segmented(
    "large_document.pdf",
    llm_model="gemini-2.0-flash",
    api_keys={"google": "YOUR_API_KEY"}
)
```

### 3. Error Handling

```python
from od_parse import parse_pdf

try:
    result = parse_pdf(
        "document.pdf",
        llm_model="gemini-2.0-flash",
        api_keys={"google": "YOUR_API_KEY"},
        output_format="json"
    )
    print("Success!")
except Exception as e:
    print(f"Error: {e}")
```

## Troubleshooting

### Issue: Images Not Being Processed

**Solution**: Ensure you're using a vision-capable model (all Google Gemini models support vision).

### Issue: API Key Error

**Solution**: 
- Check your API key is correct
- Ensure API key is passed correctly or set as environment variable
- Verify API key has proper permissions

### Issue: Slow Processing

**Solution**:
- Use `gemini-2.0-flash` or `gemini-1.5-flash` for faster processing
- Process in segments for very large PDFs
- Check your internet connection

## Next Steps

- See [api_keys_config.md](api_keys_config.md) for API key configuration
- See [llm_vs_vllm_guide.md](llm_vs_vllm_guide.md) for choosing between models
- See [README.md](../README.md) for general usage
- See [examples/pdf_with_images_google.py](../examples/pdf_with_images_google.py) for complete examples

