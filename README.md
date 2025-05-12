# od-parse

A powerful library for parsing complex PDFs and generating Markdown output. This library is designed to handle various types of PDF content, including:

- Text extraction
- Image extraction
- Table detection and extraction
- Form element detection (radio buttons, checkboxes)
- Handwritten content recognition via OCR

## Features

- **Comprehensive PDF Parsing**: Extract all content types from complex PDFs
- **Handwritten Content Recognition**: Use OCR to extract handwritten text
- **Table Detection**: Identify and extract tabular data
- **Form Element Detection**: Recognize form elements like checkboxes and radio buttons
- **Markdown Generation**: Convert parsed content into clean Markdown format
- **Modular Design**: Use only the components you need

## Installation

```bash
# Install the basic package
pip install od-parse

# Install with all dependencies
pip install "od-parse[all]"

# Install with specific feature dependencies
pip install "od-parse[ocr]"  # For OCR capabilities
pip install "od-parse[tables]"  # For table extraction
```

## Quick Start

```python
from od_parse import parse_pdf, convert_to_markdown

# Parse a PDF file
parsed_data = parse_pdf("path/to/document.pdf")

# Convert to Markdown
markdown = convert_to_markdown(parsed_data)

# Save to file
with open("output.md", "w") as f:
    f.write(markdown)
```

## Advanced Usage

### Extracting Specific Content Types

```python
from od_parse.parser import extract_text, extract_images, extract_tables, extract_forms
from od_parse.ocr import extract_handwritten_content

# Extract text only
text = extract_text("path/to/document.pdf")

# Extract images
images = extract_images("path/to/document.pdf", output_dir="./extracted_images")

# Extract tables
tables = extract_tables("path/to/document.pdf")

# Extract form elements
forms = extract_forms("path/to/document.pdf")

# Extract handwritten content from an image
handwritten = extract_handwritten_content("path/to/image.png")
```

### Customizing Markdown Output

```python
from od_parse import parse_pdf, convert_to_markdown

parsed_data = parse_pdf("path/to/document.pdf")

# Customize Markdown output
markdown = convert_to_markdown(
    parsed_data,
    output_file="output.md",
    include_images=True,
    include_tables=True,
    include_forms=False,
    include_handwritten=True
)
```

## Requirements

- Python 3.8+
- Dependencies:
  - pdfminer.six
  - pytesseract
  - Pillow
  - pdf2image
  - opencv-python
  - tabula-py
  - numpy
  - markdown

## License

MIT License
