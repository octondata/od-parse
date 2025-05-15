# od-parse

An enterprise-grade library for parsing complex PDFs using advanced AI techniques. This library is designed to handle all types of PDF content with state-of-the-art accuracy, including:

- Deep learning-based text extraction and layout analysis
- Transformer-based OCR for handwritten content
- Neural table detection and extraction
- Form element understanding (radio buttons, checkboxes, text fields)
- Semantic structure extraction and document intelligence
- Multi-column layout detection

## Features

- **Advanced Document Intelligence**: Understand document structure and content relationships
- **Neural Layout Analysis**: Detect complex layouts, including multi-column structures
- **Deep Learning Table Extraction**: Extract tables even without explicit borders
- **Form Understanding**: Detect and interpret various form elements
- **Transformer-based OCR**: Extract handwritten text with state-of-the-art accuracy
- **Semantic Structure Analysis**: Identify hierarchical structures and relationships
- **Flexible Pipeline Architecture**: Configure custom processing workflows
- **Enterprise Integrations**: Connect with databases, APIs, and vector stores
- **Markdown Generation**: Convert parsed content into clean Markdown format

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

### Using the Unified Parser

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser

# Initialize the parser with custom configuration
parser = UnifiedPDFParser({
    "use_deep_learning": True,
    "extract_handwritten": True,
    "extract_tables": True,
    "extract_forms": True,
    "extract_structure": True,
    "output_format": "json"
})

# Parse a PDF file
result = parser.parse("path/to/document.pdf")

# Convert to markdown
markdown = parser.to_markdown(result)

# Save markdown to file
with open("output.md", "w") as f:
    f.write(markdown)
```

### Using the Pipeline Architecture

```python
from od_parse.advanced.pipeline import (
    PDFPipeline,
    LoadDocumentStage,
    BasicParsingStage,
    TableExtractionStage,
    FormExtractionStage,
    HandwrittenContentStage,
    DocumentStructureStage,
    OutputFormattingStage
)

# Create a custom pipeline
pipeline = PDFPipeline()
pipeline.add_stage(LoadDocumentStage())
pipeline.add_stage(BasicParsingStage())
pipeline.add_stage(TableExtractionStage({"use_neural": True}))
pipeline.add_stage(FormExtractionStage())
pipeline.add_stage(OutputFormattingStage({"format": "json"}))

# Process a document
result = pipeline.process("path/to/document.pdf")

# Or use a pre-configured pipeline
tables_pipeline = PDFPipeline.create_tables_pipeline()
result = tables_pipeline.process("path/to/document.pdf")
```

### Enterprise Integrations

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import JSONFileConnector, CSVConnector, DatabaseConnector, VectorDBConnector

# Process a document
pipeline = PDFPipeline.create_full_pipeline()
result = pipeline.process("path/to/document.pdf")

# Export to various formats
json_connector = JSONFileConnector({"file_path": "output.json"})
json_connector.export(result)

csv_connector = CSVConnector({"file_path": "tables.csv"})
csv_connector.export(result)

# Export to database
db_connector = DatabaseConnector({
    "db_type": "sqlite",
    "db_path": "documents.db"
})
db_connector.export(result)

# Export to vector database for RAG applications
vector_connector = VectorDBConnector({
    "db_type": "pgvector",
    "conn_string": "postgresql://user:password@localhost:5432/vectordb"
})
vector_connector.export(result)
```

## Requirements

- Python 3.8+
- Base Dependencies:
  - pdfminer.six
  - Pillow
  - pdf2image
  - opencv-python
  - tabula-py
  - numpy
  - pandas

- Advanced Dependencies (optional):
  - PyTorch
  - Hugging Face Transformers
  - pytesseract
  - psycopg2-binary
  - sqlalchemy
  - requests

## Command Line Interface

The library includes a command-line interface for quick PDF processing:

```bash
# Basic usage
python -m od_parse.main document.pdf --output-file output.json

# Advanced usage with pipeline selection
python -m od_parse.main document.pdf --pipeline full --output-format markdown --output-file output.md

# Fast processing (optimized for speed)
python -m od_parse.main document.pdf --pipeline fast --output-format summary

# Extract only tables
python -m od_parse.main document.pdf --pipeline tables --output-format json

# Use deep learning capabilities
python -m od_parse.main document.pdf --deep-learning
```

## Enterprise Applications

This library is designed for enterprise AI applications:

- **RAG Systems**: Extract and embed document content for retrieval-augmented generation
- **AI Agents**: Provide structured data for AI agents to work with
- **Document Automation**: Automate document processing workflows
- **Data Unification**: Extract structured data from unstructured documents
- **Knowledge Bases**: Build searchable knowledge bases from document repositories

## License

MIT License
  - numpy
  - markdown

## License

MIT License
