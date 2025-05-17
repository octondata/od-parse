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
- **Flexible Configuration Options**: Customize parsing capabilities based on your needs
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

### Using Different Configuration Profiles

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser

# Example: Fast extraction (no deep learning)
fast_parser = UnifiedPDFParser({
    "use_deep_learning": False,
    "extract_tables": True,
    "extract_structure": False,
    "extract_handwritten": False
})
fast_result = fast_parser.parse("path/to/document.pdf")

# Example: Table extraction focus
table_parser = UnifiedPDFParser({
    "use_deep_learning": True,
    "extract_tables": True,
    "extract_forms": False,
    "extract_structure": False
})
table_result = table_parser.parse("path/to/document.pdf")

# Example: Form processing focus
form_parser = UnifiedPDFParser({
    "use_deep_learning": True,
    "extract_forms": True,
    "extract_tables": False
})
form_result = form_parser.parse("path/to/document.pdf")
```

### Enterprise Integrations

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.advanced.integrations import JSONFileConnector, CSVConnector, DatabaseConnector, VectorDBConnector

# Process a document with the unified parser
parser = UnifiedPDFParser()
result = parser.parse("path/to/document.pdf")

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

### Configurable Vector Storage for RAG Applications

The `od-parse` library provides a flexible vector storage system that supports various embedding models and vector databases:

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.advanced.vector_storage import VectorStorage

def process_document(document_path, config=None):
    # Default configuration for the parser
    if config is None:
        config = {
            "use_deep_learning": True,
            "extract_tables": True,
            "extract_forms": True,
            "extract_structure": True
        }
    
    # Parse the document
    parser = UnifiedPDFParser(config)
    parsed_data = parser.parse(document_path)
    
    # Configure vector storage with your preferred embedding model and database
    vector_storage = VectorStorage({
        # Embedding model configuration
        "embedding_model": "openai",  # Options: openai, huggingface, cohere, custom
        "embedding_model_name": "text-embedding-3-small",
        "api_key": "your-api-key",  # Or use environment variable
        
        # Vector database configuration
        "vector_db": "pgvector",  # Options: pgvector, qdrant, pinecone, milvus, weaviate, chroma, json
        "connection_string": "postgresql://user:password@localhost:5432/vectordb",
        
        # Chunking configuration
        "chunk_size": 1000,
        "chunk_overlap": 100
    })
    
    # Create embeddings from parsed data
    embeddings = vector_storage.create_embeddings(parsed_data)
    
    # Store embeddings in the configured vector database
    vector_storage.store_embeddings(embeddings)
    
    return parsed_data
```

#### Using Custom Embedding Models

You can also use your own custom embedding models:

```python
# Create a custom embedding implementation
class MyEmbeddingModel:
    def __init__(self, model_path, **kwargs):
        # Load your model here
        self.model = load_my_model(model_path)
    
    def embed(self, text):
        # Generate embeddings using your model
        return self.model.generate_embedding(text)

# Save this to my_embeddings.py

# Then use it with VectorStorage
vector_storage = VectorStorage({
    "embedding_model": "custom",
    "custom_embedding_module": "/path/to/my_embeddings.py",
    "custom_embedding_class": "MyEmbeddingModel",
    "custom_embedding_params": {
        "model_path": "/path/to/my/model"
    }
})
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
