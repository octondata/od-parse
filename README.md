# od-parse

An enterprise-grade library for parsing complex PDFs using advanced AI techniques. This library is designed to handle all types of PDF content with state-of-the-art accuracy, including:

- Vision Language Models (VLMs) like Qwen 2.5 VL for enhanced document understanding
- Deep learning-based text extraction and layout analysis
- Transformer-based OCR for handwritten content
- Neural table detection and extraction
- Form element understanding (radio buttons, checkboxes, text fields)
- Semantic structure extraction and document intelligence
- Multi-column layout detection

## Features

- **Vision Language Models (VLMs)**: Leverage state-of-the-art VLMs like Qwen 2.5 VL for enhanced document understanding
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

# Install with all advanced features
pip install "od-parse[all]"

# Install with specific advanced features
pip install "od-parse[trocr]"              # TrOCR text recognition
pip install "od-parse[table_transformer]"  # Neural table extraction
pip install "od-parse[llava_next]"         # Document understanding with VLMs
pip install "od-parse[multilingual]"       # Multi-language support
pip install "od-parse[quality_assessment]" # Quality assessment metrics
pip install "od-parse[async_processing]"   # Async processing capabilities

# Install preset combinations
pip install "od-parse[basic]"        # Essential features
pip install "od-parse[advanced]"     # All stable features
pip install "od-parse[experimental]" # All features including experimental
```

> **🔒 Privacy Note:** All processing is local by default. No API keys required for core functionality.

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

## Advanced Features

od-parse includes cutting-edge AI features that can be enabled as needed:

### Feature Configuration

```python
from od_parse.config import get_advanced_config

# Get configuration instance
config = get_advanced_config()

# Enable individual features
config.enable_feature('trocr')              # Transformer-based OCR
config.enable_feature('table_transformer')  # Neural table extraction
config.enable_feature('llava_next')         # Document understanding
config.enable_feature('multilingual')       # Multi-language support
config.enable_feature('quality_assessment') # Quality metrics
config.enable_feature('async_processing')   # Async processing

# Or use presets
config.enable_preset('advanced')  # Enable all stable features
```

### TrOCR Text Recognition

Superior text recognition using transformer models:

```python
from od_parse.ocr import TrOCREngine

engine = TrOCREngine(model_name="microsoft/trocr-base-printed")
result = engine.extract_text("document.png")
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
```

### Neural Table Extraction

Advanced table detection and extraction:

```python
from od_parse.advanced import TableTransformerEngine

engine = TableTransformerEngine()
result = engine.extract_tables("document_with_tables.png")
for table in result['tables']:
    print(f"Table confidence: {table['confidence']}")
    print(f"Structure: {table['structure']}")
```

### Document Understanding with LLaVA-NeXT

AI-powered document understanding:

```python
from od_parse.advanced import LLaVANextEngine

engine = LLaVANextEngine()
result = engine.understand_document(
    "complex_document.png",
    "Analyze this document and describe its structure and content."
)
print(f"Understanding: {result['understanding']}")
```

### Multi-Language Support

Process documents in multiple languages:

```python
from od_parse.multilingual import MultilingualProcessor

processor = MultilingualProcessor()

# Detect language
detection = processor.detect_language("Este es un documento en español.")
print(f"Language: {detection['language']}")

# Process with translation
result = processor.process_multilingual_text(
    "Bonjour, comment allez-vous?",
    target_language="en",
    include_translation=True
)
```

### Quality Assessment

Assess extraction quality and get recommendations:

```python
from od_parse.quality import assess_document_quality

quality = assess_document_quality(extraction_result)
print(f"Overall Score: {quality['overall_score']:.2f}")
for rec in quality['recommendations']:
    print(f"• {rec}")
```

### Async Processing

Process large batches asynchronously:

```python
import asyncio
from od_parse.async_processing import process_files_async

async def main():
    files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    results = await process_files_async(files, your_processor_function)

asyncio.run(main())
```

For detailed documentation on advanced features, see [ADVANCED_FEATURES.md](ADVANCED_FEATURES.md).

## 🔒 Privacy & External APIs

**od-parse is privacy-first and works completely offline by default.** No data leaves your machine unless you explicitly configure external APIs.

### Default Behavior (100% Local)
- ✅ PDF parsing, OCR, table extraction - all local
- ✅ Advanced AI models (TrOCR, LLaVA-NeXT) - download once, run locally
- ✅ Quality assessment and multilingual detection - local processing
- ✅ No API keys required for core functionality

### Optional External APIs
External APIs are **only** used for optional enhancements:

| Feature | External Service | Required? | Local Alternative |
|---------|-----------------|-----------|-------------------|
| Translation | Google Translate | No | Language detection only |
| Cloud VLMs | OpenAI/Anthropic | No | Local LLaVA-NeXT model |
| Cloud OCR | Azure/AWS | No | Local TrOCR/Tesseract |

### Configuring External APIs (Optional)

**Preferred Method - Environment Variables:**
```bash
# Only set these if you want external API features
export GOOGLE_API_KEY="your-google-translate-key"        # For translation
export OPENAI_API_KEY="your-openai-key"                  # For cloud VLMs
export ANTHROPIC_API_KEY="your-anthropic-key"            # For Claude Vision
```

**Alternative - Configuration File:**
```yaml
# ~/.config/od-parse/config.yaml
api_keys:
  google: "your-google-translate-key"
  openai: "your-openai-key"
```

**Check Configuration:**
```python
from od_parse.config import get_advanced_config

config = get_advanced_config()
print(f"Google Translate: {config.has_api_key('google')}")
print(f"OpenAI: {config.has_api_key('openai')}")
```

> **🛡️ Security Note:** All external APIs are optional. The library works fully without any API keys.

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

### Using Vision Language Models (VLMs)

The library now supports Vision Language Models like Qwen 2.5 VL to enhance document understanding by leveraging both visual and textual information:

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.advanced.vlm_processor import VLMProcessor

# Initialize the VLM processor with Qwen 2.5 VL
vlm_processor = VLMProcessor({
    "model": "qwen2.5-vl",  # Options: qwen2.5-vl, claude-3-opus-vision, gemini-pro-vision
    "api_key": "your-api-key",  # Or use environment variable
    "max_tokens": 2048,
    "temperature": 0.2,
    "system_prompt": "You are an expert document analyzer. Extract all information from this document."
})

# Parse a document with traditional methods first
parser = UnifiedPDFParser({"extract_tables": True, "extract_forms": True})
parsed_data = parser.parse("path/to/document.pdf")

# Convert the PDF to images for VLM processing
from pdf2image import convert_from_path
images = convert_from_path("path/to/document.pdf")
first_page_image = images[0]
first_page_image.save("first_page.png")

# Enhance parsing results with VLM analysis
enhanced_data = vlm_processor.enhance_parsing_results(parsed_data, "first_page.png")

# Extract tables specifically using VLM
tables = vlm_processor.extract_tables_with_vlm("first_page.png")

# Extract form fields using VLM
form_fields = vlm_processor.extract_form_fields_with_vlm("first_page.png")

# Process a document image directly with a custom prompt
vlm_analysis = vlm_processor.process_document_image(
    "first_page.png",
    "Analyze this document and identify any handwritten notes, signatures, or annotations."
)
print(vlm_analysis["analysis"])
```

#### Benefits of Using VLMs

- **Superior handling of complex layouts**: VLMs can understand document structure visually
- **Better extraction of handwritten content**: VLMs excel at recognizing handwriting
- **Improved table extraction**: VLMs can understand tables even with complex formatting
- **Form field detection**: VLMs can identify form fields and their values more accurately
- **Context-aware understanding**: VLMs consider both the visual layout and textual content

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

- VLM Dependencies (optional):
  - openai (for Qwen 2.5 VL)
  - anthropic (for Claude 3 Vision)
  - google-generativeai (for Gemini Pro Vision)
  - sentence-transformers (for local embedding models)

## Configuration System

The `od-parse` library now uses a flexible configuration system that allows you to customize all settings, including API endpoints, model names, and system prompts, without modifying the code:

### Configuration Files

You can create a configuration file in YAML or JSON format. The library will look for configuration files in the following locations:

1. Path specified when calling `load_config()`
2. `~/.config/od-parse/config.yaml` or `~/.config/od-parse/config.json`
3. `./od-parse.yaml` or `./od-parse.json` in the current working directory

Example configuration file (YAML):

```yaml
# API keys (IMPORTANT: For security, use environment variables instead of storing keys here)
# Uncomment and replace with your actual keys if needed
api_keys:
  # openai: "your-openai-api-key"  # Or set OPENAI_API_KEY environment variable
  # anthropic: "your-anthropic-api-key"  # Or set ANTHROPIC_API_KEY environment variable
  # cohere: "your-cohere-api-key"  # Or set COHERE_API_KEY environment variable
  # huggingface: "your-huggingface-api-key"  # Or set HUGGINGFACE_API_KEY environment variable
  # google: "your-google-api-key"  # Or set GOOGLE_API_KEY environment variable

# API endpoints
api_endpoints:
  openai: https://api.openai.com/v1
  anthropic: https://api.anthropic.com
  cohere: https://api.cohere.ai/v1

# VLM models
vlm_models:
  qwen: qwen2.5-vl
  claude: claude-3-opus-vision
  gemini: gemini-pro-vision

# Embedding models
embedding_models:
  openai: text-embedding-3-small
  cohere: embed-english-v3.0
  huggingface: sentence-transformers/all-mpnet-base-v2

# System prompts
system_prompts:
  document_analysis: >
    You are an expert document analyzer. Analyze the document image and extract all relevant information.
```

### API Keys Configuration

There are three ways to configure API keys for embedding models, VLMs, and other services:

1. **Using environment variables (recommended for security)**:

   ```bash
   # Standard API keys (recommended)
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export COHERE_API_KEY="your-cohere-api-key"
   export HUGGINGFACE_API_KEY="your-huggingface-api-key"
   export GOOGLE_API_KEY="your-google-api-key"
   
   # Library-specific API keys (alternative)
   export OD_PARSE_OPENAI_API_KEY="your-openai-api-key"
   export OD_PARSE_ANTHROPIC_API_KEY="your-anthropic-api-key"
   ```

2. **In the configuration file** (less secure, but convenient for development):

   ```yaml
   # In od-parse.yaml or ~/.config/od-parse/config.yaml
   api_keys:
     openai: "your-openai-api-key"
     anthropic: "your-anthropic-api-key"
     cohere: "your-cohere-api-key"
   ```

3. **Directly in code** when initializing components:

   ```python
   vector_storage = VectorStorage({
       "embedding_model": "openai",
       "api_key": "your-openai-api-key"  # Takes precedence over config files and env vars
   })
   ```

### Other Environment Variables

You can also configure other settings using environment variables:

```bash
# API endpoints
export OD_PARSE_OPENAI_API_URL="https://api.openai.com/v1"
export OD_PARSE_ANTHROPIC_API_URL="https://api.anthropic.com"

# VLM models
export OD_PARSE_QWEN_MODEL="qwen2.5-vl"
export OD_PARSE_CLAUDE_MODEL="claude-3-opus-vision"

# Embedding models
export OD_PARSE_OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
```

### Using the Configuration System in Code

```python
from od_parse.config.settings import load_config, get_config

# Load configuration from file
load_config("/path/to/config.yaml")

# Get configuration values
api_endpoint = get_config("api_endpoints.openai")
qwen_model = get_config("vlm_models.qwen")

# Get nested configuration with default value
chroma_path = get_config("vector_db.chroma.default_path", "./default_chroma_db")
```

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
