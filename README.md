# od-parse

> **🤖 LLM-First Document Parser:** od-parse now requires LLM API keys for advanced document understanding. See [Installation](#installation) for setup instructions.

An enterprise-grade, LLM-powered library for parsing complex PDFs using state-of-the-art language models. This library leverages the power of GPT-4, Claude, and Gemini for superior document understanding, including:

- **LLM-Powered Analysis**: GPT-4, Claude 3.5 Sonnet, Gemini 1.5 Pro for complex document understanding
- **Vision-Language Models**: Advanced multimodal AI for visual document analysis
- **Intelligent Document Classification**: 54+ document types with context-aware processing
- **Complex Layout Understanding**: Multi-column, tables, forms, handwriting recognition
- **Domain-Specific Processing**: Tax forms, legal contracts, medical records, financial statements
- **Structured Data Extraction**: JSON output with high accuracy and validation
- **Cost-Optimized Model Selection**: Automatic model selection based on document complexity

## Features

- **🤖 LLM-Powered Document Understanding**: GPT-4, Claude 3.5 Sonnet, Gemini 1.5 Pro for complex document analysis
- **🎯 Smart Document Classification**: Automatically identify 54+ document types with intelligent routing
- **👁️ Vision-Language Processing**: Multimodal AI for visual document understanding and layout analysis
- **📊 Domain-Specific Extraction**: Specialized processing for tax forms, legal contracts, medical records, financial statements
- **🏗️ Structured Data Output**: High-accuracy JSON extraction with validation and confidence scoring
- **💰 Cost-Optimized Processing**: Automatic model selection based on document complexity and cost
- **🔄 Intelligent Fallbacks**: Graceful degradation when LLM services are unavailable
- **🌍 Multi-Provider Support**: OpenAI, Anthropic, Google, Azure OpenAI, and local models
- **📈 Enterprise-Ready**: Batch processing, API integrations, and scalable architecture
- **🔒 Privacy-Conscious**: Support for local models and on-premises deployment

## Installation

> **🤖 LLM Required:** od-parse now requires LLM API keys for document processing. Set up your API keys before installation.

> **📦 Installation:** od-parse is distributed via GitHub. Install with `pip install git+https://github.com/octondata/od-parse.git` or download wheel files from [GitHub Releases](https://github.com/octondata/od-parse/releases).

### Step 1: Set Up LLM API Keys

Choose one or more LLM providers and set up your API keys:

```bash
# OpenAI (Recommended for best performance)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic Claude (Excellent for complex documents)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google Gemini (Great for large documents)
export GOOGLE_API_KEY="your-google-api-key"

# Azure OpenAI (Enterprise option)
export AZURE_OPENAI_API_KEY="your-azure-key"
export AZURE_OPENAI_ENDPOINT="your-azure-endpoint"
```

### Step 2: Installation

**Option 1: Install from GitHub (Recommended)** ⭐
```bash
pip install git+https://github.com/octondata/od-parse.git
```

**Option 2: Install from Wheel File (From GitHub Releases)**
```bash
# Download wheel from: https://github.com/octondata/od-parse/releases
pip install od_parse-0.2.0-py3-none-any.whl
```

**Option 3: Install from Source (For Development)**

```bash
# Clone the repository
git clone https://github.com/octondata/od-parse.git
cd od-parse

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Or install with advanced features
pip install -e .[advanced]
pip install -e .[all]
```

> **💡 Note:** See [docs/INSTALLATION_METHODS.md](docs/INSTALLATION_METHODS.md) for all installation methods and best practices.

### Quick Setup Script

For convenience, you can use the automated setup script:

```bash
# Clone and run setup script
git clone https://github.com/octondata/od-parse.git
cd od-parse
chmod +x setup_dev.sh
./setup_dev.sh
```

This script will:
- ✅ Create a virtual environment
- ✅ Install all dependencies
- ✅ Install od-parse in development mode
- ✅ Test the installation
- ✅ Provide next steps

### Alternative: Direct Download

If you don't have git, you can download and install directly:

```bash
# Download the source code
wget https://github.com/octondata/od-parse/archive/main.zip
unzip main.zip
cd od-parse-main

# Create virtual environment and install
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

### Future PyPI Installation (Coming Soon)

Once published to PyPI, you'll be able to install with:

```bash
# Install the basic package (COMING SOON)
pip install od-parse

# Install with all advanced features (COMING SOON)
pip install "od-parse[all]"

# Install with specific advanced features (COMING SOON)
pip install "od-parse[trocr]"              # TrOCR text recognition
pip install "od-parse[table_transformer]"  # Neural table extraction
pip install "od-parse[llava_next]"         # Document understanding with VLMs
pip install "od-parse[multilingual]"       # Multi-language support
pip install "od-parse[quality_assessment]" # Quality assessment metrics
pip install "od-parse[async_processing]"   # Async processing capabilities

# Install preset combinations (COMING SOON)
pip install "od-parse[basic]"        # Essential features
pip install "od-parse[advanced]"     # All stable features
pip install "od-parse[experimental]" # All features including experimental
```

### Installation Troubleshooting

**❌ Error: "No LLM API keys found"**

This is the most common error. od-parse requires LLM access:

```bash
# Set at least one API key
export OPENAI_API_KEY="your-key-here"
# OR
export ANTHROPIC_API_KEY="your-key-here"
# OR
export GOOGLE_API_KEY="your-key-here"
```

**❌ Error: "Could not find a version that satisfies the requirement od-parse"**

This error occurs because `od-parse` is not yet published to PyPI. Use the development installation method above.

**❌ Error: "externally-managed-environment"**

This is a safety feature in newer Python versions. Always use a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Then install
pip install -e .
```

**❌ Error: "pip: command not found"**

Use `python3 -m pip` instead:

```bash
python3 -m pip install -e .
```

**❌ Missing Dependencies**

Install core dependencies manually if needed:

```bash
pip install pdfminer.six tabula-py opencv-python pillow pytesseract pandas numpy openai anthropic google-generativeai pdf2image
```

> **🤖 LLM-First Approach:** od-parse now requires LLM API keys for advanced document understanding. Local processing is available for basic extraction only.

## Quick Start

```python
import os
from od_parse import parse_pdf, convert_to_markdown

# Set your LLM API key (never commit actual keys!)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Parse a PDF with LLM-powered understanding
result = parse_pdf("path/to/document.pdf")

# Access LLM analysis results
llm_analysis = result['parsed_data']['llm_analysis']
print(f"Extracted Data: {llm_analysis['extracted_data']}")
print(f"Model Used: {llm_analysis['model_info']['model']}")
print(f"Processing Cost: ${llm_analysis['model_info']['cost_estimate']:.4f}")

# Access document classification
classification = result['parsed_data']['document_classification']
print(f"Document Type: {classification['document_type']}")
print(f"Confidence: {classification['confidence']:.2f}")

# Convert to Markdown
markdown = convert_to_markdown(result)

# Save structured output
with open("output.json", "w") as f:
    json.dump(llm_analysis['extracted_data'], f, indent=2)
```

## 🔒 Security & API Key Management

**NEVER commit API keys to version control!** od-parse provides secure configuration management:

### **Method 1: Environment File (Recommended)**

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` with your actual API keys:
```bash
# .env file (never commit this!)
OPENAI_API_KEY=your-actual-openai-key-here
GOOGLE_API_KEY=your-actual-google-key-here
ANTHROPIC_API_KEY=your-actual-anthropic-key-here
```

### **Method 2: Secure Configuration in Code**

```python
from od_parse.config.env_config import get_api_key, setup_secure_environment

# Auto-loads from .env file
setup_secure_environment()

# Get API keys securely (no hardcoding!)
openai_key = get_api_key('openai')
google_key = get_api_key('google')

# Use in your code without exposing keys
result = parse_pdf("document.pdf")  # Uses configured keys automatically
```

### **Method 3: Environment Variables**

```bash
export OPENAI_API_KEY="your-api-key-here"
export GOOGLE_API_KEY="your-api-key-here"
export ANTHROPIC_API_KEY="your-api-key-here"
```

### **Security Best Practices**

- ✅ Use `.env` files (included in `.gitignore`)
- ✅ Use environment variables in production
- ✅ Use the secure configuration helper
- ❌ Never hardcode API keys in source code
- ❌ Never commit `.env` files to git
- ❌ Never share API keys in documentation

## Advanced Features

od-parse includes cutting-edge AI features that can be enabled as needed:

### Smart Document Classification

Automatically identify document types with high accuracy using multi-signal analysis:

```python
from od_parse import parse_pdf
from od_parse.intelligence import DocumentType, DocumentClassifier

# Parse with smart classification enabled
result = parse_pdf("document.pdf", use_deep_learning=True)
classification = result['parsed_data']['document_classification']

# Get classification results
doc_type = classification['document_type']  # e.g., "tax_form_1040"
confidence = classification['confidence']   # e.g., 0.95
indicators = classification['key_indicators']  # e.g., {"ssn_found": "123-45-6789"}
suggestions = classification['suggestions']  # Processing recommendations

print(f"Document Type: {doc_type}")
print(f"Confidence: {confidence:.2f}")
print(f"Key Indicators: {indicators}")
```

**Supported Document Types (54+ types):**
- **Tax Documents**: Form 1040, W-2, 1099, Schedules A-D, etc.
- **Financial**: Bank statements, credit card statements, investment reports
- **Business**: Invoices, receipts, contracts, purchase orders, quotes
- **Legal**: Legal contracts, court documents, patents, wills
- **Medical**: Medical records, prescriptions, lab reports, insurance claims
- **Academic**: Research papers, transcripts, diplomas, certificates
- **Government**: Passports, driver's licenses, birth certificates
- **General**: Resumes, letters, reports, manuals, brochures

**Classification Features:**
- **Multi-Signal Analysis**: Text patterns, document structure, keywords, format detection, semantic analysis
- **High Accuracy**: 100% accuracy on tax forms, high confidence scoring
- **Key Extraction**: Automatic detection of SSNs, EINs, account numbers, etc.
- **Smart Suggestions**: Context-aware processing recommendations
- **Extensible**: Easy to add new document types and patterns

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

### Direct Document Classification

Use the classifier independently for custom workflows:

```python
from od_parse.intelligence import DocumentClassifier, DocumentType

# Initialize classifier
classifier = DocumentClassifier()

# Classify parsed document data
parsed_data = {"text": "Form 1040 U.S. Individual Income Tax Return...",
               "tables": [], "forms": []}
analysis = classifier.classify_document(parsed_data)

# Access detailed analysis
print(f"Document Type: {analysis.document_type.value}")
print(f"Confidence: {analysis.confidence:.2f}")
print(f"Detected Patterns: {analysis.detected_patterns}")
print(f"Key Indicators: {analysis.key_indicators}")
print(f"Suggestions: {analysis.suggestions}")

# Check for specific document types
if analysis.document_type == DocumentType.TAX_FORM_1040:
    print("This is a tax form - extract tax-specific fields")
elif analysis.document_type == DocumentType.INVOICE:
    print("This is an invoice - extract billing information")
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

# Use deep learning capabilities (includes smart document classification)
python -m od_parse.main document.pdf --deep-learning
```

## Enterprise Applications

This library is designed for enterprise AI applications:

- **Document Classification & Routing**: Automatically classify and route documents based on type (tax forms, invoices, contracts, etc.)
- **RAG Systems**: Extract and embed document content for retrieval-augmented generation
- **AI Agents**: Provide structured data and document intelligence for AI agents to work with
- **Document Automation**: Automate document processing workflows with intelligent classification
- **Data Unification**: Extract structured data from unstructured documents with type-aware processing
- **Knowledge Bases**: Build searchable knowledge bases from document repositories with smart categorization
- **Compliance & Audit**: Automatically detect sensitive documents (tax forms, medical records) for compliance workflows

## License

MIT License
