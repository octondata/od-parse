# OctonData Parse: Technical Reference

This technical reference guide provides detailed information about the advanced modules in the `od-parse` library, their capabilities, configuration options, and usage examples.

## Table of Contents

- [UnifiedPDFParser](#unifiedpdfparser)
- [Pipeline Architecture](#pipeline-architecture)
- [Document Intelligence](#document-intelligence)
- [Layout Analysis](#layout-analysis)
- [Table Extraction](#table-extraction)
- [Form Understanding](#form-understanding)
- [Semantic Structure Extraction](#semantic-structure-extraction)
- [Neural Table Extraction](#neural-table-extraction)
- [Transformer OCR](#transformer-ocr)
- [Integration Connectors](#integration-connectors)

## UnifiedPDFParser

The `UnifiedPDFParser` provides a centralized interface to all the advanced parsing capabilities.

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_deep_learning` | boolean | `True` | Whether to use deep learning models for enhanced extraction |
| `extract_handwritten` | boolean | `True` | Whether to extract handwritten content |
| `extract_tables` | boolean | `True` | Whether to extract tables |
| `extract_forms` | boolean | `True` | Whether to extract form elements |
| `extract_structure` | boolean | `True` | Whether to extract document structure |
| `output_format` | string | `"json"` | Format for output (json, markdown, text) |

### Example Usage

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser

# Initialize with configuration
parser = UnifiedPDFParser({
    "use_deep_learning": True,
    "extract_handwritten": True,
    "extract_tables": True,
    "extract_forms": True,
    "extract_structure": True
})

# Parse document
result = parser.parse("document.pdf")

# Convert to markdown
markdown = parser.to_markdown(result)
```

## Pipeline Architecture

The pipeline architecture allows for flexible configuration of processing stages.

### Available Pipeline Stages

| Stage | Description |
|-------|-------------|
| `LoadDocumentStage` | Loads a document from a file path |
| `BasicParsingStage` | Performs basic PDF parsing without deep learning |
| `AdvancedParsingStage` | Performs advanced PDF parsing with deep learning |
| `TableExtractionStage` | Extracts tables from a document |
| `FormExtractionStage` | Extracts form elements from a document |
| `HandwrittenContentStage` | Extracts handwritten content from a document |
| `DocumentStructureStage` | Extracts document structure |
| `OutputFormattingStage` | Formats the output of the pipeline |

### Pre-configured Pipelines

| Pipeline | Description |
|----------|-------------|
| `PDFPipeline.create_full_pipeline()` | Complete pipeline with all extraction capabilities |
| `PDFPipeline.create_fast_pipeline()` | Speed-optimized pipeline without deep learning |
| `PDFPipeline.create_tables_pipeline()` | Pipeline focused on table extraction |
| `PDFPipeline.create_forms_pipeline()` | Pipeline focused on form extraction |
| `PDFPipeline.create_structure_pipeline()` | Pipeline focused on document structure extraction |

### Example Usage

```python
from od_parse.advanced.pipeline import (
    PDFPipeline,
    LoadDocumentStage,
    BasicParsingStage,
    TableExtractionStage,
    OutputFormattingStage
)

# Create custom pipeline
pipeline = PDFPipeline()
pipeline.add_stage(LoadDocumentStage())
pipeline.add_stage(BasicParsingStage())
pipeline.add_stage(TableExtractionStage({"use_neural": False}))
pipeline.add_stage(OutputFormattingStage({"format": "json"}))

# Process document
result = pipeline.process("document.pdf")
```

## Document Intelligence

The `DocumentIntelligence` module provides advanced algorithms for understanding document structure and content relationships.

### Key Capabilities

- Semantic region detection and classification
- Entity recognition within documents
- Content relationship mapping
- Document type classification

### Example Usage

```python
from od_parse.advanced.document_intelligence import DocumentIntelligence

# Initialize
doc_intelligence = DocumentIntelligence()

# Process document
result = doc_intelligence.analyze_document("document.pdf")

# Get document type
doc_type = result.get("document_type")
print(f"Document type: {doc_type}")

# Get semantic regions
regions = result.get("regions", [])
for region in regions:
    print(f"Region: {region.content_type} ({region.confidence:.2f})")
```

## Layout Analysis

The `LayoutAnalyzer` analyzes the spatial arrangement of document elements, including columns, reading order, and layout structure.

### Key Capabilities

- Multi-column detection
- Reading order determination
- Section boundary identification
- Layout classification (e.g., single column, two-column, mixed)

### Example Usage

```python
from od_parse.advanced.layout_analysis import LayoutAnalyzer

# Initialize
layout_analyzer = LayoutAnalyzer()

# Analyze document layout
layout = layout_analyzer.analyze(elements)

# Get column information
columns = layout.get("columns", [])
print(f"Detected {len(columns)} columns")

# Get reading order
reading_order = layout.get("reading_order", [])
```

## Table Extraction

The `AdvancedTableExtractor` implements techniques for detecting and extracting tables from PDFs, including those without explicit borders.

### Key Capabilities

- Rule-based table detection
- Cell extraction and merging
- Header detection
- Multi-page table handling
- Table structure reconstruction

### Example Usage

```python
from od_parse.advanced.table_extraction import AdvancedTableExtractor

# Initialize
table_extractor = AdvancedTableExtractor()

# Extract tables
tables = table_extractor.extract_tables(pdf_data)

# Process each table
for table in tables:
    print(f"Table with {table.rows} rows and {table.cols} columns")
    
    # Convert to various formats
    markdown = table_extractor.table_to_markdown(table)
    html = table_extractor.table_to_html(table)
    csv = table_extractor.table_to_csv(table)
```

## Form Understanding

The `FormUnderstanding` module detects and extracts various form elements from PDFs.

### Supported Form Elements

- Checkboxes
- Radio buttons
- Text fields
- Dropdowns
- Signatures

### Example Usage

```python
from od_parse.advanced.form_understanding import FormUnderstanding

# Initialize
form_analyzer = FormUnderstanding()

# Extract forms
forms = form_analyzer.extract_forms(pdf_data)

# Process form fields
fields = forms.get("fields", [])
for field in fields:
    field_type = field.get("type")
    label = field.get("label")
    value = field.get("value")
    
    print(f"Field: {label} ({field_type}) = {value}")
```

## Semantic Structure Extraction

The `SemanticStructureExtractor` extracts the hierarchical structure and semantic relationships within documents.

### Key Capabilities

- Section hierarchy detection
- Reference identification
- Table of contents extraction
- Cross-reference mapping

### Example Usage

```python
from od_parse.advanced.semantic_structure import SemanticStructureExtractor

# Initialize
structure_extractor = SemanticStructureExtractor()

# Extract structure
structure = structure_extractor.extract_structure(pdf_data)

# Process document structure
elements = structure.get("elements", [])
for element in elements:
    element_type = element.get("type")
    text = element.get("text")
    
    if element_type == "heading":
        level = element.get("level", 1)
        print(f"{'#' * level} {text}")
    elif element_type == "paragraph":
        print(text)
    elif element_type == "list_item":
        print(f"- {text}")
```

## Neural Table Extraction

The `NeuralTableExtractor` uses deep learning to detect and extract tables from PDFs.

### Key Capabilities

- Deep learning-based table detection
- Complex table structure recognition
- Borderless table extraction
- Cell content classification

### Example Usage

```python
from od_parse.advanced.neural_table_extraction import NeuralTableExtractor

# Initialize
neural_extractor = NeuralTableExtractor()

# Extract tables
tables = neural_extractor.extract_tables(pdf_data)

# Process tables
for table in tables:
    print(f"Table detected with confidence {table.confidence:.2f}")
    
    for cell in table.cells:
        print(f"Cell at ({cell.row}, {cell.col}): {cell.text}")
```

## Transformer OCR

The `TransformerOCR` module uses transformer-based models for extracting text from images and handwritten content.

### Key Capabilities

- Handwritten text recognition
- Text extraction from images
- Language detection
- Multi-language support

### Example Usage

```python
from od_parse.advanced.transformer_ocr import TransformerOCR, HandwrittenTextRecognizer

# Initialize for general OCR
ocr = TransformerOCR()

# Process an image
text = ocr.process_image("image.png")
print(f"Extracted text: {text}")

# Initialize for handwritten content
handwritten_recognizer = HandwrittenTextRecognizer()

# Process document with handwritten content
result = handwritten_recognizer.process_document("document.pdf")
handwritten_items = result.get("content", [])
```

## Integration Connectors

The `od-parse` library includes connectors for integrating with various external systems.

### Available Connectors

| Connector | Description |
|-----------|-------------|
| `JSONFileConnector` | Export to JSON files |
| `CSVConnector` | Export to CSV files |
| `DatabaseConnector` | Export to relational databases (SQLite, PostgreSQL) |
| `APIConnector` | Export to REST APIs |
| `VectorDBConnector` | Export to vector databases (pgvector) |

### Example Usage

```python
from od_parse.advanced.integrations import JSONFileConnector, VectorDBConnector

# Process document
result = pipeline.process("document.pdf")

# Export to JSON
json_connector = JSONFileConnector({"file_path": "output.json"})
json_connector.export(result)

# Export to pgvector
vector_connector = VectorDBConnector({
    "db_type": "pgvector",
    "conn_string": "postgresql://username:password@hostname:5432/vectordb"
})
vector_connector.export(result)
```

## Command Line Interface

The library includes a command-line interface for quick document processing.

### Usage Examples

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

This technical reference guide provides comprehensive documentation of the advanced PDF parsing capabilities in the `od-parse` library. For additional information or specific use cases, please refer to the examples in the `examples/` directory or reach out to the OctonData team.
