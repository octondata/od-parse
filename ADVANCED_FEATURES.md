# Advanced Features Guide

This guide covers the advanced features available in od-parse, including configuration, installation, and usage examples.

## Table of Contents

1. [Feature Configuration](#feature-configuration)
2. [TrOCR Integration](#trocr-integration)
3. [Table Transformer](#table-transformer)
4. [Document Quality Assessment](#document-quality-assessment)
5. [Async Processing](#async-processing)
6. [LLaVA-NeXT Document Understanding](#llava-next-document-understanding)
7. [Multi-Language Support](#multi-language-support)
8. [Installation Options](#installation-options)

## Feature Configuration

All advanced features are disabled by default to keep the library lightweight. You can enable them as needed:

```python
from od_parse.config import get_advanced_config

# Get configuration instance
config = get_advanced_config()

# Enable individual features
config.enable_feature('trocr')
config.enable_feature('table_transformer')
config.enable_feature('quality_assessment')
config.enable_feature('async_processing')
config.enable_feature('llava_next')
config.enable_feature('multilingual')

# Or use presets
config.enable_preset('basic')      # Essential features only
config.enable_preset('advanced')   # All stable features
config.enable_preset('experimental')  # All features including experimental

# Check feature status
print(config.list_all_features())
```

## TrOCR Integration

TrOCR provides superior text recognition using transformer models.

### Installation
```bash
pip install od-parse[trocr]
```

### Usage
```python
from od_parse.config import get_advanced_config
from od_parse.ocr import TrOCREngine

# Enable TrOCR
config = get_advanced_config()
config.enable_feature('trocr')

# Initialize engine
engine = TrOCREngine(model_name="microsoft/trocr-base-printed")

# Extract text from image
result = engine.extract_text("document.png")
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
print(f"Engine: {result['engine']}")

# Batch processing
images = ["doc1.png", "doc2.png", "doc3.png"]
results = engine.batch_extract_text(images)

# Available models:
# - microsoft/trocr-base-printed (default)
# - microsoft/trocr-base-handwritten
# - microsoft/trocr-large-printed
# - microsoft/trocr-large-handwritten
```

## Table Transformer

Advanced table detection and extraction using neural networks.

### Installation
```bash
pip install od-parse[table_transformer]
```

### Usage
```python
from od_parse.config import get_advanced_config
from od_parse.advanced import TableTransformerEngine

# Enable Table Transformer
config = get_advanced_config()
config.enable_feature('table_transformer')

# Initialize engine
engine = TableTransformerEngine()

# Extract tables from image
result = engine.extract_tables("document_with_tables.png")

for i, table in enumerate(result['tables']):
    print(f"Table {i}:")
    print(f"  Confidence: {table['confidence']}")
    print(f"  Bounding box: {table['bbox']}")
    print(f"  Structure: {table['structure']}")

# Visualize detections
image_with_boxes = engine.visualize_detections("document.png", result['tables'])
image_with_boxes.save("detected_tables.png")
```

## Document Quality Assessment

Assess the quality and reliability of extraction results.

### Installation
```bash
pip install od-parse[quality_assessment]
```

### Usage
```python
from od_parse.config import get_advanced_config
from od_parse.quality import assess_document_quality

# Enable quality assessment
config = get_advanced_config()
config.enable_feature('quality_assessment')

# Assess extraction quality
extraction_result = {
    "text": {"content": "Sample document text", "confidence": 0.9},
    "tables": [{"data": [["A", "B"], ["1", "2"]], "confidence": 0.85}],
    "forms": [{"fields": {"name": "John"}, "confidence": 0.8}]
}

quality = assess_document_quality(extraction_result)

print(f"Overall Score: {quality['overall_score']:.2f}")
print(f"Text Quality: {quality['text_quality']['score']:.2f}")
print(f"Structure Quality: {quality['structure_quality']['score']:.2f}")
print("Recommendations:")
for rec in quality['recommendations']:
    print(f"  - {rec}")
```

## Async Processing

Process large files and batches asynchronously with progress tracking.

### Installation
```bash
pip install od-parse[async_processing]
```

### Usage
```python
import asyncio
from od_parse.config import get_advanced_config
from od_parse.async_processing import process_files_async

# Enable async processing
config = get_advanced_config()
config.enable_feature('async_processing')

# Define processing function
def process_document(file_path):
    # Your document processing logic here
    return {"file": str(file_path), "processed": True}

# Progress callback
def progress_callback(progress_data):
    print(f"Progress: {progress_data['progress_percentage']:.1f}% "
          f"({progress_data['completed_items']}/{progress_data['total_items']})")

# Process files asynchronously
async def main():
    files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
    
    results = await process_files_async(
        files, 
        process_document,
        progress_callback=progress_callback,
        max_workers=4
    )
    
    for result in results:
        print(f"File: {result['file_path']}, Status: {result['status']}")

# Run async processing
asyncio.run(main())
```

## LLaVA-NeXT Document Understanding

Advanced document understanding using vision-language models.

### Installation
```bash
pip install od-parse[llava_next]
```

### Usage
```python
from od_parse.config import get_advanced_config
from od_parse.advanced import LLaVANextEngine

# Enable LLaVA-NeXT
config = get_advanced_config()
config.enable_feature('llava_next')

# Initialize engine
engine = LLaVANextEngine(
    model_name="llava-hf/llava-v1.6-mistral-7b-hf",
    load_in_4bit=True  # For memory efficiency
)

# Understand document
result = engine.understand_document(
    "complex_document.png",
    prompt="Analyze this document and describe its structure, content, and purpose."
)

print(f"Understanding: {result['understanding']}")
print(f"Confidence: {result['confidence']}")

# Extract specific information
table_info = engine.extract_structured_information(
    "document.png", 
    information_type="tables"
)

form_info = engine.extract_structured_information(
    "document.png", 
    information_type="forms"
)

# Batch processing
images = ["doc1.png", "doc2.png"]
results = engine.batch_understand_documents(images)
```

## Multi-Language Support

Comprehensive multilingual document processing.

### Installation
```bash
pip install od-parse[multilingual]
```

### Usage
```python
from od_parse.config import get_advanced_config
from od_parse.multilingual import MultilingualProcessor

# Enable multilingual support
config = get_advanced_config()
config.enable_feature('multilingual')

# Initialize processor
processor = MultilingualProcessor()

# Detect language
text = "Este es un documento en español."
detection = processor.detect_language(text)
print(f"Detected language: {detection['language']} (confidence: {detection['confidence']:.2f})")

# Process multilingual text
result = processor.process_multilingual_text(
    text,
    target_language="en",
    include_translation=True
)

print(f"Original: {result['original_text']}")
print(f"Language: {result['detected_language']['language']}")
if result['translation']:
    print(f"Translation: {result['translation']['translated_text']}")

# Get supported languages
supported = processor.get_supported_languages()
print(f"Detection: {supported['detection']}")
print(f"Processing: {supported['processing']}")
print(f"Translation: {supported['translation']}")
```

## Installation Options

### Basic Installation
```bash
pip install od-parse
```

### Feature-Specific Installation
```bash
# Individual features
pip install od-parse[trocr]
pip install od-parse[table_transformer]
pip install od-parse[quality_assessment]
pip install od-parse[async_processing]
pip install od-parse[llava_next]
pip install od-parse[multilingual]

# Preset combinations
pip install od-parse[basic]        # Essential features
pip install od-parse[advanced]     # All stable features
pip install od-parse[experimental] # All features
pip install od-parse[all]          # All features (same as experimental)
```

### Development Installation
```bash
pip install od-parse[dev]  # Development tools
```

## Configuration Management

### Save and Load Configuration
```python
from od_parse.config import get_advanced_config

config = get_advanced_config()

# Enable features
config.enable_feature('trocr')
config.enable_feature('quality_assessment')

# Save configuration
config.save_config('my_config.json')

# Load configuration
config.load_config('my_config.json')
```

### Environment Variables
```bash
# Set config file location
export OD_PARSE_CONFIG=/path/to/config.json
```

### Programmatic Configuration
```python
from od_parse.config import configure_features

# Configure multiple features at once
configure_features(
    trocr=True,
    quality_assessment=True,
    async_processing=True,
    multilingual=False
)
```

## Performance Considerations

1. **Memory Usage**: Advanced features like LLaVA-NeXT require significant GPU memory
2. **Model Loading**: First-time use downloads models (can be large)
3. **GPU Acceleration**: Most advanced features benefit from GPU acceleration
4. **Quantization**: Use 4-bit quantization for memory efficiency

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use smaller models or enable quantization
2. **Model Download Fails**: Check internet connection and disk space
3. **Feature Not Available**: Install required dependencies
4. **Import Errors**: Ensure all optional dependencies are installed

### Getting Help

```python
from od_parse.config import get_advanced_config

config = get_advanced_config()

# Check feature availability
print(config.get_available_features())

# Get feature information
info = config.get_feature_info('trocr')
print(f"Dependencies: {info['dependencies']}")
print(f"Description: {info['description']}")
```
