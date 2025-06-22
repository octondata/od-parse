#!/usr/bin/env python3
"""
Advanced Features Demo

This script demonstrates the advanced features of od-parse including:
- Feature configuration
- TrOCR text recognition
- Table Transformer table extraction
- Document quality assessment
- Async processing
- LLaVA-NeXT document understanding
- Multi-language support

Usage:
    python advanced_features_demo.py
"""

import asyncio
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import od-parse components
from od_parse.config import get_advanced_config, configure_features


def create_sample_document():
    """Create a sample document image for testing."""
    # Create a simple document image
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a default font
        font = ImageFont.load_default()
    except:
        font = None
    
    # Add title
    draw.text((50, 50), "Sample Document", fill='black', font=font)
    
    # Add some text content
    text_lines = [
        "This is a sample document created for testing advanced features.",
        "It contains various elements including text, tables, and forms.",
        "The document demonstrates multilingual capabilities:",
        "English: Hello, how are you?",
        "Spanish: Hola, ¿cómo estás?",
        "French: Bonjour, comment allez-vous?",
        "German: Hallo, wie geht es dir?"
    ]
    
    y_pos = 100
    for line in text_lines:
        draw.text((50, y_pos), line, fill='black', font=font)
        y_pos += 30
    
    # Add a simple table
    draw.text((50, y_pos + 20), "Sample Table:", fill='black', font=font)
    
    # Table headers
    table_y = y_pos + 50
    draw.rectangle([50, table_y, 400, table_y + 25], outline='black')
    draw.text((60, table_y + 5), "Name", fill='black', font=font)
    draw.text((150, table_y + 5), "Age", fill='black', font=font)
    draw.text((250, table_y + 5), "City", fill='black', font=font)
    
    # Table rows
    rows = [
        ["John Doe", "25", "New York"],
        ["Jane Smith", "30", "London"],
        ["Bob Johnson", "35", "Paris"]
    ]
    
    for i, row in enumerate(rows):
        row_y = table_y + 25 + (i * 25)
        draw.rectangle([50, row_y, 400, row_y + 25], outline='black')
        draw.text((60, row_y + 5), row[0], fill='black', font=font)
        draw.text((150, row_y + 5), row[1], fill='black', font=font)
        draw.text((250, row_y + 5), row[2], fill='black', font=font)
    
    return image


def demo_configuration():
    """Demonstrate feature configuration."""
    print("=== Feature Configuration Demo ===")
    
    config = get_advanced_config()
    
    # Show all available features
    print("\nAvailable features:")
    features = config.list_all_features()
    for name, info in features.items():
        status = "✓" if info['available'] else "✗"
        enabled = "ON" if info['enabled'] else "OFF"
        print(f"  {status} {name}: {enabled} - {info['description']}")
    
    # Enable basic features that don't require heavy dependencies
    print("\nEnabling basic features...")
    config.enable_feature('quality_assessment', check_dependencies=False)
    config.enable_feature('async_processing', check_dependencies=False)
    
    print("Configuration updated!")


def demo_quality_assessment():
    """Demonstrate document quality assessment."""
    print("\n=== Document Quality Assessment Demo ===")
    
    try:
        from od_parse.quality import assess_document_quality
        
        # Sample extraction result
        extraction_result = {
            "text": {
                "content": "This is a well-formatted document with clear text and proper structure.",
                "confidence": 0.9
            },
            "tables": [
                {
                    "data": [["Name", "Age"], ["John", "25"], ["Jane", "30"]],
                    "confidence": 0.85,
                    "shape": (3, 2)
                }
            ],
            "forms": [
                {
                    "fields": {"name": "John Doe", "email": "john@example.com"},
                    "confidence": 0.8
                }
            ],
            "metadata": {"pages": 1, "format": "pdf"}
        }
        
        # Assess quality
        quality = assess_document_quality(extraction_result)
        
        print(f"Overall Quality Score: {quality['overall_score']:.2f}")
        print(f"Text Quality: {quality['text_quality']['score']:.2f}")
        print(f"Structure Quality: {quality['structure_quality']['score']:.2f}")
        print(f"Confidence: {quality['confidence_metrics']['average_confidence']:.2f}")
        
        print("\nRecommendations:")
        for rec in quality['recommendations']:
            print(f"  • {rec}")
            
    except ImportError:
        print("Quality assessment dependencies not available.")
        print("Install with: pip install od-parse[quality_assessment]")


async def demo_async_processing():
    """Demonstrate async processing."""
    print("\n=== Async Processing Demo ===")
    
    try:
        from od_parse.async_processing import AsyncDocumentProcessor
        
        # Create sample files (simulate)
        sample_files = ["doc1.txt", "doc2.txt", "doc3.txt"]
        
        def mock_processor(file_path):
            """Mock document processor."""
            import time
            time.sleep(0.1)  # Simulate processing time
            return {"file": str(file_path), "words": len(str(file_path)) * 10}
        
        def progress_callback(progress_data):
            """Progress callback function."""
            print(f"  Progress: {progress_data['progress_percentage']:.1f}% "
                  f"({progress_data['completed_items']}/{progress_data['total_items']})")
        
        # Initialize async processor
        processor = AsyncDocumentProcessor(max_workers=2)
        
        print("Processing files asynchronously...")
        results = await processor.process_files_async(
            sample_files,
            mock_processor,
            progress_callback=progress_callback
        )
        
        print("\nResults:")
        for result in results:
            if result['status'] == 'success':
                print(f"  ✓ {result['file_path']}: {result['result']}")
            else:
                print(f"  ✗ {result['file_path']}: {result.get('error', 'Unknown error')}")
                
    except ImportError:
        print("Async processing dependencies not available.")
        print("Install with: pip install od-parse[async_processing]")


def demo_multilingual():
    """Demonstrate multilingual processing."""
    print("\n=== Multilingual Processing Demo ===")
    
    try:
        from od_parse.multilingual import MultilingualProcessor
        
        processor = MultilingualProcessor()
        
        # Test texts in different languages
        test_texts = {
            "English": "This is a sample document in English.",
            "Spanish": "Este es un documento de muestra en español.",
            "French": "Ceci est un document d'exemple en français.",
            "German": "Dies ist ein Beispieldokument auf Deutsch.",
            "Russian": "Это образец документа на русском языке.",
            "Chinese": "这是一个中文示例文档。"
        }
        
        print("Language Detection Results:")
        for lang_name, text in test_texts.items():
            result = processor.detect_language(text)
            print(f"  {lang_name}: detected as '{result['language']}' "
                  f"(confidence: {result['confidence']:.2f})")
        
        # Process multilingual text
        print("\nMultilingual Text Processing:")
        sample_text = test_texts["Spanish"]
        result = processor.process_multilingual_text(sample_text)
        
        print(f"  Original: {result['original_text']}")
        print(f"  Detected Language: {result['detected_language']['language']}")
        print(f"  Processing Method: {result['processing']['method']}")
        
        # Show supported languages
        supported = processor.get_supported_languages()
        print(f"\nSupported Languages:")
        print(f"  Detection: {len(supported['detection'])} languages")
        print(f"  Processing: {len(supported['processing'])} languages")
        print(f"  Translation: {len(supported['translation'])} languages")
        
    except ImportError:
        print("Multilingual dependencies not available.")
        print("Install with: pip install od-parse[multilingual]")


def demo_trocr():
    """Demonstrate TrOCR text recognition."""
    print("\n=== TrOCR Text Recognition Demo ===")
    
    try:
        from od_parse.ocr import TrOCREngine
        
        # Create sample image
        sample_image = create_sample_document()
        
        # Initialize TrOCR engine
        engine = TrOCREngine()
        
        if engine.is_available():
            print("TrOCR is available and initialized!")
            
            # Extract text
            result = engine.extract_text(sample_image)
            print(f"Extracted Text: {result['text'][:100]}...")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Engine: {result['engine']}")
        else:
            print("TrOCR not available - using fallback OCR")
            result = engine.extract_text(sample_image)
            print(f"Fallback Result: {result['engine']}")
            
    except ImportError:
        print("TrOCR dependencies not available.")
        print("Install with: pip install od-parse[trocr]")


def demo_table_transformer():
    """Demonstrate Table Transformer."""
    print("\n=== Table Transformer Demo ===")
    
    try:
        from od_parse.advanced import TableTransformerEngine
        
        # Create sample image with table
        sample_image = create_sample_document()
        
        # Initialize Table Transformer
        engine = TableTransformerEngine()
        
        if engine.is_available():
            print("Table Transformer is available!")
            
            # Extract tables
            result = engine.extract_tables(sample_image)
            print(f"Found {len(result['tables'])} tables")
            print(f"Engine: {result['engine']}")
        else:
            print("Table Transformer not available - using fallback")
            result = engine.extract_tables(sample_image)
            print(f"Fallback Result: {result['engine']}")
            
    except ImportError:
        print("Table Transformer dependencies not available.")
        print("Install with: pip install od-parse[table_transformer]")


def demo_llava_next():
    """Demonstrate LLaVA-NeXT document understanding."""
    print("\n=== LLaVA-NeXT Document Understanding Demo ===")
    
    try:
        from od_parse.advanced import LLaVANextEngine
        
        # Create sample image
        sample_image = create_sample_document()
        
        # Initialize LLaVA-NeXT
        engine = LLaVANextEngine()
        
        if engine.is_available():
            print("LLaVA-NeXT is available!")
            
            # Understand document
            result = engine.understand_document(
                sample_image,
                "Describe this document and its contents."
            )
            print(f"Understanding: {result['understanding'][:200]}...")
            print(f"Confidence: {result['confidence']:.2f}")
        else:
            print("LLaVA-NeXT not available - using fallback")
            result = engine.understand_document(sample_image)
            print(f"Fallback understanding available")
            
    except ImportError:
        print("LLaVA-NeXT dependencies not available.")
        print("Install with: pip install od-parse[llava_next]")


async def main():
    """Main demo function."""
    print("🚀 od-parse Advanced Features Demo")
    print("=" * 50)
    
    # Demo configuration
    demo_configuration()
    
    # Demo individual features
    demo_quality_assessment()
    await demo_async_processing()
    demo_multilingual()
    demo_trocr()
    demo_table_transformer()
    demo_llava_next()
    
    print("\n" + "=" * 50)
    print("Demo completed! 🎉")
    print("\nTo enable more features, install the required dependencies:")
    print("  pip install od-parse[advanced]  # All stable features")
    print("  pip install od-parse[all]       # All features")


if __name__ == "__main__":
    # Create examples directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Run the demo
    asyncio.run(main())
