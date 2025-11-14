#!/usr/bin/env python3
"""
TrOCR Text Recognition Example

This example demonstrates how to use TrOCR for superior text recognition.
TrOCR uses transformer models for better accuracy than traditional OCR.
"""

from od_parse.ocr import TrOCREngine
from od_parse.config import get_advanced_config

def example_trocr_basic():
    """Basic TrOCR usage example - matches the user's example."""
    
    # Enable TrOCR feature (required before initialization)
    config = get_advanced_config()
    config.enable_feature('trocr', check_dependencies=False)
    
    # Initialize TrOCR engine (exactly as shown in user's example)
    engine = TrOCREngine(model_name="microsoft/trocr-base-printed")
    
    # Extract text from image
    result = engine.extract_text("document.png")
    
    # Print results (exactly as shown in user's example)
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']}")
    
    return result


def example_trocr_different_models():
    """Example using different TrOCR models."""
    
    config = get_advanced_config()
    config.enable_feature('trocr', check_dependencies=False)
    
    models = [
        "microsoft/trocr-base-printed",      # Default, good for printed text
        "microsoft/trocr-base-handwritten",  # Better for handwritten text
        "microsoft/trocr-large-printed",    # Larger model, better accuracy
        "microsoft/trocr-large-handwritten" # Large model for handwriting
    ]
    
    for model_name in models:
        print(f"\n=== Using {model_name} ===")
        try:
            engine = TrOCREngine(model_name=model_name)
            result = engine.extract_text("document.png")
            print(f"Text: {result['text'][:100]}...")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Engine: {result['engine']}")
        except Exception as e:
            print(f"Error: {e}")


def example_trocr_check_availability():
    """Check if TrOCR is available before using."""
    
    config = get_advanced_config()
    config.enable_feature('trocr', check_dependencies=False)
    
    engine = TrOCREngine()
    
    # Check if TrOCR is available
    if engine.is_available():
        print("✅ TrOCR is available and ready to use")
        info = engine.get_engine_info()
        print(f"Model: {info['model_name']}")
        print(f"Device: {info['device']}")
    else:
        print("⚠️ TrOCR not available, using fallback (Tesseract)")
        info = engine.get_engine_info()
        print(f"Fallback engine: {info['current_engine']}")
    
    # Extract text (will use TrOCR if available, otherwise fallback)
    result = engine.extract_text("document.png")
    print(f"\nExtracted text: {result['text']}")
    print(f"Engine used: {result['engine']}")


def example_trocr_batch_processing():
    """Process multiple images at once."""
    
    config = get_advanced_config()
    config.enable_feature('trocr', check_dependencies=False)
    
    engine = TrOCREngine()
    
    # Process multiple images
    images = ["image1.png", "image2.png", "image3.png"]
    results = engine.batch_extract_text(images)
    
    for i, result in enumerate(results):
        print(f"\nImage {i+1}:")
        print(f"  Text: {result['text'][:50]}...")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  Engine: {result['engine']}")


if __name__ == "__main__":
    print("TrOCR Text Recognition Examples")
    print("=" * 50)
    
    # Note: TrOCR requires optional dependencies
    # Install with: pip install torch transformers
    
    print("\n⚠️ Note: TrOCR requires optional dependencies:")
    print("  pip install torch transformers")
    print("\nIf not installed, TrOCR will fallback to Tesseract OCR")
    
    # Example 1: Basic usage
    print("\n=== Example 1: Basic Usage ===")
    try:
        example_trocr_basic()
    except Exception as e:
        print(f"Error: {e}")
        print("This is expected if TrOCR dependencies are not installed.")
        print("The engine will automatically fallback to Tesseract.")
    
    # Example 2: Check availability
    print("\n=== Example 2: Check Availability ===")
    example_trocr_check_availability()

