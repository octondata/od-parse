#!/usr/bin/env python3
"""
Simple TrOCR Example - Matches the documentation example exactly.

This is the simplest possible TrOCR usage example.
"""

from od_parse.ocr import TrOCREngine
from od_parse.config import get_advanced_config

# Enable TrOCR feature (required before use)
config = get_advanced_config()
config.enable_feature("trocr", check_dependencies=False)

# Initialize TrOCR engine
engine = TrOCREngine(model_name="microsoft/trocr-base-printed")

# Extract text from image
result = engine.extract_text("document.png")

# Print results
print(f"Text: {result['text']}")
print(f"Confidence: {result['confidence']}")
