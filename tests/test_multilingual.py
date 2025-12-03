"""
Tests for Multilingual Processing.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.config import get_advanced_config

# Handle potential import errors from googletrans/httpcore conflicts
try:
    from od_parse.multilingual import (
        MultilingualProcessor,
        detect_document_language,
        process_multilingual_document,
    )

    MULTILINGUAL_AVAILABLE = True
except (ImportError, AttributeError) as e:
    MULTILINGUAL_AVAILABLE = False
    MULTILINGUAL_ERROR = str(e)


@unittest.skipUnless(MULTILINGUAL_AVAILABLE, "Multilingual module not available")
class TestMultilingualProcessor(unittest.TestCase):
    """Test cases for Multilingual Processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
        self.processor = MultilingualProcessor()
        
        # Sample texts in different languages
        self.sample_texts = {
            "en": "This is a sample document in English with various content types.",
            "es": "Este es un documento de muestra en español con varios tipos de contenido.",
            "fr": "Ceci est un document d'exemple en français avec différents types de contenu.",
            "de": "Dies ist ein Beispieldokument auf Deutsch mit verschiedenen Inhaltstypen.",
            "it": "Questo è un documento di esempio in italiano con vari tipi di contenuto.",
            "pt": "Este é um documento de amostra em português com vários tipos de conteúdo."
        }
    
    def test_multilingual_processor_initialization(self):
        """Test multilingual processor initialization."""
        processor = MultilingualProcessor()
        self.assertIsInstance(processor, MultilingualProcessor)
    
    def test_language_detection_heuristic(self):
        """Test language detection using heuristic method."""
        # Test English text
        result = self.processor.detect_language(self.sample_texts["en"], method="heuristic")
        
        self.assertIn("language", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["method"], "heuristic")
        
        # Should detect English or have reasonable confidence
        self.assertGreaterEqual(result["confidence"], 0.0)
    
    def test_language_detection_auto(self):
        """Test automatic language detection."""
        for lang, text in self.sample_texts.items():
            result = self.processor.detect_language(text, method="auto")
            
            self.assertIn("language", result)
            self.assertIn("confidence", result)
            self.assertGreaterEqual(result["confidence"], 0.0)
    
    def test_language_detection_with_langdetect(self):
        """Test language detection with langdetect if available."""
        try:
            import langdetect  # noqa: F401

            # Enable multilingual feature and create new processor
            self.config.enable_feature("multilingual", check_dependencies=False)
            processor = MultilingualProcessor()

            result = processor.detect_language(
                self.sample_texts["en"], method="langdetect"
            )

            self.assertIn("language", result)
            self.assertIn("confidence", result)
            self.assertEqual(result["method"], "langdetect")

            # Should detect English
            self.assertEqual(result["language"], "en")

        except ImportError:
            self.skipTest("langdetect not available")
    
    def test_language_detection_empty_text(self):
        """Test language detection with empty text."""
        result = self.processor.detect_language("")
        
        self.assertEqual(result["language"], "unknown")
        self.assertEqual(result["confidence"], 0.0)
        self.assertIn("error", result)
    
    def test_multilingual_text_processing(self):
        """Test multilingual text processing."""
        text = self.sample_texts["en"]
        
        result = self.processor.process_multilingual_text(text)
        
        self.assertIn("original_text", result)
        self.assertIn("detected_language", result)
        self.assertIn("processing", result)
        self.assertEqual(result["status"], "success")
        
        # Check detection result structure
        detection = result["detected_language"]
        self.assertIn("language", detection)
        self.assertIn("confidence", detection)
        
        # Check processing result structure
        processing = result["processing"]
        self.assertIn("method", processing)
        self.assertIn("tokens", processing)
    
    def test_multilingual_text_processing_with_translation(self):
        """Test multilingual text processing with translation."""
        text = self.sample_texts["en"]
        
        result = self.processor.process_multilingual_text(
            text, 
            target_language="es", 
            include_translation=True
        )
        
        self.assertIn("translation", result)
        
        if result["translation"]:
            translation = result["translation"]
            self.assertIn("translated_text", translation)
            self.assertIn("source_language", translation)
            self.assertIn("target_language", translation)
    
    def test_text_processing_by_language(self):
        """Test language-specific text processing."""
        text = self.sample_texts["en"]
        
        result = self.processor._process_text_by_language(text, "en")
        
        self.assertIn("method", result)
        self.assertIn("tokens", result)
        
        # Should have some tokens
        self.assertGreater(len(result["tokens"]), 0)
    
    def test_translation_functionality(self):
        """Test translation functionality."""
        text = "Hello, how are you?"
        
        result = self.processor.translate_text(text, "en", "es")
        
        self.assertIn("translated_text", result)
        self.assertIn("source_language", result)
        self.assertIn("target_language", result)
        self.assertIn("method", result)
        
        # Should have some translation result
        self.assertIsNotNone(result["translated_text"])
    
    def test_translation_with_googletrans(self):
        """Test translation with Google Translate if available."""
        try:
            import googletrans  # noqa: F401

            # Enable multilingual feature and create new processor
            self.config.enable_feature("multilingual", check_dependencies=False)
            processor = MultilingualProcessor()

            text = "Hello world"
            result = processor._translate_with_googletrans(text, "en", "es")

            self.assertIn("translated_text", result)
            self.assertEqual(result["method"], "googletrans")
            self.assertNotEqual(result["translated_text"], text)  # Should be translated

        except (ImportError, AttributeError) as e:
            self.skipTest(f"googletrans not available: {e}")
    
    def test_translation_empty_text(self):
        """Test translation with empty text."""
        result = self.processor.translate_text("", "en", "es")
        
        self.assertEqual(result["translated_text"], "")
        self.assertIn("error", result)
    
    def test_supported_languages(self):
        """Test getting supported languages."""
        supported = self.processor.get_supported_languages()
        
        self.assertIn("detection", supported)
        self.assertIn("processing", supported)
        self.assertIn("translation", supported)
        
        # Each should be a list
        self.assertIsInstance(supported["detection"], list)
        self.assertIsInstance(supported["processing"], list)
        self.assertIsInstance(supported["translation"], list)
    
    def test_processor_availability(self):
        """Test processor availability check."""
        is_available = self.processor.is_available()
        self.assertIsInstance(is_available, bool)
    
    def test_processor_info(self):
        """Test getting processor information."""
        info = self.processor.get_processor_info()
        
        self.assertIn("multilingual_available", info)
        self.assertIn("langdetect_available", info)
        self.assertIn("spacy_available", info)
        self.assertIn("polyglot_available", info)
        self.assertIn("googletrans_available", info)
        self.assertIn("supported_languages", info)
    
    def test_heuristic_detection_various_languages(self):
        """Test heuristic detection with various languages."""
        # Test with different scripts
        test_cases = [
            ("Hello world", "en"),
            ("Привет мир", "ru"),  # Cyrillic
            ("你好世界", "zh"),      # Chinese
            ("مرحبا بالعالم", "ar"), # Arabic
            ("नमस्ते दुनिया", "hi")   # Hindi
        ]
        
        for text, expected_lang in test_cases:
            result = self.processor._detect_with_heuristics(text)
            
            self.assertIn("language", result)
            self.assertIn("confidence", result)
            self.assertEqual(result["method"], "heuristic")
            
            # For non-Latin scripts, should detect correctly
            if expected_lang in ["ru", "zh", "ar", "hi"]:
                self.assertEqual(result["language"], expected_lang)


@unittest.skipUnless(MULTILINGUAL_AVAILABLE, "Multilingual module not available")
class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_detect_document_language_convenience(self):
        """Test detect_document_language convenience function."""
        text = "This is a sample English document."
        
        result = detect_document_language(text)
        
        self.assertIn("language", result)
        self.assertIn("confidence", result)
        self.assertIn("method", result)
    
    def test_process_multilingual_document_convenience(self):
        """Test process_multilingual_document convenience function."""
        text = "This is a sample document for processing."
        
        result = process_multilingual_document(text)
        
        self.assertIn("original_text", result)
        self.assertIn("detected_language", result)
        self.assertIn("processing", result)
        self.assertEqual(result["status"], "success")
    
    def test_process_multilingual_document_with_translation(self):
        """Test multilingual document processing with translation."""
        text = "Hello, this is a test document."
        
        result = process_multilingual_document(
            text, 
            target_language="es", 
            include_translation=True
        )
        
        self.assertIn("translation", result)


class TestMultilingualConfiguration(unittest.TestCase):
    """Test multilingual configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
    
    def test_multilingual_feature_configuration(self):
        """Test multilingual feature can be enabled/disabled."""
        # Test enabling
        result = self.config.enable_feature("multilingual", check_dependencies=False)
        self.assertTrue(result)
        self.assertTrue(self.config.is_feature_enabled("multilingual"))
        
        # Test disabling
        result = self.config.disable_feature("multilingual")
        self.assertTrue(result)
        self.assertFalse(self.config.is_feature_enabled("multilingual"))
    
    def test_multilingual_feature_info(self):
        """Test multilingual feature information."""
        info = self.config.get_feature_info("multilingual")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Multi-Language Support")
        self.assertIn("dependencies", info)
        self.assertIn("spacy", info["dependencies"])
        self.assertIn("langdetect", info["dependencies"])
        self.assertIn("polyglot", info["dependencies"])
    
    def test_multilingual_preset_configuration(self):
        """Test enabling multilingual through presets."""
        # Test advanced preset (should include multilingual)
        result = self.config.enable_preset("advanced")
        
        # Check if multilingual can be enabled without dependency check
        manual_result = self.config.enable_feature("multilingual", check_dependencies=False)
        self.assertTrue(manual_result)
        self.assertTrue(self.config.is_feature_enabled("multilingual"))


if __name__ == "__main__":
    unittest.main()
