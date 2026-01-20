"""
Tests for TrOCR integration.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.config import get_advanced_config


class TestTrOCREngine(unittest.TestCase):
    """Test cases for TrOCR engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 50), color='white')
        
    def test_trocr_engine_import(self):
        """Test that TrOCR engine can be imported when dependencies are available."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            self.assertTrue(True, "TrOCR engine imported successfully")
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_trocr_engine_initialization_disabled(self):
        """Test TrOCR engine initialization when feature is disabled."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            
            # Ensure feature is disabled
            self.config.disable_feature("trocr")
            
            engine = TrOCREngine()
            self.assertFalse(engine.is_available())
            
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_trocr_engine_fallback(self):
        """Test that TrOCR engine falls back to Tesseract when TrOCR is not available."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            
            # Disable TrOCR feature to force fallback
            self.config.disable_feature("trocr")
            
            engine = TrOCREngine()
            result = engine.extract_text(self.test_image)
            
            # Should use fallback engine (tesseract, vlm, none, or error)
            self.assertIn("engine", result)
            self.assertIn(result["engine"], ["tesseract", "vlm", "none", "error"])
            
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_trocr_engine_with_mocked_dependencies(self):
        """Test TrOCR engine with mocked dependencies."""
        try:
            # Check if transformers is available
            import transformers
            from od_parse.ocr.trocr_engine import TrOCREngine

            # Enable TrOCR feature
            self.config.enable_feature("trocr", check_dependencies=False)

            with patch('transformers.TrOCRProcessor') as mock_processor, \
                 patch('transformers.VisionEncoderDecoderModel') as mock_model, \
                 patch('torch.cuda.is_available', return_value=False):

                # Mock the processor and model
                mock_processor_instance = MagicMock()
                mock_model_instance = MagicMock()

                mock_processor.from_pretrained.return_value = mock_processor_instance
                mock_model.from_pretrained.return_value = mock_model_instance

                # Mock processor output
                mock_processor_instance.return_value = MagicMock(pixel_values=MagicMock())
                mock_processor_instance.batch_decode.return_value = ["Sample text"]

                # Mock model output
                mock_model_instance.generate.return_value = MagicMock()

                engine = TrOCREngine()

                # Test if initialization was attempted
                self.assertTrue(mock_processor.from_pretrained.called)
                self.assertTrue(mock_model.from_pretrained.called)

        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_image_preparation(self):
        """Test image preparation from different formats."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            
            engine = TrOCREngine()
            
            # Test PIL Image
            pil_result = engine._prepare_image(self.test_image)
            self.assertIsInstance(pil_result, Image.Image)
            
            # Test numpy array
            np_array = np.array(self.test_image)
            np_result = engine._prepare_image(np_array)
            self.assertIsInstance(np_result, Image.Image)
            
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_confidence_estimation(self):
        """Test confidence estimation logic."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            
            engine = TrOCREngine()
            
            # Test with good text
            good_confidence = engine._estimate_confidence("This is clear text")
            self.assertGreater(good_confidence, 0.5)
            
            # Test with empty text
            empty_confidence = engine._estimate_confidence("")
            self.assertEqual(empty_confidence, 0.0)
            
            # Test with artifacts
            artifact_confidence = engine._estimate_confidence("Text with |||")
            self.assertLess(artifact_confidence, good_confidence)
            
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_engine_info(self):
        """Test engine information retrieval."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            
            engine = TrOCREngine()
            info = engine.get_engine_info()
            
            # Check required fields
            self.assertIn("trocr_available", info)
            self.assertIn("fallback_available", info)
            self.assertIn("current_engine", info)
            
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_batch_processing(self):
        """Test batch text extraction."""
        try:
            from od_parse.ocr.trocr_engine import TrOCREngine
            
            engine = TrOCREngine()
            
            # Create multiple test images
            images = [self.test_image, self.test_image]
            
            results = engine.batch_extract_text(images)
            
            # Should return results for all images
            self.assertEqual(len(results), 2)
            
            # Each result should have required fields
            for result in results:
                self.assertIn("text", result)
                self.assertIn("confidence", result)
                self.assertIn("engine", result)
                self.assertIn("image_index", result)
                
        except ImportError:
            self.skipTest("TrOCR dependencies not available")
    
    def test_convenience_function(self):
        """Test the convenience function for TrOCR."""
        try:
            from od_parse.ocr.trocr_engine import extract_text_with_trocr
            
            result = extract_text_with_trocr(self.test_image)
            
            # Should return a valid result dictionary
            self.assertIsInstance(result, dict)
            self.assertIn("text", result)
            self.assertIn("confidence", result)
            self.assertIn("engine", result)
            
        except ImportError:
            self.skipTest("TrOCR dependencies not available")


class TestTrOCRConfiguration(unittest.TestCase):
    """Test TrOCR configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
    
    def test_trocr_feature_configuration(self):
        """Test TrOCR feature can be enabled/disabled."""
        # Test enabling
        result = self.config.enable_feature("trocr", check_dependencies=False)
        self.assertTrue(result)
        self.assertTrue(self.config.is_feature_enabled("trocr"))
        
        # Test disabling
        result = self.config.disable_feature("trocr")
        self.assertTrue(result)
        self.assertFalse(self.config.is_feature_enabled("trocr"))
    
    def test_trocr_feature_info(self):
        """Test TrOCR feature information."""
        info = self.config.get_feature_info("trocr")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "TrOCR")
        self.assertIn("dependencies", info)
        self.assertIn("torch", info["dependencies"])
        self.assertIn("transformers", info["dependencies"])


if __name__ == "__main__":
    unittest.main()
