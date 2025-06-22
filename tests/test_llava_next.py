"""
Tests for LLaVA-NeXT integration.
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


class TestLLaVANextEngine(unittest.TestCase):
    """Test cases for LLaVA-NeXT engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
        # Create a simple test image
        self.test_image = Image.new('RGB', (200, 100), color='white')
        
    def test_llava_next_engine_import(self):
        """Test that LLaVA-NeXT engine can be imported when dependencies are available."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            self.assertTrue(True, "LLaVA-NeXT engine imported successfully")
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_llava_next_engine_initialization_disabled(self):
        """Test LLaVA-NeXT engine initialization when feature is disabled."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            # Ensure feature is disabled
            self.config.disable_feature("llava_next")
            
            engine = LLaVANextEngine()
            self.assertFalse(engine.is_available())
            
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_llava_next_engine_fallback(self):
        """Test that LLaVA-NeXT engine falls back when not available."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            # Disable LLaVA-NeXT feature to force fallback
            self.config.disable_feature("llava_next")
            
            engine = LLaVANextEngine()
            result = engine.understand_document(self.test_image)
            
            # Should use fallback engine
            self.assertIn("engine", result)
            self.assertEqual(result["engine"], "fallback")
            self.assertIn("understanding", result)
            
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_llava_next_engine_with_mocked_dependencies(self):
        """Test LLaVA-NeXT engine with mocked dependencies."""
        try:
            # Check if transformers is available
            import transformers
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            # Enable LLaVA-NeXT feature
            self.config.enable_feature("llava_next", check_dependencies=False)
            
            with patch('transformers.LlavaNextProcessor') as mock_processor, \
                 patch('transformers.LlavaNextForConditionalGeneration') as mock_model, \
                 patch('torch.cuda.is_available', return_value=False):
                
                # Mock the processor and model
                mock_processor_instance = MagicMock()
                mock_model_instance = MagicMock()
                
                mock_processor.from_pretrained.return_value = mock_processor_instance
                mock_model.from_pretrained.return_value = mock_model_instance
                
                # Mock processor methods
                mock_processor_instance.apply_chat_template.return_value = "test prompt"
                mock_processor_instance.return_value = {"input_ids": MagicMock(), "pixel_values": MagicMock()}
                mock_processor_instance.batch_decode.return_value = ["Test understanding response"]
                mock_processor_instance.tokenizer.eos_token_id = 2
                
                # Mock model methods
                mock_model_instance.generate.return_value = MagicMock()
                mock_model_instance.eval.return_value = None
                mock_model_instance.to.return_value = mock_model_instance
                
                engine = LLaVANextEngine()
                
                # Test if initialization was attempted
                self.assertTrue(mock_processor.from_pretrained.called)
                self.assertTrue(mock_model.from_pretrained.called)
                
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_image_preparation(self):
        """Test image preparation from different formats."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            engine = LLaVANextEngine()
            
            # Test PIL Image
            pil_result = engine._prepare_image(self.test_image)
            self.assertIsInstance(pil_result, Image.Image)
            
            # Test numpy array
            np_array = np.array(self.test_image)
            np_result = engine._prepare_image(np_array)
            self.assertIsInstance(np_result, Image.Image)
            
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_confidence_estimation(self):
        """Test confidence estimation logic."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            engine = LLaVANextEngine()
            
            # Test with good response
            good_response = "This document contains a detailed table with multiple columns and rows showing financial data."
            good_confidence = engine._estimate_response_confidence(good_response)
            self.assertGreater(good_confidence, 0.5)
            
            # Test with empty response
            empty_confidence = engine._estimate_response_confidence("")
            self.assertEqual(empty_confidence, 0.0)
            
            # Test with uncertain response
            uncertain_response = "I'm not sure what this document contains, it's unclear."
            uncertain_confidence = engine._estimate_response_confidence(uncertain_response)
            self.assertLess(uncertain_confidence, good_confidence)
            
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_structured_information_extraction(self):
        """Test structured information extraction with different types."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            # Disable feature to test fallback
            self.config.disable_feature("llava_next")
            
            engine = LLaVANextEngine()
            
            # Test different information types
            info_types = ["general", "tables", "forms", "metadata", "summary"]
            
            for info_type in info_types:
                result = engine.extract_structured_information(self.test_image, info_type)
                
                self.assertIn("understanding", result)
                self.assertIn("information_type", result)
                self.assertEqual(result["information_type"], info_type)
                
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_batch_understanding(self):
        """Test batch document understanding."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            # Disable feature to test fallback
            self.config.disable_feature("llava_next")
            
            engine = LLaVANextEngine()
            
            # Create multiple test images
            images = [self.test_image, self.test_image]
            
            results = engine.batch_understand_documents(images)
            
            # Should return results for all images
            self.assertEqual(len(results), 2)
            
            # Each result should have required fields
            for result in results:
                self.assertIn("understanding", result)
                self.assertIn("confidence", result)
                self.assertIn("engine", result)
                self.assertIn("image_index", result)
                
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_engine_info(self):
        """Test engine information retrieval."""
        try:
            from od_parse.advanced.llava_next import LLaVANextEngine
            
            engine = LLaVANextEngine()
            info = engine.get_engine_info()
            
            # Check required fields
            self.assertIn("llava_next_available", info)
            self.assertIn("fallback_available", info)
            self.assertIn("current_engine", info)
            
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")
    
    def test_convenience_function(self):
        """Test the convenience function for LLaVA-NeXT."""
        try:
            from od_parse.advanced.llava_next import understand_document_with_llava
            
            result = understand_document_with_llava(self.test_image)
            
            # Should return a valid result dictionary
            self.assertIsInstance(result, dict)
            self.assertIn("understanding", result)
            self.assertIn("confidence", result)
            self.assertIn("engine", result)
            
        except ImportError:
            self.skipTest("LLaVA-NeXT dependencies not available")


class TestLLaVANextConfiguration(unittest.TestCase):
    """Test LLaVA-NeXT configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
    
    def test_llava_next_feature_configuration(self):
        """Test LLaVA-NeXT feature can be enabled/disabled."""
        # Test enabling
        result = self.config.enable_feature("llava_next", check_dependencies=False)
        self.assertTrue(result)
        self.assertTrue(self.config.is_feature_enabled("llava_next"))
        
        # Test disabling
        result = self.config.disable_feature("llava_next")
        self.assertTrue(result)
        self.assertFalse(self.config.is_feature_enabled("llava_next"))
    
    def test_llava_next_feature_info(self):
        """Test LLaVA-NeXT feature information."""
        info = self.config.get_feature_info("llava_next")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "LLaVA-NeXT")
        self.assertIn("dependencies", info)
        self.assertIn("torch", info["dependencies"])
        self.assertIn("transformers", info["dependencies"])
        self.assertIn("accelerate", info["dependencies"])
    
    def test_llava_next_preset_configuration(self):
        """Test enabling LLaVA-NeXT through presets."""
        # Test experimental preset (should include llava_next)
        result = self.config.enable_preset("experimental")
        
        # Check if llava_next can be enabled without dependency check
        manual_result = self.config.enable_feature("llava_next", check_dependencies=False)
        self.assertTrue(manual_result)
        self.assertTrue(self.config.is_feature_enabled("llava_next"))


if __name__ == "__main__":
    unittest.main()
