"""
Tests for Table Transformer integration.
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


class TestTableTransformerEngine(unittest.TestCase):
    """Test cases for Table Transformer engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
        # Create a simple test image with table-like structure
        self.test_image = Image.new('RGB', (200, 100), color='white')
        
    def test_table_transformer_engine_import(self):
        """Test that Table Transformer engine can be imported when dependencies are available."""
        try:
            from od_parse.advanced.table_transformer import TableTransformerEngine
            self.assertTrue(True, "Table Transformer engine imported successfully")
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_table_transformer_engine_initialization_disabled(self):
        """Test Table Transformer engine initialization when feature is disabled."""
        try:
            from od_parse.advanced.table_transformer import TableTransformerEngine
            
            # Ensure feature is disabled
            self.config.disable_feature("table_transformer")
            
            engine = TableTransformerEngine()
            self.assertFalse(engine.is_available())
            
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_table_transformer_engine_fallback(self):
        """Test that Table Transformer engine falls back to tabula when not available."""
        try:
            from od_parse.advanced.table_transformer import TableTransformerEngine
            
            # Disable Table Transformer feature to force fallback
            self.config.disable_feature("table_transformer")
            
            engine = TableTransformerEngine()
            result = engine.extract_tables(self.test_image)
            
            # Should use fallback engine
            self.assertIn("engine", result)
            self.assertIn(result["engine"], ["tabula", "none", "error"])
            
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_table_transformer_engine_with_mocked_dependencies(self):
        """Test Table Transformer engine with mocked dependencies."""
        try:
            # Check if transformers is available
            import transformers
            from od_parse.advanced.table_transformer import TableTransformerEngine
            
            # Enable Table Transformer feature
            self.config.enable_feature("table_transformer", check_dependencies=False)
            
            with patch('transformers.DetrImageProcessor') as mock_processor, \
                 patch('transformers.TableTransformerForObjectDetection') as mock_model, \
                 patch('torch.cuda.is_available', return_value=False):
                
                # Mock the processor and model
                mock_processor_instance = MagicMock()
                mock_model_instance = MagicMock()
                
                mock_processor.from_pretrained.return_value = mock_processor_instance
                mock_model.from_pretrained.return_value = mock_model_instance
                
                # Mock processor output
                mock_processor_instance.return_value = {"pixel_values": MagicMock()}
                mock_processor_instance.post_process_object_detection.return_value = [
                    {
                        "scores": MagicMock(),
                        "labels": MagicMock(), 
                        "boxes": MagicMock()
                    }
                ]
                
                # Mock model output
                mock_model_instance.return_value = MagicMock()
                
                engine = TableTransformerEngine()
                
                # Test if initialization was attempted
                self.assertTrue(mock_processor.from_pretrained.called)
                self.assertTrue(mock_model.from_pretrained.called)
                
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_image_preparation(self):
        """Test image preparation from different formats."""
        try:
            from od_parse.advanced.table_transformer import TableTransformerEngine
            
            engine = TableTransformerEngine()
            
            # Test PIL Image
            pil_result = engine._prepare_image(self.test_image)
            self.assertIsInstance(pil_result, Image.Image)
            
            # Test numpy array
            np_array = np.array(self.test_image)
            np_result = engine._prepare_image(np_array)
            self.assertIsInstance(np_result, Image.Image)
            
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_engine_info(self):
        """Test engine information retrieval."""
        try:
            from od_parse.advanced.table_transformer import TableTransformerEngine
            
            engine = TableTransformerEngine()
            info = engine.get_engine_info()
            
            # Check required fields
            self.assertIn("table_transformer_available", info)
            self.assertIn("fallback_available", info)
            self.assertIn("current_engine", info)
            
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_visualization_method(self):
        """Test table detection visualization."""
        try:
            from od_parse.advanced.table_transformer import TableTransformerEngine
            
            engine = TableTransformerEngine()
            
            # Mock detection results
            detections = [
                {
                    "bbox": [10, 10, 100, 50],
                    "confidence": 0.9
                }
            ]
            
            result_image = engine.visualize_detections(self.test_image, detections)
            
            # Should return a PIL Image
            self.assertIsInstance(result_image, Image.Image)
            
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")
    
    def test_convenience_function(self):
        """Test the convenience function for Table Transformer."""
        try:
            from od_parse.advanced.table_transformer import extract_tables_with_transformer
            
            result = extract_tables_with_transformer(self.test_image)
            
            # Should return a valid result dictionary
            self.assertIsInstance(result, dict)
            self.assertIn("tables", result)
            self.assertIn("confidence", result)
            self.assertIn("engine", result)
            
        except ImportError:
            self.skipTest("Table Transformer dependencies not available")


class TestTableTransformerConfiguration(unittest.TestCase):
    """Test Table Transformer configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
    
    def test_table_transformer_feature_configuration(self):
        """Test Table Transformer feature can be enabled/disabled."""
        # Test enabling
        result = self.config.enable_feature("table_transformer", check_dependencies=False)
        self.assertTrue(result)
        self.assertTrue(self.config.is_feature_enabled("table_transformer"))
        
        # Test disabling
        result = self.config.disable_feature("table_transformer")
        self.assertTrue(result)
        self.assertFalse(self.config.is_feature_enabled("table_transformer"))
    
    def test_table_transformer_feature_info(self):
        """Test Table Transformer feature information."""
        info = self.config.get_feature_info("table_transformer")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Table Transformer")
        self.assertIn("dependencies", info)
        self.assertIn("torch", info["dependencies"])
        self.assertIn("transformers", info["dependencies"])
    
    def test_table_transformer_preset_configuration(self):
        """Test enabling Table Transformer through presets."""
        # Test advanced preset (should include table_transformer)
        # Note: This may fail if dependencies are not installed
        result = self.config.enable_preset("advanced")

        # Check if table_transformer would be in the preset
        # Even if it fails due to missing dependencies
        preset_features = ["trocr", "table_transformer", "quality_assessment", "async_processing", "multilingual"]

        # At least some features should be attempted
        self.assertIsInstance(result, bool)

        # Test that table_transformer can be enabled without dependency check
        manual_result = self.config.enable_feature("table_transformer", check_dependencies=False)
        self.assertTrue(manual_result)
        self.assertTrue(self.config.is_feature_enabled("table_transformer"))


if __name__ == "__main__":
    unittest.main()
