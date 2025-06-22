"""
Tests for Document Quality Assessment.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.config import get_advanced_config
from od_parse.quality import DocumentQualityAssessor, assess_document_quality


class TestDocumentQualityAssessor(unittest.TestCase):
    """Test cases for Document Quality Assessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
        self.assessor = DocumentQualityAssessor()
        
        # Sample extraction results for testing
        self.good_extraction = {
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
        
        self.poor_extraction = {
            "text": {
                "content": "Th|s |s p00r OCR w|th m@ny @rt|f@cts |||",
                "confidence": 0.3
            },
            "tables": [],
            "forms": []
        }
        
        # Create test image
        self.test_image = Image.new('RGB', (100, 100), color='white')
    
    def test_quality_assessor_initialization(self):
        """Test quality assessor initialization."""
        assessor = DocumentQualityAssessor()
        self.assertIsInstance(assessor, DocumentQualityAssessor)
    
    def test_assess_extraction_quality_good_document(self):
        """Test quality assessment with good extraction results."""
        result = self.assessor.assess_extraction_quality(self.good_extraction)
        
        # Check required fields
        self.assertIn("overall_score", result)
        self.assertIn("text_quality", result)
        self.assertIn("structure_quality", result)
        self.assertIn("confidence_metrics", result)
        self.assertIn("completeness", result)
        self.assertIn("consistency", result)
        self.assertIn("recommendations", result)
        
        # Good extraction should have decent scores
        self.assertGreater(result["overall_score"], 0.5)
        self.assertGreater(result["text_quality"]["score"], 0.5)
        self.assertGreater(result["confidence_metrics"]["average_confidence"], 0.7)
    
    def test_assess_extraction_quality_poor_document(self):
        """Test quality assessment with poor extraction results."""
        result = self.assessor.assess_extraction_quality(self.poor_extraction)
        
        # Check required fields
        self.assertIn("overall_score", result)
        self.assertIn("text_quality", result)
        
        # Poor extraction should have lower scores
        self.assertLess(result["overall_score"], 0.7)
        self.assertLess(result["confidence_metrics"]["average_confidence"], 0.5)
    
    def test_text_quality_assessment(self):
        """Test text quality assessment specifically."""
        # Test with good text
        good_result = self.assessor._assess_text_quality(self.good_extraction)
        self.assertIn("score", good_result)
        self.assertIn("metrics", good_result)
        self.assertGreater(good_result["score"], 0.5)
        
        # Test with poor text
        poor_result = self.assessor._assess_text_quality(self.poor_extraction)
        self.assertLess(poor_result["score"], good_result["score"])
    
    def test_structure_quality_assessment(self):
        """Test structure quality assessment."""
        # Test with structured document
        structured_result = self.assessor._assess_structure_quality(self.good_extraction)
        self.assertIn("score", structured_result)
        self.assertIn("metrics", structured_result)
        
        # Should detect tables and forms
        self.assertTrue(structured_result["metrics"]["has_tables"])
        self.assertTrue(structured_result["metrics"]["has_forms"])
        
        # Test with unstructured document
        unstructured_result = self.assessor._assess_structure_quality(self.poor_extraction)
        self.assertFalse(unstructured_result["metrics"]["has_tables"])
        self.assertFalse(unstructured_result["metrics"]["has_forms"])
    
    def test_confidence_metrics_assessment(self):
        """Test confidence metrics assessment."""
        result = self.assessor._assess_confidence_metrics(self.good_extraction)
        
        self.assertIn("average_confidence", result)
        self.assertIn("confidence_variance", result)
        self.assertIn("min_confidence", result)
        self.assertIn("max_confidence", result)
        self.assertIn("confidence_distribution", result)
        
        # Should have reasonable confidence values
        self.assertGreater(result["average_confidence"], 0.7)
        self.assertGreaterEqual(result["min_confidence"], 0.0)
        self.assertLessEqual(result["max_confidence"], 1.0)
    
    def test_completeness_assessment(self):
        """Test completeness assessment."""
        result = self.assessor._assess_completeness(self.good_extraction)
        
        self.assertIn("score", result)
        self.assertIn("indicators", result)
        
        indicators = result["indicators"]
        self.assertTrue(indicators["has_text_content"])
        self.assertTrue(indicators["has_structured_data"])
        self.assertTrue(indicators["has_metadata"])
    
    def test_consistency_assessment(self):
        """Test consistency assessment."""
        result = self.assessor._assess_consistency(self.good_extraction)
        
        self.assertIn("score", result)
        self.assertIn("checks", result)
        
        # Should pass basic consistency checks
        self.assertGreater(result["score"], 0.5)
    
    def test_image_quality_assessment(self):
        """Test image quality assessment."""
        result = self.assessor._assess_image_quality(self.test_image)
        
        self.assertIn("score", result)
        self.assertIn("metrics", result)
        
        metrics = result["metrics"]
        self.assertIn("resolution", metrics)
        self.assertIn("brightness", metrics)
        self.assertIn("contrast", metrics)
        self.assertIn("sharpness", metrics)
    
    def test_readability_calculation(self):
        """Test readability score calculation."""
        good_text = "This is a clear and readable sentence with proper structure."
        poor_text = "Th|s |s p00r t3xt w|th m@ny pr0bl3ms."
        
        good_score = self.assessor._calculate_readability(good_text)
        poor_score = self.assessor._calculate_readability(poor_text)
        
        self.assertGreater(good_score, poor_score)
        self.assertGreaterEqual(good_score, 0)
        self.assertLessEqual(good_score, 100)
    
    def test_ocr_artifact_detection(self):
        """Test OCR artifact detection."""
        clean_text = "This is clean text without artifacts."
        artifact_text = "Th|s h@s m@ny ||| @rt|f@cts ___"
        
        clean_artifacts = self.assessor._detect_ocr_artifacts(clean_text)
        artifact_ratio = self.assessor._detect_ocr_artifacts(artifact_text)
        
        self.assertLess(clean_artifacts, artifact_ratio)
        self.assertGreaterEqual(clean_artifacts, 0.0)
        self.assertLessEqual(artifact_ratio, 1.0)
    
    def test_language_coherence_assessment(self):
        """Test language coherence assessment."""
        coherent_text = "This is a coherent sentence with proper word structure."
        incoherent_text = "a b c d e f g h i j k l m n o p q r s t"
        
        coherent_score = self.assessor._assess_language_coherence(coherent_text)
        incoherent_score = self.assessor._assess_language_coherence(incoherent_text)
        
        self.assertGreater(coherent_score, incoherent_score)
    
    def test_table_quality_assessment(self):
        """Test table quality assessment."""
        good_tables = [
            {
                "data": [["A", "B"], ["1", "2"]],
                "headers": ["A", "B"],
                "confidence": 0.9,
                "shape": (2, 2)
            }
        ]
        
        poor_tables = [
            {
                "confidence": 0.3
            }
        ]
        
        good_score = self.assessor._assess_table_quality(good_tables)
        poor_score = self.assessor._assess_table_quality(poor_tables)
        
        self.assertGreater(good_score, poor_score)
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        # Test with good extraction
        good_assessment = self.assessor.assess_extraction_quality(self.good_extraction)
        good_recommendations = good_assessment["recommendations"]
        
        self.assertIsInstance(good_recommendations, list)
        self.assertGreater(len(good_recommendations), 0)
        
        # Test with poor extraction
        poor_assessment = self.assessor.assess_extraction_quality(self.poor_extraction)
        poor_recommendations = poor_assessment["recommendations"]
        
        # Poor extraction should have more recommendations
        self.assertGreaterEqual(len(poor_recommendations), len(good_recommendations))
    
    def test_convenience_function(self):
        """Test the convenience function."""
        result = assess_document_quality(self.good_extraction)
        
        self.assertIsInstance(result, dict)
        self.assertIn("overall_score", result)
        self.assertIn("recommendations", result)
    
    def test_empty_extraction_handling(self):
        """Test handling of empty extraction results."""
        empty_extraction = {}
        
        result = self.assessor.assess_extraction_quality(empty_extraction)
        
        # Should handle empty input gracefully
        self.assertIn("overall_score", result)
        self.assertGreaterEqual(result["overall_score"], 0.0)
    
    def test_feature_configuration(self):
        """Test quality assessment feature configuration."""
        # Test enabling/disabling feature
        self.config.enable_feature("quality_assessment", check_dependencies=False)
        self.assertTrue(self.config.is_feature_enabled("quality_assessment"))
        
        self.config.disable_feature("quality_assessment")
        self.assertFalse(self.config.is_feature_enabled("quality_assessment"))


class TestQualityAssessmentConfiguration(unittest.TestCase):
    """Test quality assessment configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
    
    def test_quality_assessment_feature_info(self):
        """Test quality assessment feature information."""
        info = self.config.get_feature_info("quality_assessment")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Document Quality Assessment")
        self.assertIn("dependencies", info)
        self.assertIn("scikit-learn", info["dependencies"])
        self.assertIn("scipy", info["dependencies"])


if __name__ == "__main__":
    unittest.main()
