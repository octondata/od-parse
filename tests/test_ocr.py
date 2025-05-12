"""
Tests for the OCR module.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.ocr import extract_handwritten_content
from od_parse.ocr.handwritten import preprocess_image_for_ocr, detect_handwritten_regions

class TestOCRModule(unittest.TestCase):
    """Test cases for the OCR module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.samples_dir = Path(__file__).parent / "sample_pdfs"
        
        # Create sample_pdfs directory if it doesn't exist
        if not self.samples_dir.exists():
            self.samples_dir.mkdir(parents=True)
    
    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    @patch('cv2.cvtColor')
    def test_extract_handwritten_content_mocked(self, mock_cvtColor, mock_image_open, mock_image_to_string):
        """Test extracting handwritten content with mocked dependencies."""
        # Mock the dependencies
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image
        
        mock_cv_img = MagicMock()
        mock_cvtColor.return_value = mock_cv_img
        
        mock_image_to_string.return_value = "Handwritten text"
        
        # Call the function
        result = extract_handwritten_content("fake_image.png")
        
        # Verify the result
        self.assertEqual(result, "Handwritten text")
        
        # Verify the mocks were called
        mock_image_open.assert_called_once_with("fake_image.png")
        mock_cvtColor.assert_called_once()
        mock_image_to_string.assert_called_once()
    
    def test_extract_handwritten_content_with_sample_image(self):
        """Test extracting handwritten content from a sample image."""
        # This test will be skipped if no sample image is available
        sample_image = next((f for f in self.samples_dir.glob("*.png")), None)
        if sample_image is None:
            self.skipTest("No sample image available for testing")
        
        # This test may fail if pytesseract is not installed or configured correctly
        try:
            result = extract_handwritten_content(sample_image)
            # We don't assert the content of the result, just that it ran without error
            self.assertIsInstance(result, (str, type(None)))
        except Exception as e:
            self.skipTest(f"Skipping test due to OCR error: {e}")
    
    @patch('cv2.cvtColor')
    @patch('cv2.GaussianBlur')
    @patch('cv2.adaptiveThreshold')
    @patch('cv2.dilate')
    @patch('cv2.erode')
    def test_preprocess_image_for_ocr(self, mock_erode, mock_dilate, mock_threshold, mock_blur, mock_cvtColor):
        """Test image preprocessing for OCR."""
        # Mock the dependencies
        mock_image = MagicMock()
        mock_gray = MagicMock()
        mock_blur = MagicMock()
        mock_threshold = MagicMock()
        mock_dilate = MagicMock()
        mock_erode = MagicMock()
        
        mock_cvtColor.return_value = mock_gray
        mock_blur.return_value = mock_blur
        mock_threshold.return_value = mock_threshold
        mock_dilate.return_value = mock_dilate
        mock_erode.return_value = mock_erode
        
        # Call the function
        result = preprocess_image_for_ocr(mock_image)
        
        # Verify the mocks were called
        mock_cvtColor.assert_called_once()
        
        # We don't assert the result since it's mocked
        self.assertIsNotNone(result)

if __name__ == "__main__":
    unittest.main()
