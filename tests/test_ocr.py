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
        # Check both sample directories
        self.samples_dir = Path(__file__).parent / "samples"
        if not self.samples_dir.exists():
            self.samples_dir = Path(__file__).parent / "sample_pdfs"

        # Create directory if it doesn't exist
        if not self.samples_dir.exists():
            self.samples_dir.mkdir(parents=True)
    
    @patch('pytesseract.image_to_string')
    @patch('PIL.Image.open')
    @patch('od_parse.ocr.handwritten.preprocess_image_for_ocr')
    @patch('cv2.cvtColor')
    @patch('numpy.array')
    def test_extract_handwritten_content_mocked(self, mock_np_array, mock_cvtColor, mock_preprocess, mock_image_open, mock_image_to_string):
        """Test extracting handwritten content with mocked dependencies."""
        import numpy as np

        # Mock the dependencies
        mock_image = MagicMock()
        mock_image_open.return_value = mock_image

        # Create a proper numpy array mock
        fake_array = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_np_array.return_value = fake_array

        # Mock cv2.cvtColor to return a proper numpy array
        fake_cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvtColor.return_value = fake_cv_img

        # Mock the preprocessing function to return a proper numpy array
        fake_processed_img = np.zeros((100, 100), dtype=np.uint8)
        mock_preprocess.return_value = fake_processed_img

        mock_image_to_string.return_value = "Handwritten text"

        # Call the function
        result = extract_handwritten_content("fake_image.png")

        # Verify the result
        self.assertEqual(result, "Handwritten text")

        # Verify the mocks were called
        mock_image_open.assert_called_once_with("fake_image.png")
        mock_cvtColor.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_image_to_string.assert_called_once()
    
    def test_extract_handwritten_content_with_sample_image(self):
        """Test extracting handwritten content from a sample image."""
        # This test will be skipped if no sample image is available
        # Look for common image formats
        sample_image = None
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            sample_image = next((f for f in self.samples_dir.glob(ext)), None)
            if sample_image:
                break

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
        import numpy as np

        # Create proper numpy arrays for mocking
        input_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        blurred_image = np.zeros((100, 100), dtype=np.uint8)
        threshold_image = np.zeros((100, 100), dtype=np.uint8)
        dilated_image = np.zeros((100, 100), dtype=np.uint8)
        eroded_image = np.zeros((100, 100), dtype=np.uint8)

        # Set up the mock chain
        mock_cvtColor.return_value = gray_image
        mock_blur.return_value = blurred_image
        mock_threshold.return_value = threshold_image
        mock_dilate.return_value = dilated_image
        mock_erode.return_value = eroded_image

        # Call the function
        result = preprocess_image_for_ocr(input_image)

        # Verify the mocks were called
        mock_cvtColor.assert_called_once()
        mock_blur.assert_called_once()
        mock_threshold.assert_called_once()
        mock_dilate.assert_called_once()
        mock_erode.assert_called_once()

        # Verify the result is the final processed image
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, eroded_image.shape)

if __name__ == "__main__":
    unittest.main()
