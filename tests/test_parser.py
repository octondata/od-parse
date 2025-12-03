"""
Tests for the PDF parser module.
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.parser import parse_pdf, extract_text, extract_images, extract_tables, extract_forms
from od_parse.utils.file_utils import validate_file, FileValidationError

class TestPDFParser(unittest.TestCase):
    """Test cases for the PDF parser module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Check both sample directories - prefer 'samples' over 'sample_pdfs'
        self.samples_dir = Path(__file__).parent / "samples"
        if not self.samples_dir.exists():
            self.samples_dir = Path(__file__).parent / "sample_pdfs"

        # Create directory if it doesn't exist
        if not self.samples_dir.exists():
            self.samples_dir.mkdir(parents=True)
    
    def test_validate_file(self):
        """Test file validation."""
        # Test with non-existent file
        with self.assertRaises(FileValidationError):
            validate_file(self.samples_dir / "non_existent.pdf")
        
        # Test with directory instead of file
        with self.assertRaises(FileValidationError):
            validate_file(self.samples_dir)
        
        # Create a temporary test file
        test_file = self.samples_dir / "test.txt"
        with open(test_file, "w") as f:
            f.write("Test content")
        
        # Test with incorrect extension
        with self.assertRaises(FileValidationError):
            validate_file(test_file, extension=".pdf")
        
        # Test with correct extension
        validated_file = validate_file(test_file, extension=".txt")
        self.assertEqual(validated_file, test_file)
        
        # Clean up
        test_file.unlink()
    
    def test_parse_pdf_with_missing_file(self):
        """Test parsing a non-existent PDF file."""
        with self.assertRaises(FileValidationError):
            parse_pdf(self.samples_dir / "non_existent.pdf")
    
    def test_extract_text_with_sample_pdf(self):
        """Test extracting text from a sample PDF file."""
        # This test will be skipped if no sample PDF is available
        sample_pdf = next((f for f in self.samples_dir.glob("*.pdf")), None)
        if sample_pdf is None:
            self.skipTest("No sample PDF available for testing")
        
        text = extract_text(sample_pdf)
        self.assertIsInstance(text, str)
    
    def test_extract_images_with_sample_pdf(self):
        """Test extracting images from a sample PDF file."""
        # This test will be skipped if no sample PDF is available
        sample_pdf = next((f for f in self.samples_dir.glob("*.pdf")), None)
        if sample_pdf is None:
            self.skipTest("No sample PDF available for testing")
        
        images = extract_images(sample_pdf)
        self.assertIsInstance(images, list)
    
    def test_extract_tables_with_sample_pdf(self):
        """Test extracting tables from a sample PDF file."""
        # This test will be skipped if no sample PDF is available
        sample_pdf = next((f for f in self.samples_dir.glob("*.pdf")), None)
        if sample_pdf is None:
            self.skipTest("No sample PDF available for testing")
        
        tables = extract_tables(sample_pdf)
        self.assertIsInstance(tables, list)
    
    def test_extract_forms_with_sample_pdf(self):
        """Test extracting form elements from a sample PDF file."""
        # This test will be skipped if no sample PDF is available
        sample_pdf = next((f for f in self.samples_dir.glob("*.pdf")), None)
        if sample_pdf is None:
            self.skipTest("No sample PDF available for testing")
        
        forms = extract_forms(sample_pdf)
        self.assertIsInstance(forms, list)

if __name__ == "__main__":
    unittest.main()
