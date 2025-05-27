"""
Tests for the Markdown converter module.
"""

import os
import sys
import tempfile
import unittest

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.converter import convert_to_markdown
from od_parse.converter.markdown_converter import (
    format_form_element,
    format_table,
    format_text_content,
)


class TestMarkdownConverter(unittest.TestCase):
    """Test cases for the Markdown converter module."""

    def test_convert_to_markdown_basic(self):
        """Test basic Markdown conversion."""
        # Create a simple parsed data dictionary
        parsed_data = {
            "text": "This is a test document.\n\nIt has multiple paragraphs.",
            "images": [],
            "tables": [],
            "forms": [],
            "handwritten_content": [],
        }

        # Convert to Markdown
        markdown = convert_to_markdown(parsed_data)

        # Verify the result
        self.assertIsInstance(markdown, str)
        self.assertIn("This is a test document.", markdown)
        self.assertIn("It has multiple paragraphs.", markdown)
        self.assertIn("## Content", markdown)

    def test_convert_to_markdown_with_images(self):
        """Test Markdown conversion with images."""
        # Create a parsed data dictionary with images
        parsed_data = {
            "text": "Document with images.",
            "images": ["/path/to/image1.png", "/path/to/image2.png"],
            "tables": [],
            "forms": [],
            "handwritten_content": [],
        }

        # Convert to Markdown
        markdown = convert_to_markdown(parsed_data)

        # Verify the result
        self.assertIn("## Images", markdown)
        self.assertIn("![Image 1](/path/to/image1.png)", markdown)
        self.assertIn("![Image 2](/path/to/image2.png)", markdown)

    def test_convert_to_markdown_with_tables(self):
        """Test Markdown conversion with tables."""
        # Create a parsed data dictionary with tables
        parsed_data = {
            "text": "Document with tables.",
            "images": [],
            "tables": [
                [{"Name": "John", "Age": 30}, {"Name": "Jane", "Age": 25}],
                [
                    {"Product": "Widget", "Price": 10.99},
                    {"Product": "Gadget", "Price": 19.99},
                ],
            ],
            "forms": [],
            "handwritten_content": [],
        }

        # Convert to Markdown
        markdown = convert_to_markdown(parsed_data)

        # Verify the result
        self.assertIn("## Tables", markdown)
        self.assertIn("### Table 1", markdown)
        self.assertIn("### Table 2", markdown)

    def test_convert_to_markdown_with_forms(self):
        """Test Markdown conversion with form elements."""
        # Create a parsed data dictionary with form elements
        parsed_data = {
            "text": "Document with form elements.",
            "images": [],
            "tables": [],
            "forms": [
                {
                    "type": "checkbox",
                    "text": "Check this box",
                    "page": 1,
                    "bbox": (10, 20, 30, 40),
                },
                {
                    "type": "radio",
                    "text": "Select an option",
                    "page": 1,
                    "bbox": (50, 60, 70, 80),
                },
            ],
            "handwritten_content": [],
        }

        # Convert to Markdown
        markdown = convert_to_markdown(parsed_data)

        # Verify the result
        self.assertIn("## Form Elements", markdown)
        self.assertIn("### Form Element 1", markdown)
        self.assertIn("**Type:** checkbox", markdown)
        self.assertIn("**Text:** Check this box", markdown)

    def test_convert_to_markdown_with_handwritten(self):
        """Test Markdown conversion with handwritten content."""
        # Create a parsed data dictionary with handwritten content
        parsed_data = {
            "text": "Document with handwritten content.",
            "images": [],
            "tables": [],
            "forms": [],
            "handwritten_content": [
                "This is handwritten text.",
                "More handwritten text.",
            ],
        }

        # Convert to Markdown
        markdown = convert_to_markdown(parsed_data)

        # Verify the result
        self.assertIn("## Handwritten Content", markdown)
        self.assertIn("### Handwritten Text 1", markdown)
        self.assertIn("This is handwritten text.", markdown)
        self.assertIn("### Handwritten Text 2", markdown)
        self.assertIn("More handwritten text.", markdown)

    def test_convert_to_markdown_with_output_file(self):
        """Test Markdown conversion with output to file."""
        # Create a simple parsed data dictionary
        parsed_data = {
            "text": "Test document for file output.",
            "images": [],
            "tables": [],
            "forms": [],
            "handwritten_content": [],
        }

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Convert to Markdown and write to file
            markdown = convert_to_markdown(parsed_data, output_file=temp_path)

            # Verify the file was created and contains the expected content
            self.assertTrue(os.path.exists(temp_path))
            with open(temp_path) as f:
                content = f.read()

            self.assertIn("Test document for file output.", content)
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_format_text_content(self):
        """Test formatting text content for Markdown."""
        # Test with single paragraph
        text = "This is a single paragraph."
        formatted = format_text_content(text)
        self.assertEqual(formatted, text)

        # Test with multiple paragraphs
        text = "This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3."
        formatted = format_text_content(text)
        self.assertEqual(formatted, text)

        # Test with empty lines
        text = "Paragraph 1.\n\n\n\nParagraph 2."
        formatted = format_text_content(text)
        self.assertEqual(formatted, "Paragraph 1.\n\nParagraph 2.")

    def test_format_table(self):
        """Test formatting table data as Markdown table."""
        # Test with empty table
        table_data = []
        formatted = format_table(table_data)
        self.assertEqual(formatted, "*No table data available*")

        # Test with simple table
        table_data = [
            {"Name": "John", "Age": 30, "City": "New York"},
            {"Name": "Jane", "Age": 25, "City": "Los Angeles"},
        ]
        formatted = format_table(table_data)
        self.assertIn("| Name | Age | City |", formatted)
        self.assertIn("| --- | --- | --- |", formatted)
        self.assertIn("| John | 30 | New York |", formatted)
        self.assertIn("| Jane | 25 | Los Angeles |", formatted)

    def test_format_form_element(self):
        """Test formatting form element data for Markdown."""
        # Test with minimal form element
        form_data = {"type": "checkbox"}
        formatted = format_form_element(form_data)
        self.assertEqual(formatted, "**Type:** checkbox")

        # Test with complete form element
        form_data = {
            "type": "radio",
            "text": "Select an option",
            "page": 2,
            "bbox": (10.5, 20.5, 30.5, 40.5),
        }
        formatted = format_form_element(form_data)
        self.assertIn("**Type:** radio", formatted)
        self.assertIn("**Text:** Select an option", formatted)
        self.assertIn("**Page:** 2", formatted)
        self.assertIn(
            "**Position:** (x1=10.50, y1=20.50, x2=30.50, y2=40.50)", formatted
        )


if __name__ == "__main__":
    unittest.main()
