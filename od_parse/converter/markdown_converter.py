"""
Module for converting parsed PDF content to Markdown format.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)

def convert_to_markdown(parsed_data: Dict[str, Any], **kwargs) -> str:
    """
    Convert parsed PDF data to Markdown format.
    
    Args:
        parsed_data: Dictionary containing parsed PDF data
        **kwargs: Additional arguments for conversion
    
    Returns:
        Markdown formatted string
    """
    output_file = kwargs.get('output_file', None)
    include_images = kwargs.get('include_images', True)
    include_tables = kwargs.get('include_tables', True)
    include_forms = kwargs.get('include_forms', True)
    include_handwritten = kwargs.get('include_handwritten', True)
    
    markdown = []
    
    # Add title if provided
    if 'title' in parsed_data:
        markdown.append(f"# {parsed_data['title']}\n")
    
    # Add text content
    if 'text' in parsed_data and parsed_data['text']:
        markdown.append("## Content\n")
        markdown.append(format_text_content(parsed_data['text']))
        markdown.append("\n")
    
    # Add images
    if include_images and 'images' in parsed_data and parsed_data['images']:
        markdown.append("## Images\n")
        for i, img_path in enumerate(parsed_data['images']):
            img_path = Path(img_path)
            markdown.append(f"![Image {i+1}]({img_path})\n")
        markdown.append("\n")
    
    # Add tables
    if include_tables and 'tables' in parsed_data and parsed_data['tables']:
        markdown.append("## Tables\n")
        for i, table in enumerate(parsed_data['tables']):
            markdown.append(f"### Table {i+1}\n")
            markdown.append(format_table(table))
            markdown.append("\n")
    
    # Add form elements
    if include_forms and 'forms' in parsed_data and parsed_data['forms']:
        markdown.append("## Form Elements\n")
        for i, form in enumerate(parsed_data['forms']):
            markdown.append(f"### Form Element {i+1}\n")
            markdown.append(format_form_element(form))
            markdown.append("\n")
    
    # Add handwritten content
    if include_handwritten and 'handwritten_content' in parsed_data and parsed_data['handwritten_content']:
        markdown.append("## Handwritten Content\n")
        for i, content in enumerate(parsed_data['handwritten_content']):
            if content:
                markdown.append(f"### Handwritten Text {i+1}\n")
                markdown.append(f"{content}\n\n")
    
    # Join all markdown sections
    markdown_text = "\n".join(markdown)
    
    # Write to file if output_file is provided
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"Markdown saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving Markdown to {output_file}: {e}")
    
    return markdown_text

def format_text_content(text: str) -> str:
    """
    Format text content for Markdown.
    
    Args:
        text: Text content to format
    
    Returns:
        Formatted text
    """
    # Split text into paragraphs
    paragraphs = text.split('\n\n')
    
    # Format each paragraph
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Clean up paragraph
        paragraph = paragraph.strip()
        if paragraph:
            formatted_paragraphs.append(paragraph)
    
    return '\n\n'.join(formatted_paragraphs)

def format_table(table_data: List[Dict[str, Any]]) -> str:
    """
    Format table data as Markdown table.
    
    Args:
        table_data: Table data as a list of dictionaries
    
    Returns:
        Markdown formatted table
    """
    if not table_data:
        return "*No table data available*"
    
    # Get column headers
    headers = list(table_data[0].keys())
    
    # Create table header
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Add table rows
    for row in table_data:
        row_values = []
        for header in headers:
            value = row.get(header, "")
            # Convert non-string values to strings
            if not isinstance(value, str):
                value = str(value)
            # Escape pipe characters in cell values
            value = value.replace("|", "\\|")
            row_values.append(value)
        markdown_table += "| " + " | ".join(row_values) + " |\n"
    
    return markdown_table

def format_form_element(form_data: Dict[str, Any]) -> str:
    """
    Format form element data for Markdown.
    
    Args:
        form_data: Form element data
    
    Returns:
        Markdown formatted form element description
    """
    markdown = []
    
    # Add form element type
    if 'type' in form_data:
        markdown.append(f"**Type:** {form_data['type']}")
    
    # Add form element text
    if 'text' in form_data:
        markdown.append(f"**Text:** {form_data['text']}")
    
    # Add form element position
    if 'page' in form_data:
        markdown.append(f"**Page:** {form_data['page']}")
    
    # Add form element bounding box
    if 'bbox' in form_data:
        bbox = form_data['bbox']
        markdown.append(f"**Position:** (x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f})")
    
    return "\n".join(markdown)
