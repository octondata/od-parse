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
    output_file = kwargs.get("output_file", None)
    include_images = kwargs.get("include_images", True)
    include_tables = kwargs.get("include_tables", True)
    include_forms = kwargs.get("include_forms", True)
    include_handwritten = kwargs.get("include_handwritten", True)

    markdown = []

    # Add title if provided
    if "title" in parsed_data:
        markdown.append(f"# {parsed_data['title']}\n")

    # Add text content
    if "text" in parsed_data and parsed_data["text"]:
        markdown.append("## Content\n")
        markdown.append(format_text_content(parsed_data["text"]))
        markdown.append("\n")

    # Add images
    if include_images and "images" in parsed_data and parsed_data["images"]:
        markdown.append("## Images\n")
        for i, img_path in enumerate(parsed_data["images"]):
            img_path = Path(img_path)
            markdown.append(f"![Image {i+1}]({img_path})\n")
        markdown.append("\n")

    # Add tables
    if include_tables and "tables" in parsed_data and parsed_data["tables"]:
        markdown.append("## Tables\n")
        for i, table in enumerate(parsed_data["tables"]):
            try:
                markdown.append(f"### Table {i+1}\n")
                formatted_table = format_table(table)
                if not formatted_table:
                    markdown.append("*No table data could be extracted*\n")
                else:
                    markdown.append(formatted_table)
                markdown.append("\n")
            except Exception as e:
                logger.error(f"Error processing table {i+1}: {e}")
                markdown.append(f"*Error processing table {i+1}: {str(e)}*\n\n")

    # Add form elements
    if include_forms and "forms" in parsed_data and parsed_data["forms"]:
        markdown.append("## Form Elements\n")
        for i, form in enumerate(parsed_data["forms"]):
            markdown.append(f"### Form Element {i+1}\n")
            markdown.append(format_form_element(form))
            markdown.append("\n")

    # Add handwritten content
    if (
        include_handwritten
        and "handwritten_content" in parsed_data
        and parsed_data["handwritten_content"]
    ):
        markdown.append("## Handwritten Content\n")
        for i, content in enumerate(parsed_data["handwritten_content"]):
            if content:
                markdown.append(f"### Handwritten Text {i+1}\n")
                markdown.append(f"{content}\n\n")

    # Join all markdown sections
    markdown_text = "\n".join(markdown)

    # Write to file if output_file is provided
    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as f:
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
    paragraphs = text.split("\n\n")

    # Format each paragraph
    formatted_paragraphs = []
    for paragraph in paragraphs:
        # Clean up paragraph
        paragraph = paragraph.strip()
        if paragraph:
            formatted_paragraphs.append(paragraph)

    return "\n\n".join(formatted_paragraphs)


def format_table(table_data: Union[List[Dict[str, Any]], List[List[Any]]]) -> str:
    """
    Format table data as Markdown table.

    Args:
        table_data: Table data which can be:
                   - List of dictionaries (each dict represents a row)
                   - 2D list (list of lists)
                   - pandas DataFrame (handled by tabula)

    Returns:
        Markdown formatted table or error message
    """
    if not table_data:
        return "No table data available"

    try:
        # Handle pandas DataFrame (from tabula)
        if hasattr(table_data, "to_dict"):
            table_data = table_data.fillna("").to_dict("records")

        # Handle different table formats
        if (
            isinstance(table_data, list)
            and table_data
            and hasattr(table_data[0], "keys")
        ):
            # List of dictionaries format
            headers = list(table_data[0].keys())
            rows = []

            for row in table_data:
                rows.append([str(row.get(header, "")) for header in headers])

        elif (
            isinstance(table_data, list)
            and table_data
            and isinstance(table_data[0], (list, tuple))
        ):
            # 2D list format
            headers = [f"Column {i+1}" for i in range(len(table_data[0]))]
            rows = [[str(cell) for cell in row] for row in table_data]
        else:
            return "Unsupported table format"

        # Create markdown table
        markdown_table = "| " + " | ".join(map(str, headers)) + " |\n"
        markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"

        for row in rows:
            # Ensure the row has the same number of columns as headers
            row = (
                row + [""] * (len(headers) - len(row))
                if len(row) < len(headers)
                else row[: len(headers)]
            )
            markdown_table += "| " + " | ".join(map(str, row)) + " |\n"

        return markdown_table

    except Exception as e:
        logger.error(f"Error formatting table: {e}")
        return f"Error formatting table: {str(e)}"


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
    if "type" in form_data:
        markdown.append(f"**Type:** {form_data['type']}")

    # Add form element text
    if "text" in form_data:
        markdown.append(f"**Text:** {form_data['text']}")

    # Add form element position
    if "page" in form_data:
        markdown.append(f"**Page:** {form_data['page']}")

    # Add form element bounding box
    if "bbox" in form_data:
        bbox = form_data["bbox"]
        markdown.append(
            f"**Position:** (x1={bbox[0]:.2f}, y1={bbox[1]:.2f}, x2={bbox[2]:.2f}, y2={bbox[3]:.2f})"
        )

    return "\n".join(markdown)
