"""
Excel processing module using DuckDB for efficient data handling.

This module provides functionality to parse Excel files (.xlsx, .xls)
and output data in JSON or Markdown format.
"""

from od_parse.excel.duckdb_processor import (
    ExcelProcessor,
    ExcelProcessorConfig,
    OutputFormat,
    parse_excel,
    excel_to_json,
    excel_to_markdown,
    check_duckdb_available,
)

__all__ = [
    "ExcelProcessor",
    "ExcelProcessorConfig",
    "OutputFormat",
    "parse_excel",
    "excel_to_json",
    "excel_to_markdown",
    "check_duckdb_available",
]
