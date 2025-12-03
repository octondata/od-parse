#!/usr/bin/env python3
"""
Excel Processing with DuckDB Example

This example demonstrates how to use od-parse to process Excel files
using DuckDB for efficient data handling. Supports both JSON and Markdown output.

Prerequisites:
    pip install od-parse[excel]
    # or
    pip install duckdb openpyxl pandas

Usage:
    python excel_processing.py
"""

import json
import os
import tempfile


def create_sample_excel(filepath: str) -> bool:
    """Create a sample Excel file for demonstration."""
    try:
        import pandas as pd
    except ImportError:
        print("pandas required for this example: pip install pandas")
        return False

    # Sales data
    sales_df = pd.DataFrame(
        {
            "Date": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
            "Product": ["Widget A", "Widget B", "Gadget X", "Widget A"],
            "Quantity": [100, 75, 50, 120],
            "Unit Price": [9.99, 14.99, 29.99, 9.99],
            "Total": [999.00, 1124.25, 1499.50, 1198.80],
            "Region": ["North", "South", "East", "West"],
        }
    )

    # Inventory data
    inventory_df = pd.DataFrame(
        {
            "SKU": ["WA-001", "WB-002", "GX-003", "GY-004"],
            "Product Name": ["Widget A", "Widget B", "Gadget X", "Gadget Y"],
            "Stock": [500, 250, 100, 75],
            "Reorder Level": [100, 50, 25, 20],
            "Warehouse": ["Main", "Main", "Secondary", "Secondary"],
        }
    )

    # Write to Excel with multiple sheets
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        sales_df.to_excel(writer, sheet_name="Sales", index=False)
        inventory_df.to_excel(writer, sheet_name="Inventory", index=False)

    print(f"Created sample Excel file: {filepath}")
    return True


def example_basic_json():
    """Basic example: Parse Excel to JSON."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Excel to JSON")
    print("=" * 60)

    from od_parse import parse_excel, EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        # Parse to JSON
        result = parse_excel(sample_file, output_format="json")

        print(f"\nFile: {result['filename']}")
        print(f"Sheets: {result['sheet_count']}")

        for sheet_name, sheet_data in result["sheets"].items():
            print(f"\n  Sheet: {sheet_name}")
            print(f"  Rows: {sheet_data['row_count']}")
            print(f"  Columns: {sheet_data['columns']}")
            print(f"  Sample data: {sheet_data['data'][:2]}")

    finally:
        os.unlink(sample_file)


def example_basic_markdown():
    """Basic example: Parse Excel to Markdown."""
    print("\n" + "=" * 60)
    print("Example 2: Basic Excel to Markdown")
    print("=" * 60)

    from od_parse import parse_excel, EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        # Parse to Markdown
        markdown = parse_excel(sample_file, output_format="markdown")

        print("\nGenerated Markdown:")
        print("-" * 40)
        print(markdown)

    finally:
        os.unlink(sample_file)


def example_specific_sheets():
    """Example: Process only specific sheets."""
    print("\n" + "=" * 60)
    print("Example 3: Process Specific Sheets Only")
    print("=" * 60)

    from od_parse import parse_excel, EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        # Parse only the Sales sheet
        result = parse_excel(
            sample_file,
            output_format="json",
            sheets=["Sales"],
        )

        print(f"\nProcessed sheets: {list(result['sheets'].keys())}")
        print(f"Sales data rows: {result['sheets']['Sales']['row_count']}")

    finally:
        os.unlink(sample_file)


def example_save_to_file():
    """Example: Save output to file."""
    print("\n" + "=" * 60)
    print("Example 4: Save to File")
    print("=" * 60)

    from od_parse import excel_to_json, excel_to_markdown, EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        # Save as JSON
        json_output = sample_file.replace(".xlsx", ".json")
        excel_to_json(sample_file, output_file=json_output)
        print(f"Saved JSON to: {json_output}")

        # Save as Markdown
        md_output = sample_file.replace(".xlsx", ".md")
        excel_to_markdown(sample_file, output_file=md_output)
        print(f"Saved Markdown to: {md_output}")

        # Clean up output files
        os.unlink(json_output)
        os.unlink(md_output)

    finally:
        os.unlink(sample_file)


def example_sql_query():
    """Example: Run SQL queries on Excel data."""
    print("\n" + "=" * 60)
    print("Example 5: SQL Queries on Excel Data")
    print("=" * 60)

    from od_parse import EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    from od_parse.excel import ExcelProcessor

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        with ExcelProcessor() as processor:
            # Query: Get high-value sales (Total > 1100)
            print("\n1. High-value sales (Total > 1100):")
            result = processor.query(
                sample_file, "SELECT Product, Quantity, Total FROM Sales WHERE Total > 1100"
            )
            for row in result:
                print(f"   {row}")

            # Query: Aggregate by region
            print("\n2. Sales by Region:")
            result = processor.query(
                sample_file,
                """
                SELECT Region, SUM(Total) as TotalSales, COUNT(*) as Orders
                FROM Sales
                GROUP BY Region
                ORDER BY TotalSales DESC
                """,
            )
            for row in result:
                print(f"   {row}")

            # Query: Low stock items
            print("\n3. Low Stock Items (below reorder level + 50):")
            result = processor.query(
                sample_file,
                """
                SELECT "Product Name", Stock, "Reorder Level"
                FROM Inventory
                WHERE Stock < "Reorder Level" + 50
                """,
            )
            for row in result:
                print(f"   {row}")

    finally:
        os.unlink(sample_file)


def example_processor_config():
    """Example: Advanced processor configuration."""
    print("\n" + "=" * 60)
    print("Example 6: Advanced Configuration")
    print("=" * 60)

    from od_parse import EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    from od_parse.excel import ExcelProcessor, ExcelProcessorConfig, OutputFormat

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        # Create custom configuration
        config = ExcelProcessorConfig(
            output_format=OutputFormat.JSON,
            sheets=["Sales"],
            first_row_as_header=True,
            skip_empty_rows=True,
            max_rows=2,  # Limit to first 2 rows
            column_mapping={
                "Unit Price": "price",
                "Total": "amount",
            },
        )

        with ExcelProcessor(config) as processor:
            result = processor.parse(sample_file)

        print("\nProcessed with custom config:")
        print(f"  Sheets: {list(result['sheets'].keys())}")
        print(f"  Rows (limited to 2): {result['sheets']['Sales']['row_count']}")
        print(f"  Columns (renamed): {result['sheets']['Sales']['columns']}")
        print(f"  Data: {json.dumps(result['sheets']['Sales']['data'], indent=4)}")

    finally:
        os.unlink(sample_file)


def example_sheet_info():
    """Example: Get information about sheets."""
    print("\n" + "=" * 60)
    print("Example 7: Get Sheet Information")
    print("=" * 60)

    from od_parse import EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("Excel processing not available.")
        print("Install with: pip install od-parse[excel]")
        return

    from od_parse.excel import ExcelProcessor

    # Create sample file
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
        sample_file = f.name

    try:
        if not create_sample_excel(sample_file):
            return

        with ExcelProcessor() as processor:
            info = processor.get_sheet_info(sample_file)

        print("\nSheet Information:")
        for sheet_name, sheet_info in info.items():
            print(f"\n  {sheet_name}:")
            print(f"    Rows: {sheet_info['row_count']}")
            print(f"    Columns: {sheet_info['column_count']}")
            print(f"    Column Names: {sheet_info['columns']}")

    finally:
        os.unlink(sample_file)


def main():
    """Run all examples."""
    print("=" * 60)
    print("od-parse Excel Processing Examples (DuckDB)")
    print("=" * 60)

    # Check availability
    from od_parse import EXCEL_AVAILABLE

    if not EXCEL_AVAILABLE:
        print("\n⚠️  Excel processing is not available.")
        print("Install dependencies with: pip install od-parse[excel]")
        print("\nThis installs:")
        print("  - duckdb: Fast in-process SQL database")
        print("  - openpyxl: Excel file reading/writing")
        print("  - pandas: Data manipulation")
        return

    print("\n✓ Excel processing is available!")

    # Run examples
    example_basic_json()
    example_basic_markdown()
    example_specific_sheets()
    example_save_to_file()
    example_sql_query()
    example_processor_config()
    example_sheet_info()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
