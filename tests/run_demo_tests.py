#!/usr/bin/env python3
"""
Demo test script to parse the provided PDF and Excel files.
Outputs results to the test_output directory.
"""

import json
import os
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("od-parse Demo Test - PDF and Excel Processing")
print("=" * 60)


# Test 1: Parse PDF
print("\n" + "-" * 60)
print("TEST 1: PDF Parsing")
print("-" * 60)

pdf_file = Path("W2-2025 (2).pdf")
if pdf_file.exists():
    print(f"Processing: {pdf_file}")
    
    try:
        from od_parse import parse_pdf, convert_to_markdown
        
        # Parse PDF
        result = parse_pdf(str(pdf_file))
        
        # Save JSON output
        json_output = OUTPUT_DIR / "pdf_output.json"
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✓ JSON saved to: {json_output}")
        
        # Save Markdown output
        md_output = OUTPUT_DIR / "pdf_output.md"
        markdown = convert_to_markdown(result)
        with open(md_output, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"✓ Markdown saved to: {md_output}")
        
        # Print summary
        print(f"\nPDF Summary:")
        print(f"  - Text length: {len(result.get('text', ''))} characters")
        print(f"  - Tables found: {len(result.get('tables', []))}")
        print(f"  - Images found: {len(result.get('images', []))}")
        print(f"  - Forms found: {len(result.get('forms', []))}")
        
    except Exception as e:
        print(f"✗ Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ PDF file not found: {pdf_file}")


# Test 2: Parse Excel
print("\n" + "-" * 60)
print("TEST 2: Excel Parsing (DuckDB)")
print("-" * 60)

excel_file = Path("parser_excel_demo.xlsx")
if excel_file.exists():
    print(f"Processing: {excel_file}")
    
    try:
        from od_parse import EXCEL_AVAILABLE
        
        if not EXCEL_AVAILABLE:
            print("✗ Excel processing not available. Install with: pip install od-parse[excel]")
        else:
            from od_parse import parse_excel, excel_to_json, excel_to_markdown
            
            # Parse to JSON
            json_result = excel_to_json(
                str(excel_file),
                output_file=str(OUTPUT_DIR / "excel_output.json")
            )
            print(f"✓ JSON saved to: {OUTPUT_DIR / 'excel_output.json'}")
            
            # Parse to Markdown
            md_result = excel_to_markdown(
                str(excel_file),
                output_file=str(OUTPUT_DIR / "excel_output.md")
            )
            print(f"✓ Markdown saved to: {OUTPUT_DIR / 'excel_output.md'}")
            
            # Print summary
            print(f"\nExcel Summary:")
            print(f"  - Filename: {json_result['filename']}")
            print(f"  - Sheets: {json_result['sheet_count']}")
            for sheet_name, sheet_data in json_result['sheets'].items():
                print(f"    - {sheet_name}: {sheet_data['row_count']} rows, {len(sheet_data['columns'])} columns")
                print(f"      Columns: {sheet_data['columns']}")
            
    except Exception as e:
        print(f"✗ Error parsing Excel: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"✗ Excel file not found: {excel_file}")


# Test 3: Excel SQL Query (if available)
print("\n" + "-" * 60)
print("TEST 3: Excel SQL Query")
print("-" * 60)

if excel_file.exists():
    try:
        from od_parse import EXCEL_AVAILABLE
        
        if EXCEL_AVAILABLE:
            from od_parse.excel import ExcelProcessor
            
            with ExcelProcessor() as processor:
                # Get sheet info first
                info = processor.get_sheet_info(str(excel_file))
                
                # Try to query each sheet
                for sheet_name in info.keys():
                    safe_name = sheet_name.replace(" ", "_").replace("-", "_")
                    print(f"\nQuerying sheet: {sheet_name}")
                    
                    try:
                        # Get first 5 rows
                        result = processor.query(
                            str(excel_file),
                            f'SELECT * FROM "{safe_name}" LIMIT 5'
                        )
                        
                        # Save query result
                        query_output = OUTPUT_DIR / f"excel_query_{safe_name}.json"
                        with open(query_output, "w", encoding="utf-8") as f:
                            json.dump(result, f, indent=2, default=str)
                        print(f"✓ Query result saved to: {query_output}")
                        print(f"  Rows returned: {len(result)}")
                        
                    except Exception as e:
                        print(f"  Query error: {e}")
                        
    except Exception as e:
        print(f"✗ Error with SQL query: {e}")


print("\n" + "=" * 60)
print("Demo Complete!")
print(f"Output files are in: {OUTPUT_DIR.absolute()}")
print("=" * 60)

# List output files
print("\nGenerated files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    size = f.stat().st_size
    print(f"  - {f.name} ({size:,} bytes)")

