#!/usr/bin/env python3
"""
Quick Start Example - Get started with od-parse in 60 seconds!

This is the simplest possible example to get you started.
"""

from od_parse import parse_pdf

# Step 1: Set your API key (or use environment variable)
# export GOOGLE_API_KEY="your-api-key-here"

# Step 2: Parse a PDF
result = parse_pdf(
    "your_document.pdf",  # Replace with your PDF path
    llm_model="gemini-2.0-flash",  # Google Gemini model
    api_keys={"google": "YOUR_GOOGLE_API_KEY"},  # Or use env var
    output_format="json"
)

# Step 3: Access the results
print(f"✅ Parsed {result['metadata']['num_pages']} pages")
print(f"✅ Extracted {len(result['parsed_data']['text'])} characters")
print(f"✅ Found {len(result['parsed_data']['tables'])} tables")
print(f"✅ Found {len(result['parsed_data']['forms'])} forms")

# That's it! You're ready to use od-parse 🎉

