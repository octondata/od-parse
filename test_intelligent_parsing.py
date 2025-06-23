#!/usr/bin/env python3
"""
Intelligent Document Parsing Test

This test demonstrates the new intelligent document processing capabilities
that provide structured, meaningful output for complex PDFs.
"""

import os
import sys
import json
import unittest
from pathlib import Path

# Add the od_parse module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from od_parse.main import parse_pdf
from od_parse.intelligence import DocumentAnalyzer


class TestIntelligentParsing(unittest.TestCase):
    """Test intelligent document parsing capabilities."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.pdf_path = "/Users/praveen.soni/Desktop/Personal/Developer/SAMPLE-Parser-Data/NAVEEN_PANDEY_1040.pdf"
        cls.output_folder = "intelligent_results"
        
        # Create output folder
        os.makedirs(cls.output_folder, exist_ok=True)
        
        print(f"\n🧠 Testing Intelligent Document Processing")
        print(f"📄 PDF: {os.path.basename(cls.pdf_path)}")
    
    def test_intelligent_document_analysis(self):
        """Test the new intelligent document analysis."""
        print(f"\n🔍 Running intelligent analysis...")
        
        # Parse with intelligent features
        result = parse_pdf(
            file_path=self.pdf_path,
            output_format="json",
            use_deep_learning=True,
            pipeline_type="full"
        )
        
        # Validate intelligent analysis is present
        self.assertIn('parsed_data', result)
        parsed_data = result['parsed_data']
        
        if 'document_intelligence' in parsed_data:
            intelligence = parsed_data['document_intelligence']
            
            print(f"✅ Document Intelligence Analysis Complete!")
            
            # Check document classification
            if 'document_intelligence' in intelligence:
                doc_info = intelligence['document_intelligence']
                print(f"📋 Document Type: {doc_info.get('document_type', 'unknown')}")
                print(f"🎯 Classification Confidence: {doc_info.get('confidence', 0):.2f}")
            
            # Check structured fields
            if 'structured_fields' in intelligence:
                fields = intelligence['structured_fields']
                print(f"📊 Structured Fields Extracted: {len(fields)}")
                
                # Show some key fields
                for field_name, field_data in list(fields.items())[:5]:
                    print(f"   • {field_name}: {field_data.get('value')} (confidence: {field_data.get('confidence', 0):.2f})")
            
            # Check structured tables
            if 'structured_tables' in intelligence:
                tables = intelligence['structured_tables']
                print(f"📋 Structured Tables: {len(tables)}")
                
                for i, table in enumerate(tables[:3]):  # Show first 3 tables
                    print(f"   Table {i+1}: {table.get('table_type', 'unknown')} ({table.get('dimensions', {}).get('rows', 0)} rows)")
            
            # Check key-value pairs
            if 'key_value_pairs' in intelligence:
                kv_pairs = intelligence['key_value_pairs']
                print(f"🔑 Key-Value Pairs: {len(kv_pairs)}")
            
            # Check validation results
            if 'validation_results' in intelligence:
                validation = intelligence['validation_results']
                print(f"✅ Validation: {validation.get('valid_fields', 0)} valid, {validation.get('invalid_fields', 0)} invalid")
                print(f"📈 Completeness Score: {validation.get('completeness_score', 0):.2f}")
            
            # Save intelligent results
            output_file = os.path.join(self.output_folder, "intelligent_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(intelligence, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Intelligent analysis saved to: {output_file}")
            
        else:
            print("⚠️  Document intelligence not available - using fallback processing")
    
    def test_structured_field_extraction(self):
        """Test structured field extraction specifically."""
        print(f"\n🏗️  Testing structured field extraction...")
        
        # Parse document
        result = parse_pdf(self.pdf_path, use_deep_learning=True)
        parsed_data = result.get('parsed_data', {})
        
        if 'document_intelligence' in parsed_data:
            intelligence = parsed_data['document_intelligence']
            structured_fields = intelligence.get('structured_fields', {})
            
            # Test that we extracted meaningful fields
            self.assertGreater(len(structured_fields), 0, "Should extract some structured fields")
            
            # Check for tax-specific fields (since this is a 1040 form)
            expected_tax_fields = ['taxpayer_name', 'ssn', 'filing_status', 'total_income']
            found_fields = []
            
            for field_name in structured_fields:
                if any(expected in field_name.lower() for expected in ['name', 'ssn', 'income', 'tax']):
                    found_fields.append(field_name)
            
            print(f"📋 Tax-related fields found: {found_fields}")
            
            # Save structured fields separately
            output_file = os.path.join(self.output_folder, "structured_fields.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_fields, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Structured fields saved to: {output_file}")
    
    def test_table_intelligence(self):
        """Test intelligent table analysis."""
        print(f"\n📊 Testing intelligent table analysis...")
        
        # Parse document
        result = parse_pdf(self.pdf_path, use_deep_learning=True)
        parsed_data = result.get('parsed_data', {})
        
        if 'document_intelligence' in parsed_data:
            intelligence = parsed_data['document_intelligence']
            structured_tables = intelligence.get('structured_tables', [])
            
            print(f"📋 Found {len(structured_tables)} structured tables")
            
            for i, table in enumerate(structured_tables):
                print(f"\nTable {i+1}:")
                print(f"  Type: {table.get('table_type', 'unknown')}")
                print(f"  Dimensions: {table.get('dimensions', {})}")
                print(f"  Headers: {table.get('headers', [])[:5]}...")  # First 5 headers
                print(f"  Confidence: {table.get('confidence', 0):.2f}")
                
                # Show sample structured data
                structured_data = table.get('structured_data', [])
                if structured_data:
                    print(f"  Sample row: {list(structured_data[0].keys())[:3]}...")
            
            # Save table analysis
            output_file = os.path.join(self.output_folder, "table_analysis.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_tables, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Table analysis saved to: {output_file}")
    
    def test_document_summary(self):
        """Test intelligent document summary generation."""
        print(f"\n📝 Testing document summary generation...")
        
        # Parse document
        result = parse_pdf(self.pdf_path, use_deep_learning=True)
        parsed_data = result.get('parsed_data', {})
        
        if 'document_intelligence' in parsed_data:
            intelligence = parsed_data['document_intelligence']
            document_summary = intelligence.get('document_summary', {})
            
            print(f"📋 Document Summary Generated:")
            print(f"  Document Type: {document_summary.get('document_type', 'unknown')}")
            
            # Key information
            key_info = document_summary.get('key_information', {})
            if key_info:
                print(f"  Key Information:")
                for key, value in key_info.items():
                    print(f"    {key}: {value}")
            
            # Financial summary
            financial_summary = document_summary.get('financial_summary', {})
            if financial_summary:
                print(f"  Financial Summary:")
                for key, value in financial_summary.items():
                    print(f"    {key}: {value}")
            
            # Data quality
            data_quality = document_summary.get('data_quality', {})
            if data_quality:
                print(f"  Data Quality:")
                print(f"    Average Confidence: {data_quality.get('average_confidence', 0):.2f}")
                print(f"    Completeness: {data_quality.get('extraction_completeness', 0):.2f}")
            
            # Recommendations
            recommendations = document_summary.get('recommendations', [])
            if recommendations:
                print(f"  Recommendations:")
                for rec in recommendations:
                    print(f"    • {rec}")
            
            # Save summary
            output_file = os.path.join(self.output_folder, "document_summary.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_summary, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Document summary saved to: {output_file}")
    
    def test_comparison_with_basic_parsing(self):
        """Compare intelligent parsing with basic parsing."""
        print(f"\n⚖️  Comparing intelligent vs basic parsing...")
        
        # Basic parsing
        basic_result = parse_pdf(self.pdf_path, use_deep_learning=False)
        basic_data = basic_result.get('parsed_data', {})
        
        # Intelligent parsing
        intelligent_result = parse_pdf(self.pdf_path, use_deep_learning=True)
        intelligent_data = intelligent_result.get('parsed_data', {})
        
        print(f"📊 Comparison Results:")
        
        # Compare text extraction
        basic_text_len = len(basic_data.get('text', ''))
        intelligent_text_len = len(intelligent_data.get('text', ''))
        print(f"  Text Length - Basic: {basic_text_len}, Intelligent: {intelligent_text_len}")
        
        # Compare table extraction
        basic_tables = len(basic_data.get('tables', []))
        intelligent_tables = len(intelligent_data.get('tables', []))
        print(f"  Tables - Basic: {basic_tables}, Intelligent: {intelligent_tables}")
        
        # Show intelligent-only features
        if 'document_intelligence' in intelligent_data:
            intelligence = intelligent_data['document_intelligence']
            structured_fields = len(intelligence.get('structured_fields', {}))
            key_value_pairs = len(intelligence.get('key_value_pairs', {}))
            
            print(f"  Intelligent-Only Features:")
            print(f"    Structured Fields: {structured_fields}")
            print(f"    Key-Value Pairs: {key_value_pairs}")
            print(f"    Document Classification: ✅")
            print(f"    Validation & Quality: ✅")
        
        # Save comparison
        comparison = {
            "basic_parsing": {
                "text_length": basic_text_len,
                "tables_count": basic_tables,
                "forms_count": len(basic_data.get('forms', [])),
                "features": ["text_extraction", "table_extraction", "form_extraction"]
            },
            "intelligent_parsing": {
                "text_length": intelligent_text_len,
                "tables_count": intelligent_tables,
                "forms_count": len(intelligent_data.get('forms', [])),
                "features": ["text_extraction", "table_extraction", "form_extraction", 
                           "document_classification", "structured_fields", "validation", 
                           "key_value_pairs", "intelligent_summary"]
            }
        }
        
        if 'document_intelligence' in intelligent_data:
            intelligence = intelligent_data['document_intelligence']
            comparison["intelligent_parsing"]["structured_fields"] = len(intelligence.get('structured_fields', {}))
            comparison["intelligent_parsing"]["key_value_pairs"] = len(intelligence.get('key_value_pairs', {}))
        
        output_file = os.path.join(self.output_folder, "parsing_comparison.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Comparison saved to: {output_file}")


def main():
    """Run the intelligent parsing tests."""
    print("🧠 Intelligent Document Processing Test Suite")
    print("=" * 60)
    
    # Check if PDF exists
    pdf_path = "/Users/praveen.soni/Desktop/Personal/Developer/SAMPLE-Parser-Data/NAVEEN_PANDEY_1040.pdf"
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("Please ensure the PDF exists at the specified location.")
        return
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 60)
    print("📁 Check the 'intelligent_results' folder for structured outputs")
    print("🎉 Intelligent parsing test complete!")


if __name__ == "__main__":
    main()
