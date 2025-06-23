#!/usr/bin/env python3
"""
Test class for parsing PDFs using od-parse library.
Tests PDFs from the SAMPLE-Parser-Data folder and outputs JSON results.
"""

import os
import sys
import json
import glob
import unittest
from pathlib import Path
from typing import List, Dict, Any

# Add the od_parse module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from od_parse.main import parse_pdf
from od_parse.config import get_advanced_config


class TestPDFParsing(unittest.TestCase):
    """Test class for parsing PDFs and generating JSON output."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with PDF folder and configuration."""
        cls.pdf_folder = "/Users/praveen.soni/Desktop/Personal/Developer/SAMPLE-Parser-Data/"
        cls.output_folder = "test_results"
        
        # Create output folder
        os.makedirs(cls.output_folder, exist_ok=True)
        
        # Find all PDFs in the folder
        cls.pdf_files = glob.glob(os.path.join(cls.pdf_folder, "*.pdf"))
        
        print(f"\n🔍 Found {len(cls.pdf_files)} PDF files in {cls.pdf_folder}")
        for pdf_file in cls.pdf_files:
            print(f"  📄 {os.path.basename(pdf_file)}")
        
        # Configure advanced features
        cls.config = get_advanced_config()
        cls.config.enable_feature('quality_assessment', check_dependencies=False)
        cls.config.enable_feature('multilingual', check_dependencies=False)
    
    def test_pdf_folder_exists(self):
        """Test that the PDF folder exists and contains PDFs."""
        self.assertTrue(os.path.exists(self.pdf_folder), f"PDF folder does not exist: {self.pdf_folder}")
        self.assertGreater(len(self.pdf_files), 0, f"No PDF files found in {self.pdf_folder}")
        print(f"✅ PDF folder exists with {len(self.pdf_files)} PDF files")
    
    def test_parse_first_pdf_basic(self):
        """Test parsing the first PDF with basic settings."""
        if not self.pdf_files:
            self.skipTest("No PDF files found")
        
        pdf_file = self.pdf_files[0]
        pdf_name = os.path.basename(pdf_file)
        
        print(f"\n📖 Testing basic parsing of: {pdf_name}")
        
        try:
            # Parse with basic settings
            result = parse_pdf(
                file_path=pdf_file,
                output_format="json",
                use_deep_learning=False
            )
            
            # Validate result structure
            self.assertIsInstance(result, dict)
            self.assertIn('parsed_data', result)
            self.assertIn('metadata', result)
            self.assertIn('summary', result)
            
            # Save JSON output
            output_file = os.path.join(self.output_folder, f"basic_{pdf_name.replace('.pdf', '.json')}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Basic parsing successful")
            print(f"💾 Results saved to: {output_file}")
            
            # Print summary
            summary = result.get('summary', {})
            stats = summary.get('extraction_statistics', {})
            print(f"📊 Summary:")
            print(f"   Text length: {stats.get('text_length', 0)} characters")
            print(f"   Tables found: {stats.get('tables_extracted', 0)}")
            print(f"   Forms found: {stats.get('form_fields_extracted', 0)}")
            print(f"   Processing time: {summary.get('processing_time_seconds', 0):.2f}s")
            
        except Exception as e:
            self.fail(f"Basic parsing failed: {e}")
    
    def test_parse_first_pdf_advanced(self):
        """Test parsing the first PDF with advanced features."""
        if not self.pdf_files:
            self.skipTest("No PDF files found")
        
        pdf_file = self.pdf_files[0]
        pdf_name = os.path.basename(pdf_file)
        
        print(f"\n🚀 Testing advanced parsing of: {pdf_name}")
        
        try:
            # Parse with advanced settings
            result = parse_pdf(
                file_path=pdf_file,
                output_format="json",
                use_deep_learning=True,
                pipeline_type="full"
            )
            
            # Validate result structure
            self.assertIsInstance(result, dict)
            self.assertIn('parsed_data', result)
            self.assertIn('metadata', result)
            self.assertIn('summary', result)
            
            # Save JSON output
            output_file = os.path.join(self.output_folder, f"advanced_{pdf_name.replace('.pdf', '.json')}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Advanced parsing successful")
            print(f"💾 Results saved to: {output_file}")
            
            # Print enhanced summary
            summary = result.get('summary', {})
            stats = summary.get('extraction_statistics', {})
            print(f"📊 Enhanced Summary:")
            print(f"   Text length: {stats.get('text_length', 0)} characters")
            print(f"   Tables found: {stats.get('tables_extracted', 0)}")
            print(f"   Forms found: {stats.get('form_fields_extracted', 0)}")
            print(f"   Quality score: {summary.get('quality_score', 'N/A')}")
            print(f"   Detected language: {summary.get('detected_language', 'N/A')}")
            print(f"   Processing time: {summary.get('processing_time_seconds', 0):.2f}s")
            
        except Exception as e:
            self.fail(f"Advanced parsing failed: {e}")
    
    def test_parse_all_pdfs_summary(self):
        """Test parsing all PDFs and generate summary report."""
        if not self.pdf_files:
            self.skipTest("No PDF files found")
        
        print(f"\n📚 Testing all {len(self.pdf_files)} PDFs...")
        
        results_summary = []
        
        for i, pdf_file in enumerate(self.pdf_files, 1):
            pdf_name = os.path.basename(pdf_file)
            print(f"\n📄 [{i}/{len(self.pdf_files)}] Processing: {pdf_name}")
            
            try:
                # Parse each PDF
                result = parse_pdf(
                    file_path=pdf_file,
                    output_format="json",
                    use_deep_learning=False  # Use basic for speed
                )
                
                # Extract summary info
                summary = result.get('summary', {})
                stats = summary.get('extraction_statistics', {})
                
                pdf_summary = {
                    "filename": pdf_name,
                    "status": "success",
                    "file_size": summary.get('file_size', 0),
                    "page_count": summary.get('page_count', 'unknown'),
                    "text_length": stats.get('text_length', 0),
                    "tables_found": stats.get('tables_extracted', 0),
                    "forms_found": stats.get('form_fields_extracted', 0),
                    "processing_time": summary.get('processing_time_seconds', 0)
                }
                
                # Save individual result
                output_file = os.path.join(self.output_folder, f"batch_{pdf_name.replace('.pdf', '.json')}")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                print(f"   ✅ Success - {stats.get('text_length', 0)} chars, {stats.get('tables_extracted', 0)} tables")
                
            except Exception as e:
                pdf_summary = {
                    "filename": pdf_name,
                    "status": "error",
                    "error": str(e)
                }
                print(f"   ❌ Error: {e}")
            
            results_summary.append(pdf_summary)
        
        # Save batch summary
        summary_file = os.path.join(self.output_folder, "batch_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Batch Summary:")
        successful = sum(1 for r in results_summary if r['status'] == 'success')
        failed = len(results_summary) - successful
        print(f"   ✅ Successful: {successful}")
        print(f"   ❌ Failed: {failed}")
        print(f"   💾 Summary saved to: {summary_file}")
        
        # Validate at least one PDF was processed successfully
        self.assertGreater(successful, 0, "No PDFs were processed successfully")
    
    def test_json_output_validity(self):
        """Test that all JSON outputs are valid and can be parsed."""
        if not self.pdf_files:
            self.skipTest("No PDF files found")
        
        pdf_file = self.pdf_files[0]
        
        print(f"\n🔍 Testing JSON validity...")
        
        # Parse PDF
        result = parse_pdf(
            file_path=pdf_file,
            output_format="json",
            use_deep_learning=False
        )
        
        # Test JSON serialization
        try:
            json_string = json.dumps(result, ensure_ascii=False)
            print(f"✅ JSON serialization successful ({len(json_string)} characters)")
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")
        
        # Test JSON parsing
        try:
            parsed_back = json.loads(json_string)
            self.assertEqual(len(parsed_back), len(result))
            print(f"✅ JSON round-trip successful")
        except Exception as e:
            self.fail(f"JSON parsing failed: {e}")
        
        # Validate no NaN or invalid values
        json_str = json.dumps(result)
        self.assertNotIn('NaN', json_str, "JSON contains NaN values")
        self.assertNotIn('Infinity', json_str, "JSON contains Infinity values")
        print(f"✅ JSON contains no invalid values")


def main():
    """Run the PDF parsing tests."""
    print("🚀 od-parse PDF Testing Suite")
    print("=" * 50)
    
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "=" * 50)
    print("📁 Check the 'test_results' folder for JSON outputs")
    print("🎉 Testing complete!")


if __name__ == "__main__":
    main()
