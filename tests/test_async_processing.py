"""
Tests for Async Processing.
"""

import os
import sys
import unittest
import asyncio
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to sys.path to import od_parse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from od_parse.config import get_advanced_config
from od_parse.async_processing import (
    AsyncDocumentProcessor,
    ProgressTracker,
    process_files_async,
    process_large_file_async
)


class TestProgressTracker(unittest.TestCase):
    """Test cases for Progress Tracker."""
    
    def test_progress_tracker_initialization(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(10, "Test operation")
        
        self.assertEqual(tracker.total_items, 10)
        self.assertEqual(tracker.completed_items, 0)
        self.assertEqual(tracker.failed_items, 0)
        self.assertEqual(tracker.description, "Test operation")
    
    def test_progress_tracker_update(self):
        """Test progress tracker updates."""
        tracker = ProgressTracker(10, "Test operation")
        
        # Update progress
        tracker.update(completed=3, failed=1)
        
        self.assertEqual(tracker.completed_items, 3)
        self.assertEqual(tracker.failed_items, 1)
        
        # Get progress data
        progress_data = tracker.get_progress_data()
        
        self.assertEqual(progress_data["total_items"], 10)
        self.assertEqual(progress_data["completed_items"], 3)
        self.assertEqual(progress_data["failed_items"], 1)
        self.assertEqual(progress_data["processed_items"], 4)
        self.assertEqual(progress_data["progress_percentage"], 40.0)
    
    def test_progress_tracker_callbacks(self):
        """Test progress tracker callbacks."""
        tracker = ProgressTracker(5, "Test operation")
        
        callback_data = []
        
        def test_callback(data):
            callback_data.append(data)
        
        tracker.add_callback(test_callback)
        tracker.update(completed=2)
        
        # Check that callback was called
        self.assertEqual(len(callback_data), 1)
        self.assertEqual(callback_data[0]["completed_items"], 2)


class TestAsyncDocumentProcessor(unittest.TestCase):
    """Test cases for Async Document Processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
        self.processor = AsyncDocumentProcessor(max_workers=2)
        
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
        for i in range(3):
            test_file = Path(self.temp_dir) / f"test_{i}.txt"
            test_file.write_text(f"Test content {i}")
            self.test_files.append(test_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_async_processor_initialization(self):
        """Test async processor initialization."""
        processor = AsyncDocumentProcessor()
        
        self.assertIsInstance(processor, AsyncDocumentProcessor)
        self.assertGreater(processor.max_workers, 0)
        self.assertFalse(processor.use_process_pool)  # Default
    
    def test_async_processor_with_process_pool(self):
        """Test async processor with process pool."""
        processor = AsyncDocumentProcessor(use_process_pool=True, max_workers=2)
        
        self.assertTrue(processor.use_process_pool)
        self.assertEqual(processor.max_workers, 2)
    
    def test_processing_stats(self):
        """Test processing statistics."""
        stats = self.processor.get_processing_stats()
        
        self.assertIn("max_workers", stats)
        self.assertIn("use_process_pool", stats)
        self.assertIn("chunk_size_mb", stats)
        self.assertIn("aiofiles_available", stats)
        self.assertIn("cpu_count", stats)
    
    def test_async_file_processing(self):
        """Test asynchronous file processing."""
        async def run_test():
            def simple_processor(file_path):
                """Simple test processor function."""
                with open(file_path, 'r') as f:
                    content = f.read()
                return {"content": content, "length": len(content)}
            
            results = await self.processor.process_files_async(
                self.test_files, simple_processor
            )
            
            # Check results
            self.assertEqual(len(results), 3)
            
            for result in results:
                self.assertIn("file_path", result)
                self.assertIn("status", result)
                self.assertEqual(result["status"], "success")
                self.assertIn("result", result)
        
        # Run the async test
        asyncio.run(run_test())
    
    def test_async_file_processing_with_progress(self):
        """Test async file processing with progress tracking."""
        async def run_test():
            progress_updates = []
            
            def progress_callback(data):
                progress_updates.append(data)
            
            def simple_processor(file_path):
                time.sleep(0.1)  # Simulate processing time
                return {"processed": str(file_path)}
            
            results = await self.processor.process_files_async(
                self.test_files, simple_processor, progress_callback
            )
            
            # Check that progress was tracked
            self.assertGreater(len(progress_updates), 0)
            self.assertEqual(len(results), 3)
        
        asyncio.run(run_test())
    
    def test_async_file_processing_with_errors(self):
        """Test async file processing with error handling."""
        async def run_test():
            def error_processor(file_path):
                """Processor that always raises an error."""
                raise ValueError("Test error")
            
            results = await self.processor.process_files_async(
                self.test_files, error_processor
            )
            
            # Check that errors were handled
            self.assertEqual(len(results), 3)
            
            for result in results:
                self.assertEqual(result["status"], "error")
                self.assertIn("error", result)
        
        asyncio.run(run_test())
    
    def test_large_file_processing(self):
        """Test large file processing."""
        async def run_test():
            # Create a larger test file
            large_file = Path(self.temp_dir) / "large_test.txt"
            large_content = "Large file content " * 1000
            large_file.write_text(large_content)
            
            def large_file_processor(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                return {"size": len(content)}
            
            result = await self.processor.process_large_file_async(
                large_file, large_file_processor
            )
            
            # Check result
            self.assertEqual(result["status"], "success")
            self.assertIn("result", result)
            self.assertIn("processing_time", result)
            self.assertIn("file_size", result)
        
        asyncio.run(run_test())
    
    def test_batch_processing_with_retry(self):
        """Test batch processing with retry logic."""
        async def run_test():
            items = ["item1", "item2"]  # Reduce items for faster test

            def simple_processor(item):
                """Simple processor that always succeeds."""
                return {"processed": item}

            results = await self.processor.batch_process_with_retry(
                items, simple_processor, max_retries=1, retry_delay=0.01
            )

            # Check results
            self.assertEqual(len(results), 2)

            for result in results:
                self.assertEqual(result["status"], "success")
                self.assertIn("attempts", result)

        asyncio.run(run_test())
    
    def test_empty_file_list_handling(self):
        """Test handling of empty file lists."""
        async def run_test():
            def dummy_processor(file_path):
                return {"processed": True}
            
            results = await self.processor.process_files_async([], dummy_processor)
            
            self.assertEqual(len(results), 0)
        
        asyncio.run(run_test())


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
        for i in range(2):
            test_file = Path(self.temp_dir) / f"test_{i}.txt"
            test_file.write_text(f"Test content {i}")
            self.test_files.append(test_file)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_process_files_async_convenience(self):
        """Test process_files_async convenience function."""
        async def run_test():
            def simple_processor(file_path):
                return {"file": str(file_path)}
            
            results = await process_files_async(self.test_files, simple_processor)
            
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertEqual(result["status"], "success")
        
        asyncio.run(run_test())
    
    def test_process_large_file_async_convenience(self):
        """Test process_large_file_async convenience function."""
        async def run_test():
            def simple_processor(file_path):
                return {"processed": True}
            
            result = await process_large_file_async(self.test_files[0], simple_processor)
            
            self.assertEqual(result["status"], "success")
        
        asyncio.run(run_test())


class TestAsyncProcessingConfiguration(unittest.TestCase):
    """Test async processing configuration management."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_advanced_config()
    
    def test_async_processing_feature_configuration(self):
        """Test async processing feature can be enabled/disabled."""
        # Test enabling
        result = self.config.enable_feature("async_processing", check_dependencies=False)
        self.assertTrue(result)
        self.assertTrue(self.config.is_feature_enabled("async_processing"))
        
        # Test disabling
        result = self.config.disable_feature("async_processing")
        self.assertTrue(result)
        self.assertFalse(self.config.is_feature_enabled("async_processing"))
    
    def test_async_processing_feature_info(self):
        """Test async processing feature information."""
        info = self.config.get_feature_info("async_processing")
        
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "Async Processing")
        self.assertIn("dependencies", info)
        self.assertIn("aiofiles", info["dependencies"])


if __name__ == "__main__":
    unittest.main()
