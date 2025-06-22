"""
Async Processing Module

This module provides asynchronous processing capabilities for large files
and batch operations, with progress tracking and concurrent processing support.
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable, AsyncGenerator
import logging

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """Progress tracking for async operations."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description of the operation
        """
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.description = description
        self.start_time = time.time()
        self.callbacks = []
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a progress callback function."""
        self.callbacks.append(callback)
    
    def update(self, completed: int = 1, failed: int = 0):
        """Update progress counters."""
        self.completed_items += completed
        self.failed_items += failed
        
        # Notify callbacks
        progress_data = self.get_progress_data()
        for callback in self.callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def get_progress_data(self) -> Dict[str, Any]:
        """Get current progress data."""
        elapsed_time = time.time() - self.start_time
        processed_items = self.completed_items + self.failed_items
        
        if processed_items > 0:
            items_per_second = processed_items / elapsed_time
            eta = (self.total_items - processed_items) / items_per_second if items_per_second > 0 else 0
        else:
            items_per_second = 0
            eta = 0
        
        return {
            "description": self.description,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "processed_items": processed_items,
            "progress_percentage": (processed_items / self.total_items * 100) if self.total_items > 0 else 0,
            "elapsed_time": elapsed_time,
            "items_per_second": items_per_second,
            "eta_seconds": eta,
            "is_complete": processed_items >= self.total_items
        }


class AsyncDocumentProcessor:
    """
    Asynchronous document processing engine.
    
    This class provides async processing capabilities for document parsing,
    with support for concurrent processing, progress tracking, and batch operations.
    """
    
    def __init__(self, 
                 max_workers: Optional[int] = None,
                 use_process_pool: bool = False,
                 chunk_size: int = 10):
        """
        Initialize async processor.
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_process_pool: Whether to use process pool instead of thread pool
            chunk_size: Number of items to process in each chunk
        """
        self.logger = get_logger(__name__)
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_process_pool = use_process_pool
        self.chunk_size = chunk_size
        self._aiofiles_available = False
        
        # Check dependencies
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check for async processing dependencies."""
        config = get_advanced_config()
        
        if not config.is_feature_enabled("async_processing"):
            self.logger.info("Async processing feature is disabled. Use config.enable_feature('async_processing') to enable.")
            return
        
        try:
            import aiofiles
            self._aiofiles_available = True
        except ImportError:
            self.logger.warning("aiofiles not available. File I/O will be synchronous.")
    
    async def process_files_async(
        self,
        file_paths: List[Union[str, Path]],
        processor_func: Callable,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files asynchronously.
        
        Args:
            file_paths: List of file paths to process
            processor_func: Function to process each file
            progress_callback: Optional progress callback function
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processing results
        """
        if not file_paths:
            return []
        
        # Initialize progress tracker
        progress = ProgressTracker(len(file_paths), "Processing files")
        if progress_callback:
            progress.add_callback(progress_callback)
        
        # Process files in chunks
        results = []
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_file(file_path: Union[str, Path]) -> Dict[str, Any]:
            """Process a single file with semaphore control."""
            async with semaphore:
                try:
                    # Run processor function in thread pool
                    loop = asyncio.get_event_loop()
                    if self.use_process_pool:
                        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                            result = await loop.run_in_executor(
                                executor, processor_func, file_path, **kwargs
                            )
                    else:
                        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                            result = await loop.run_in_executor(
                                executor, processor_func, file_path, **kwargs
                            )
                    
                    progress.update(completed=1)
                    return {
                        "file_path": str(file_path),
                        "status": "success",
                        "result": result,
                        "processing_time": time.time()
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {e}")
                    progress.update(failed=1)
                    return {
                        "file_path": str(file_path),
                        "status": "error",
                        "error": str(e),
                        "processing_time": time.time()
                    }
        
        # Create tasks for all files
        tasks = [process_single_file(file_path) for file_path in file_paths]
        
        # Process tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "file_path": str(file_paths[i]),
                    "status": "error",
                    "error": str(result),
                    "processing_time": time.time()
                })
            else:
                final_results.append(result)
        
        return final_results
    
    async def process_large_file_async(
        self,
        file_path: Union[str, Path],
        processor_func: Callable,
        chunk_processor: Optional[Callable] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a large file asynchronously with chunking.
        
        Args:
            file_path: Path to the large file
            processor_func: Function to process the file
            chunk_processor: Optional function to process file chunks
            progress_callback: Optional progress callback function
            **kwargs: Additional arguments for processor function
            
        Returns:
            Processing result
        """
        try:
            file_path = Path(file_path)
            
            # Get file size for progress tracking
            file_size = file_path.stat().st_size
            
            # Initialize progress tracker
            progress = ProgressTracker(1, f"Processing {file_path.name}")
            if progress_callback:
                progress.add_callback(progress_callback)
            
            start_time = time.time()
            
            if chunk_processor and file_size > 10 * 1024 * 1024:  # 10MB threshold
                # Process large file in chunks
                result = await self._process_file_in_chunks(
                    file_path, chunk_processor, progress, **kwargs
                )
            else:
                # Process file normally
                loop = asyncio.get_event_loop()
                if self.use_process_pool:
                    with ProcessPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor, processor_func, file_path, **kwargs
                        )
                else:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor, processor_func, file_path, **kwargs
                        )
            
            processing_time = time.time() - start_time
            progress.update(completed=1)
            
            return {
                "file_path": str(file_path),
                "status": "success",
                "result": result,
                "processing_time": processing_time,
                "file_size": file_size
            }
            
        except Exception as e:
            self.logger.error(f"Error processing large file {file_path}: {e}")
            progress.update(failed=1)
            return {
                "file_path": str(file_path),
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0
            }
    
    async def _process_file_in_chunks(
        self,
        file_path: Path,
        chunk_processor: Callable,
        progress: ProgressTracker,
        **kwargs
    ) -> Any:
        """Process a file in chunks."""
        chunk_results = []
        
        if self._aiofiles_available:
            # Use aiofiles for async file I/O
            import aiofiles
            
            async with aiofiles.open(file_path, 'rb') as f:
                chunk_num = 0
                while True:
                    chunk_data = await f.read(self.chunk_size * 1024 * 1024)  # chunk_size MB
                    if not chunk_data:
                        break
                    
                    # Process chunk in thread pool
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        chunk_result = await loop.run_in_executor(
                            executor, chunk_processor, chunk_data, chunk_num, **kwargs
                        )
                    
                    chunk_results.append(chunk_result)
                    chunk_num += 1
        else:
            # Fallback to synchronous file I/O
            with open(file_path, 'rb') as f:
                chunk_num = 0
                while True:
                    chunk_data = f.read(self.chunk_size * 1024 * 1024)
                    if not chunk_data:
                        break
                    
                    # Process chunk in thread pool
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        chunk_result = await loop.run_in_executor(
                            executor, chunk_processor, chunk_data, chunk_num, **kwargs
                        )
                    
                    chunk_results.append(chunk_result)
                    chunk_num += 1
        
        # Combine chunk results
        return self._combine_chunk_results(chunk_results)
    
    def _combine_chunk_results(self, chunk_results: List[Any]) -> Any:
        """Combine results from multiple chunks."""
        if not chunk_results:
            return None
        
        # Simple concatenation for now - can be made more sophisticated
        if isinstance(chunk_results[0], dict):
            combined = {}
            for chunk_result in chunk_results:
                if isinstance(chunk_result, dict):
                    combined.update(chunk_result)
            return combined
        elif isinstance(chunk_results[0], list):
            combined = []
            for chunk_result in chunk_results:
                if isinstance(chunk_result, list):
                    combined.extend(chunk_result)
            return combined
        else:
            return chunk_results
    
    async def batch_process_with_retry(
        self,
        items: List[Any],
        processor_func: Callable,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process items in batches with retry logic.
        
        Args:
            items: List of items to process
            processor_func: Function to process each item
            max_retries: Maximum number of retries for failed items
            retry_delay: Delay between retries in seconds
            progress_callback: Optional progress callback function
            **kwargs: Additional arguments for processor function
            
        Returns:
            List of processing results
        """
        if not items:
            return []
        
        # Initialize progress tracker
        progress = ProgressTracker(len(items), "Batch processing with retry")
        if progress_callback:
            progress.add_callback(progress_callback)
        
        results = []
        failed_items = []
        
        # First attempt
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_retry(item: Any, attempt: int = 1) -> Dict[str, Any]:
            """Process item with retry logic."""
            async with semaphore:
                try:
                    loop = asyncio.get_event_loop()
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor, processor_func, item, **kwargs
                        )
                    
                    progress.update(completed=1)
                    return {
                        "item": item,
                        "status": "success",
                        "result": result,
                        "attempts": attempt
                    }
                    
                except Exception as e:
                    if attempt < max_retries:
                        # Retry after delay
                        await asyncio.sleep(retry_delay)
                        return await process_with_retry(item, attempt + 1)
                    else:
                        progress.update(failed=1)
                        return {
                            "item": item,
                            "status": "error",
                            "error": str(e),
                            "attempts": attempt
                        }
        
        # Create tasks for all items
        tasks = [process_with_retry(item) for item in items]
        
        # Process tasks and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append({
                    "item": items[i],
                    "status": "error",
                    "error": str(result),
                    "attempts": max_retries
                })
            else:
                final_results.append(result)
        
        return final_results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "max_workers": self.max_workers,
            "use_process_pool": self.use_process_pool,
            "chunk_size_mb": self.chunk_size,
            "aiofiles_available": self._aiofiles_available,
            "cpu_count": os.cpu_count()
        }


# Convenience functions for easy usage
async def process_files_async(
    file_paths: List[Union[str, Path]],
    processor_func: Callable,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Convenience function to process multiple files asynchronously.
    
    Args:
        file_paths: List of file paths to process
        processor_func: Function to process each file
        max_workers: Maximum number of worker threads
        progress_callback: Optional progress callback function
        **kwargs: Additional arguments for processor function
        
    Returns:
        List of processing results
    """
    processor = AsyncDocumentProcessor(max_workers=max_workers)
    return await processor.process_files_async(
        file_paths, processor_func, progress_callback, **kwargs
    )


async def process_large_file_async(
    file_path: Union[str, Path],
    processor_func: Callable,
    chunk_processor: Optional[Callable] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to process a large file asynchronously.
    
    Args:
        file_path: Path to the large file
        processor_func: Function to process the file
        chunk_processor: Optional function to process file chunks
        progress_callback: Optional progress callback function
        **kwargs: Additional arguments for processor function
        
    Returns:
        Processing result
    """
    processor = AsyncDocumentProcessor()
    return await processor.process_large_file_async(
        file_path, processor_func, chunk_processor, progress_callback, **kwargs
    )
