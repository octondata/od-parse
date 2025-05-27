"""
Optimized document processor for handling large documents with minimal resources.

This module provides memory-efficient and CPU-efficient approaches to process
large documents while maintaining a small resource footprint.
"""

import gc
import mmap
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from od_parse.config.settings import get_config, load_config
from od_parse.utils.logging_utils import get_logger


class OptimizedProcessor:
    """
    Memory and CPU efficient document processor.
    
    This class provides methods for processing large documents with minimal
    resource usage through techniques like streaming, batching, and lazy loading.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the optimized processor with configuration options.
        
        Args:
            config: Configuration dictionary with the following options:
                - batch_size: Number of pages to process at once (default: 5)
                - use_memory_mapping: Whether to use memory mapping (default: True)
                - max_workers: Maximum number of worker threads (default: 2)
                - low_memory_mode: Enable aggressive memory optimization (default: False)
                - temp_dir: Directory for temporary files (default: system temp)
                - image_dpi: DPI for image conversion (default: 150)
                - image_quality: JPEG quality for image conversion (default: 85)
                - max_image_dimension: Maximum dimension for images (default: 1500)
                
        Raises:
            ConfigurationError: If the configuration is invalid
            ResourceError: If required system resources are not available
        """
        try:
            self.logger = get_logger(__name__)
            self._validate_config(config or {})
            self.config = config or {}
            
            # Load configuration if not already loaded
            if not get_config():
                load_config()
                
        except Exception as e:
            if isinstance(e, ConfigurationError):
                raise
            raise ConfigurationError(f"Initialization failed: {str(e)}")
        
        # Load configuration if not already loaded
        if not get_config():
            load_config()
            
        # Set default configuration values
        self.batch_size = self.config.get("batch_size", 5)
        self.use_memory_mapping = self.config.get("use_memory_mapping", True)
        self.max_workers = self.config.get("max_workers", 2)
        self.low_memory_mode = self.config.get("low_memory_mode", False)
        self.temp_dir = self.config.get("temp_dir", tempfile.gettempdir())
        self.image_dpi = self.config.get("image_dpi", 150)
        self.image_quality = self.config.get("image_quality", 85)
        self.max_image_dimension = self.config.get("max_image_dimension", 1500)
        
        # Create temp directory if it doesn't exist
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Track temporary files for cleanup
        self._temp_files = []
    
    def __del__(self):
        """Clean up temporary files and resources."""
        self.cleanup()
    
    def cleanup(self):
        """Clean up temporary files and resources."""
        # Close thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        # Remove temporary files
        for temp_file in self._temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
        
        # Clear temporary files list
        self._temp_files = []
        
        # Force garbage collection
        gc.collect()
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration settings."""
        try:
            # Validate batch size
            batch_size = config.get('batch_size', 5)
            if batch_size < 1:
                raise ConfigurationError("batch_size must be greater than 0")
                
            # Validate worker count
            max_workers = config.get('max_workers', 2)
            if max_workers < 1:
                raise ConfigurationError("max_workers must be greater than 0")
                
            # Validate image settings
            image_dpi = config.get('image_dpi', 150)
            if image_dpi < 72:
                raise ConfigurationError("image_dpi must be at least 72")
                
            image_quality = config.get('image_quality', 85)
            if not 1 <= image_quality <= 100:
                raise ConfigurationError("image_quality must be between 1 and 100")
                
            # Validate temp directory
            temp_dir = config.get('temp_dir')
            if temp_dir and not Path(temp_dir).exists():
                raise ConfigurationError(f"Temp directory does not exist: {temp_dir}")
                
        except Exception as e:
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(f"Invalid configuration: {str(e)}")
            raise

    def process_document(self, file_path: Union[str, Path],
                         progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Process a document with the configured settings.
        
        Args:
            file_path: Path to the document file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict containing the processed document data
            
        Raises:
            FileNotFoundError: If the file does not exist
            FileTypeError: If the file type is not supported
            FileCorruptedError: If the file is corrupted
            ProcessingError: If processing fails
            MemoryError: If there's not enough memory
            OCRError: If OCR processing fails
            TableExtractionError: If table extraction fails
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            if file_path.suffix.lower() not in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
                raise FileTypeError(f"Unsupported file type: {file_path.suffix}")
                
            self.logger.info(f"Processing document: {file_path}")
            try:
                # Process the document
                result = self.process_large_pdf(file_path)
                return result
            except Exception as e:
                if isinstance(e, (FileNotFoundError, FileTypeError, FileCorruptedError, ProcessingError, MemoryError, OCRError, TableExtractionError)):
                    raise
                raise ProcessingError(f"Document processing failed: {str(e)}") from e
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, FileTypeError, FileCorruptedError, ProcessingError, MemoryError, OCRError, TableExtractionError)):
                raise
            raise ProcessingError(f"Document processing failed: {str(e)}")

    def process_large_pdf(self, pdf_path: Union[str, Path], 
                          processor_func: callable = None, 
                          **processor_kwargs) -> Dict[str, Any]:
        """
        Process a large PDF document with minimal memory usage.
        
        Args:
            pdf_path: Path to the PDF file
            processor_func: Optional function to process each page
            **processor_kwargs: Additional arguments for processor_func
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            FileCorruptedError: If the PDF is corrupted
            ProcessingError: If processing fails
            MemoryError: If there's not enough memory
        """
        try:
            # Import libraries here to avoid loading them if not needed
            import fitz  # PyMuPDF
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProcessingError("Required libraries not installed. Please install with: pip install pymupdf pdf2image") from e
            
        try:
            # Convert to Path object and validate
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            # Create temp directory
            try:
                temp_dir = tempfile.mkdtemp()
            except Exception as e:
                raise ProcessingError(f"Failed to create temp directory: {str(e)}")
                
            # Open the PDF document with PyMuPDF
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            
            # Initialize result dictionary
            result = {
                "metadata": {
                    "file_name": os.path.basename(pdf_path),
                    "page_count": total_pages
                },
                "content": [],
                "tables": [],
                "forms": [],
                "images": []
            }
            
            # Process the document in batches
            for batch_start in range(0, total_pages, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total_pages)
                self.logger.info(f"Processing pages {batch_start+1}-{batch_end} of {total_pages}")
                
                # Process batch
                batch_result = self._process_page_batch(
                    pdf_path, batch_start, batch_end, temp_dir, processor_func, **processor_kwargs
                )
                
                # Merge batch results
                self._merge_results(result, batch_result)
                
                # Force garbage collection after each batch
                if self.low_memory_mode:
                    gc.collect()
            
            # Close the document
            doc.close()
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error processing large PDF: {str(e)}")
            return {"error": str(e)}

    def _process_page_batch(self, pdf_path: Union[str, Path], 
                           start_page: int, 
                           end_page: int,
                           temp_dir: str,
                           processor_func: callable,
                           **processor_kwargs) -> Dict[str, Any]:
        """Process a batch of pages from the PDF.
        
        Args:
            pdf_path: Path to the PDF file
            start_page: Starting page number
            end_page: Ending page number
            temp_dir: Directory for temporary files
            processor_func: Function to process each page
            **processor_kwargs: Additional arguments for processor_func
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            ProcessingError: If batch processing fails
            MemoryError: If there's not enough memory
        """
        try:
            # Convert pages to images using pdf2image
            # This is more memory efficient than processing the PDF directly
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ProcessingError("pdf2image not installed. Please install with: pip install pdf2image") from e
            
        try:
            
            # Create temporary directory for this batch
            batch_temp_dir = os.path.join(temp_dir, f"batch_{start_page}_{end_page}")
            os.makedirs(batch_temp_dir, exist_ok=True)
            
            # Convert pages to images with memory optimization
            images = convert_from_path(
                pdf_path,
                first_page=start_page + 1,  # pdf2image uses 1-based indexing
                last_page=end_page,
                dpi=self.image_dpi,
                output_folder=batch_temp_dir,
                fmt="jpeg",
                thread_count=1,  # Use single thread to reduce memory usage
                use_pdftocairo=True,  # pdftocairo is more memory efficient
                paths_only=True  # Return paths instead of PIL images
            )
            
            # Track temporary files for cleanup
            self._temp_files.extend(images)
            
            # Process images in parallel
            if self.max_workers > 1 and not self.low_memory_mode:
                # Process images in parallel with thread pool
                futures = []
                for i, image_path in enumerate(images):
                    page_num = start_page + i
                    future = self.executor.submit(
                        self._process_single_page, 
                        image_path, 
                        page_num,
                        processor_func,
                        **processor_kwargs
                    )
                    futures.append(future)
                
                # Collect results
                page_results = [future.result() for future in futures]
            else:
                # Process images sequentially
                page_results = []
                for i, image_path in enumerate(images):
                    page_num = start_page + i
                    result = self._process_single_page(
                        image_path, 
                        page_num,
                        processor_func,
                        **processor_kwargs
                    )
                    page_results.append(result)
                    
                    # Force garbage collection after each page in low memory mode
                    if self.low_memory_mode:
                        gc.collect()
            
            # Combine page results
            batch_result = self._combine_page_results(page_results)
            
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(batch_temp_dir)
            except Exception as e:
                self.logger.warning(f"Failed to remove temporary directory {batch_temp_dir}: {str(e)}")
            
            return batch_result
        
        except Exception as e:
            self.logger.error(f"Error processing page batch {start_page}-{end_page}: {str(e)}")
            return {"error": str(e)}

    def _process_single_page(self, image_path: str, page_num: int, processor_func: callable, **processor_kwargs) -> Dict[str, Any]:
        """Process a single page.
        
        Args:
            image_path: Path to the page image
            page_num: Page number
            processor_func: Function to process the page
            **processor_kwargs: Additional arguments for processor_func
            
        Returns:
            Dictionary containing processed results
            
        Raises:
            ProcessingError: If page processing fails
            OCRError: If OCR fails
            MemoryError: If there's not enough memory
        """
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            # Open and process the image
            with Image.open(image_path) as img:
                # Resize image if needed to reduce memory usage
                if max(img.width, img.height) > self.max_image_dimension:
                    scale = self.max_image_dimension / max(img.width, img.height)
                    new_size = tuple(int(dim * scale) for dim in (img.width, img.height))
                    img = img.resize(new_size, Image.LANCZOS)
                    
                # Convert to numpy array for processing
                img_array = np.array(img)
                
                # Process the image
                if processor_func:
                    result = processor_func(img_array, **processor_kwargs)
                else:
                    result = self._default_processor(img_array)
                    
                # Add page metadata
                if isinstance(result, dict):
                    if "metadata" not in result:
                        result["metadata"] = {}
                    result["metadata"]["page_num"] = page_num
                    
                return result
                    
        except FileNotFoundError:
            raise
        except MemoryError:
            raise
        except Exception as e:
            self.logger.error(f"Error processing page {page_num}: {str(e)}")
            raise ProcessingError(f"Failed to process image: {str(e)}") from e

    def _combine_page_results(self, page_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine results from multiple pages.
        
        Args:
            page_results: List of results from individual pages
            
        Returns:
            Combined results dictionary
            
        Raises:
            ProcessingError: If combining results fails
        """
        try:
            if not page_results:
                raise ProcessingError("No page results to combine")
                
            combined = {
                "content": [],
                "tables": [],
                "forms": [],
                "images": []
            }
        except Exception as e:
            raise ProcessingError(f"Failed to initialize combined results: {str(e)}") from e
            
        try:
            for result in page_results:
                if not isinstance(result, dict):
                    continue
                    
                # Merge the results
                self._merge_results(combined, result)
                
            return combined
            
        except Exception as e:
            raise ProcessingError(f"Failed to combine page results: {str(e)}") from e
        
        for result in page_results:
            if not isinstance(result, dict):
                continue
                
            # Combine content
            if "content" in result and isinstance(result["content"], list):
                combined["content"].extend(result["content"])
            
            # Combine tables
            if "tables" in result and isinstance(result["tables"], list):
                combined["tables"].extend(result["tables"])
            
            # Combine forms
            if "forms" in result and isinstance(result["forms"], list):
                combined["forms"].extend(result["forms"])
            
            # Combine images
            if "images" in result and isinstance(result["images"], list):
                combined["images"].extend(result["images"])
        
        return combined
    
    def _merge_results(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """Merge source results into target results."""
        if not isinstance(source, dict):
            return
            
        # Merge content
        if "content" in source and isinstance(source["content"], list):
            if "content" not in target:
                target["content"] = []
            target["content"].extend(source["content"])
        
        # Merge tables
        if "tables" in source and isinstance(source["tables"], list):
            if "tables" not in target:
                target["tables"] = []
            target["tables"].extend(source["tables"])
        
        # Merge forms
        if "forms" in source and isinstance(source["forms"], list):
            if "forms" not in target:
                target["forms"] = []
            target["forms"].extend(source["forms"])
        
        # Merge images
        if "images" in source and isinstance(source["images"], list):
            if "images" not in target:
                target["images"] = []
            target["images"].extend(source["images"])

    def stream_large_pdf_text(self, pdf_path: Union[str, Path]) -> Iterator[Tuple[int, str]]:
        """
        Stream text from a large PDF document with minimal memory usage.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Iterator yielding (page_number, text) tuples
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
            FileCorruptedError: If the PDF is corrupted
            ProcessingError: If streaming fails
            MemoryError: If there's not enough memory
        """
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
                
            if pdf_path.suffix.lower() != '.pdf':
                raise FileTypeError(f"Not a PDF file: {pdf_path}")          # Import PyMuPDF here to avoid loading it if not needed
            import fitz
        except ImportError as e:
            raise ProcessingError("PyMuPDF not installed. Please install with: pip install pymupdf") from e
            
        try:
            # Open the PDF document with PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_num in range(len(doc)):
                try:
                    # Get page
                    page = doc[page_num]
                    
                    # Extract text
                    text = page.get_text()
                    
                    # Yield page number and text
                    yield page_num, text
                    
                    # Free page resources
                    page = None
                    
                    # Force garbage collection in low memory mode
                    if self.low_memory_mode and page_num % 10 == 0:
                        gc.collect()
                
                except Exception as e:
                    self.logger.error(f"Error processing page {page_num}: {str(e)}")
                    yield page_num, f"[ERROR: {str(e)}]"
            
            # Close the document
            doc.close()
        
        except Exception as e:
            self.logger.error(f"Error streaming PDF text: {str(e)}")
            yield 0, f"[ERROR: {str(e)}]"
    
    def memory_mapped_file_processing(self, file_path: Union[str, Path], 
                                     chunk_size: int = 1024*1024) -> Iterator[bytes]:
        """
        Process a large file using memory mapping to minimize memory usage.
        
        Args:
            file_path: Path to the file
            chunk_size: Size of chunks to process at once
            
        Yields:
            Chunks of file data
        """
        if not self.use_memory_mapping:
            # Fallback to regular file reading if memory mapping is disabled
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            return
        
        try:
            # Open the file for reading
            with open(file_path, 'rb') as f:
                # Create memory map
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Process file in chunks
                    file_size = mm.size()
                    for offset in range(0, file_size, chunk_size):
                        # Calculate chunk size
                        current_chunk_size = min(chunk_size, file_size - offset)
                        
                        # Read chunk
                        mm.seek(offset)
                        chunk = mm.read(current_chunk_size)
                        
                        # Yield chunk
                        yield chunk
        
        except Exception as e:
            self.logger.error(f"Error processing file with memory mapping: {str(e)}")
            # Fallback to regular file reading
            self.logger.info("Falling back to regular file reading")
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
