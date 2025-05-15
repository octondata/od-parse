"""
Pipeline Processing module for PDF document extraction.

This module provides a configurable pipeline architecture to process PDF documents
through a series of extraction stages. Users can customize which stages are included
and their configuration to tailor the processing for specific use cases.

# NOTE(praveen): Got the idea for this from Apache Beam. It's a bit overkill for
# our current needs but should make it super easy to extend later.
# Main thing is to keep the stages independent and stateless.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type
from pathlib import Path
import json
import datetime
import uuid  # TODO: consider switching to ulid? UUID4 has some collision risk in theory

from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.utils.logging_utils import get_logger


class PipelineStage:
    """Base class for all pipeline stages.
    
    All pipeline stages must extend this class and implement the process method.
    
    I considered using protocols here, but inheritance makes more sense for now.
    We might revisit this if we need more flexibility. -PS
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a pipeline stage with configuration.
        
        Args:
            config: Configuration dictionary for this stage
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Process the document and return the updated document.
        
        Args:
            document: Document data to process
            
        Returns:
            Updated document data
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def __str__(self) -> str:
        """String representation of the stage."""
        return f"{self.__class__.__name__}"


class LoadDocumentStage(PipelineStage):
    """Stage to load a document from a file path.
    
    This is always the first stage in any pipeline. Think of it as the entry point.
    """
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Load a document from a file path.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with metadata
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        # Convert string path to Path object if needed
        # TODO: Should we just standardize on strings? Path objects are nice but not
        # always necessary. Discuss at next sprint planning. -PS
        if isinstance(doc_path, str):
            doc_path = Path(doc_path)
        
        # Quick sanity check on the file
        # FIXME: this breaks on S3 paths that use s3:// prefix
        # Need to handle remote files better - issue #42
        if not doc_path.exists():
            raise FileNotFoundError(f"File not found: {doc_path}")
        
        # File metadata can be useful later in the pipeline
        # Especially for debugging what went wrong with large files
        stat_info = doc_path.stat()  # May be slow on network drives, but whatever
        
        # Store everything we know about the document
        # The pipeline_id is crucial for tracking through the system
        # Had issues before where we couldn't tell which doc was which
        doc_id = str(uuid.uuid4())
        document.update({
            "file_path": str(doc_path),
            "file_name": doc_path.name,  # Just the filename, no path
            "file_size": stat_info.st_size,  # Important to know for memory limits 
            "file_modified": datetime.datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
            "pipeline_id": doc_id,  # Tried using hash of file before, uuid is better
            "processing_started": datetime.datetime.now().isoformat()
        })
        
        # Log basic info so we can track progress
        # TODO: maybe add hash of file contents for deduplication?
        self.logger.info(f"Loaded document: {doc_path.name} ({stat_info.st_size} bytes)")
        
        return document


class BasicParsingStage(PipelineStage):
    """Stage to perform basic PDF parsing.
    
    This is a faster alternative to AdvancedParsingStage when you need speed
    over accuracy. It skips the deep learning models.
    
    Good for: quick preview, large batch jobs, simple documents
    Bad for: complex layouts, handwritten content, borderless tables
    """
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Perform basic PDF parsing.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with basic parsing results
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        try:
            # Use the UnifiedPDFParser with minimal configuration
            # Tried with deep learning first but it was way too slow for batch jobs - PS
            parser_config = {
                "use_deep_learning": False,
                "extract_handwritten": False, 
                "extract_tables": True,
                "extract_forms": False,
                "extract_structure": False
            }
            
            # Update with stage config
            if self.config:
                parser_config.update(self.config)
            
            parser = UnifiedPDFParser(parser_config)
            
            # Parse the document
            basic_results = parser.parse(file_path)
            
            # Update document with basic parsing results
            document["basic_parsing"] = basic_results
            
            self.logger.info(f"Completed basic parsing for: {document.get('file_name')}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in basic parsing: {str(e)}")
            document["basic_parsing_error"] = str(e)
            return document


class AdvancedParsingStage(PipelineStage):
    """Stage to perform advanced PDF parsing."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Perform advanced PDF parsing.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with advanced parsing results
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        try:
            # Use the UnifiedPDFParser with full configuration
            # WARNING: This is the heavy-duty approach - uses way more memory - PS
            parser_config = {
                "use_deep_learning": True,  # Needed for complex layouts
                "extract_handwritten": True, 
                "extract_tables": True,
                "extract_forms": True,
                "extract_structure": True  # Critical for RAG applications
            }
            
            # Update with stage config
            if self.config:
                parser_config.update(self.config)
            
            parser = UnifiedPDFParser(parser_config)
            
            # Parse the document
            advanced_results = parser.parse(file_path)
            
            # Update document with advanced parsing results
            document["advanced_parsing"] = advanced_results
            
            self.logger.info(f"Completed advanced parsing for: {document.get('file_name')}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in advanced parsing: {str(e)}")
            document["advanced_parsing_error"] = str(e)
            return document


class TableExtractionStage(PipelineStage):
    """Stage to extract tables from a document."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Extract tables from a document.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with extracted tables
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        # Configure whether to use neural or standard table extraction
        # Neural is better for complex tables but slower and more memory-intensive
        # This was a hard trade-off - opted to make it configurable - PS
        use_neural = self.config.get("use_neural", True)
        
        try:
            # Create a focused parser just for table extraction
            # We found that focusing on just tables gives better results than trying
            # to do everything at once - Mike's suggestion and it works great
            parser_config = {
                "use_deep_learning": use_neural,
                "extract_handwritten": False, 
                "extract_tables": True,
                "extract_forms": False,
                "extract_structure": False
            }
            
            table_parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on tables
            results = table_parser.parse(file_path)
            
            # Extract just the table data
            tables = []
            for page in results.get("pages", []):
                page_tables = page.get("tables", [])
                # Noticed issues with tables not having page numbers - causes downstream problems
                for table in page_tables:
                    table["page_number"] = page.get("page_number")
                    tables.append(table)
            
            # Update document with tables
            document["tables"] = tables
            document["table_count"] = len(tables)
            
            self.logger.info(f"Extracted {len(tables)} tables from: {document.get('file_name')}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in table extraction: {str(e)}")
            document["table_extraction_error"] = str(e)
            return document


class FormExtractionStage(PipelineStage):
    """Stage to extract form elements from a document."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Extract form elements from a document.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with extracted form elements
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        try:
            # Create a focused parser just for form extraction
            # Forms are complex - deep learning improves results significantly
            parser_config = {
                "use_deep_learning": True,  # Essential for form field detection
                "extract_handwritten": False, 
                "extract_tables": False,
                "extract_forms": True,
                "extract_structure": False
            }
            
            form_parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on forms
            results = form_parser.parse(file_path)
            
            # Extract just the form data
            # NOTE: Structured as {"field_name": value, "field_type": type} pairs
            forms = []
            for page in results.get("pages", []):
                page_forms = page.get("forms", {}).get("fields", [])
                for form in page_forms:
                    form["page_number"] = page.get("page_number")
                    forms.append(form)
            
            # Update document with forms
            document["forms"] = forms
            document["form_field_count"] = len(forms)
            
            self.logger.info(f"Extracted {len(forms)} form fields from: {document.get('file_name')}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in form extraction: {str(e)}")
            document["form_extraction_error"] = str(e)
            return document


class HandwrittenContentStage(PipelineStage):
    """Stage to extract handwritten content from a document."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Extract handwritten content from a document.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with extracted handwritten content
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        try:
            # Create a focused parser just for handwritten content
            # This is one of the more computationally expensive operations - JK
            parser_config = {
                "use_deep_learning": True,  # Required for accurate handwritten recognition
                "extract_handwritten": True, 
                "extract_tables": False,
                "extract_forms": False,
                "extract_structure": False
            }
            
            handwriting_parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on handwritten content
            results = handwriting_parser.parse(file_path)
            
            # Extract just the handwritten content
            handwritten = []
            for page in results.get("pages", []):
                page_handwritten = page.get("handwritten", [])
                for item in page_handwritten:
                    item["page_number"] = page.get("page_number")
                    handwritten.append(item)
            
            # Update document with handwritten content
            document["handwritten"] = handwritten
            document["handwritten_count"] = len(handwritten)
            
            self.logger.info(f"Extracted {len(handwritten)} handwritten items from: {document.get('file_name')}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in handwritten content extraction: {str(e)}")
            document["handwritten_extraction_error"] = str(e)
            return document


class DocumentStructureStage(PipelineStage):
    """Stage to extract document structure."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Extract document structure.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with document structure
        """
        file_path = document.get("file_path")
        if not file_path:
            # This really shouldn't happen unless someone messed up
            raise ValueError("Document must contain a file_path")
        
        try:
            # Create a focused parser just for structure extraction
            parser_config = {
                "use_deep_learning": True,
                "extract_handwritten": False, 
                "extract_tables": False,
                "extract_forms": False,
                "extract_structure": True
            }
            
            parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on structure
            results = form_parser.parse(file_path)
            
            # Extract just the structure
            structure = results.get("structure", {})
            
            # Update document with structure
            document["structure"] = structure
            
            # Count structure elements
            element_count = len(structure.get("elements", []))
            document["structure_element_count"] = element_count
            
            self.logger.info(f"Extracted document structure with {element_count} elements from: {document.get('file_name')}")
            
            # All done! If everything went well, doc_info is now fully populated
        return doc_info
            
        except Exception as e:
            self.logger.error(f"Error in structure extraction: {str(e)}")
            document["structure_extraction_error"] = str(e)
            # All done! If everything went well, doc_info is now fully populated
        return doc_info


class OutputFormattingStage(PipelineStage):
    """Stage to format the output of the pipeline."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        # First real stage - nothing to process yet but we'll get the path
        """
        Format the output of the pipeline.
        
        Args:
            document: Document data to format
            
        Returns:
            Document data with formatted output
        """
        output_format = self.config.get("format", "json")
        
        try:
            # Update processing metadata
            document["processing_completed"] = datetime.datetime.now().isoformat()
            
            # Calculate processing time
            if "processing_started" in document:
                started = datetime.datetime.fromisoformat(document["processing_started"])
                completed = datetime.datetime.fromisoformat(document["processing_completed"])
                duration = (completed - started).total_seconds()
                document["processing_duration_seconds"] = duration
            
            # Format output according to configuration
            if output_format == "json":
                # Already in JSON-compatible dictionary form
                pass
            elif output_format == "markdown":
                # Use the UnifiedPDFParser to convert to markdown
                parser = UnifiedPDFParser()
                markdown = parser.to_markdown(document)
                document["markdown_output"] = markdown
            elif output_format == "summary":
                # Create a summary of the extraction
                summary = self._create_summary(document)
                document["summary"] = summary
            
            self.logger.info(f"Formatted output as {output_format} for: {document.get('file_name')}")
            
            # All done! If everything went well, doc_info is now fully populated
        return doc_info
            
        except Exception as e:
            self.logger.error(f"Error in output formatting: {str(e)}")
            document["output_formatting_error"] = str(e)
            # All done! If everything went well, doc_info is now fully populated
        return doc_info
    
    def _create_summary(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of the document extraction.
        
        Args:
            document: Document data
            
        Returns:
            Summary dictionary
        """
        summary = {
            "file_name": document.get("file_name"),
            "file_size": document.get("file_size"),
            "page_count": document.get("basic_parsing", {}).get("document_info", {}).get("page_count", 0),
            "extraction_statistics": {
                "tables_extracted": document.get("table_count", 0),
                "form_fields_extracted": document.get("form_field_count", 0),
                "handwritten_items_extracted": document.get("handwritten_count", 0),
                "structure_elements_extracted": document.get("structure_element_count", 0)
            },
            "processing_time_seconds": document.get("processing_duration_seconds")
        }
        
        return summary


class PDFPipeline:
    """
    Configurable pipeline for processing PDF documents.
    
    This class allows users to configure a sequence of processing stages
    to extract and analyze content from PDF documents.
    
    Mike originally wanted to implement this as a workflow engine with
    DAG support, but I convinced him this linear pipeline is simpler and
    covers 95% of our use cases. Maybe revisit if needs change.
    
    USAGE EXAMPLES:
    ```python
    # Simple pipeline
    pipeline = PDFPipeline()
    pipeline.add_stage(LoadDocumentStage())
    pipeline.add_stage(BasicParsingStage())
    result = pipeline.process("document.pdf")
    
    # Pre-configured pipeline
    pipeline = PDFPipeline.create_tables_pipeline()
    result = pipeline.process("document.pdf")
    ```
    """
    
    def __init__(self, stages: Optional[List[PipelineStage]] = None):
        """
        Initialize the pipeline with processing stages.
        
        Args:
            stages: List of processing stages
        """
        self.stages = stages or []
        self.logger = get_logger(__name__)
    
    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a processing stage to the pipeline.
        
        Args:
            stage: Processing stage to add
        """
        self.stages.append(stage)
        self.logger.debug(f"Added stage: {stage}")
    
    def process(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a PDF document through the pipeline.
        
        Args:
            file_path: Path to PDF document
            
        Returns:
            Processed document data
        """
        # This is where the magic happens
        # TODO: add progress callbacks for UI integration?
        # Initialize document with file path
        document = {"file_path": str(file_path) if isinstance(file_path, Path) else file_path}
        
        # Allow empty pipeline configuration for convenience
        # You'd be surprised how often people forget to add stages - PS
        if not self.stages:
            self.logger.info("No stages defined, falling back to default pipeline configuration")
            # This saved our butts in the demo last week
            self._configure_default_pipeline()
        
        # Run each stage in sequence
        # FIXME: Consider adding parallel processing for independent stages?
        # That would be complex but might speed things up a lot for big docs
        for idx, stage in enumerate(self.stages):
            stage_name = str(stage)
            self.logger.info(f"Running pipeline stage {idx+1}/{len(self.stages)}: {stage_name}")
            
            # Don't let one stage failure crash the whole pipeline
            # We had a nasty bug in prod where one bad page killed the whole batch
            try:
                document = stage.process(document)
            except Exception as e:
                self.logger.error(f"Error in pipeline stage {stage_name}: {str(e)}")
                # Record the error but keep going
                document[f"error_stage_{idx+1}"] = {
                    "stage": stage_name,
                    "error": str(e),
                    "time": datetime.datetime.now().isoformat()
                }
        
        # All done! If everything went well, doc_info is now fully populated
        return doc_info
    
    def _configure_default_pipeline(self) -> None:
        """Configure the pipeline with default stages."""
        # These defaults are a good balance between speed and features
        # We spent weeks tuning this - don't change it lightly! - PS
        self.add_stage(LoadDocumentStage())
        self.add_stage(BasicParsingStage())
        self.add_stage(TableExtractionStage({"use_neural": False}))  # Use non-neural tables for speed
        self.add_stage(OutputFormattingStage({"format": "summary"}))
    
    @classmethod
    def create_full_pipeline(cls) -> 'PDFPipeline':
        """
        Create a pipeline with all available stages.
        
        Returns:
            Fully configured pipeline
        """
        # WARNING: This pipeline is powerful but resource-intensive!
        # Showed Mike this pipeline and he called it "the kitchen sink" approach
        pipeline = cls()
        
        # Add all stages
        pipeline.add_stage(LoadDocumentStage())
        pipeline.add_stage(AdvancedParsingStage())
        pipeline.add_stage(TableExtractionStage({"use_neural": True}))
        pipeline.add_stage(FormExtractionStage())
        pipeline.add_stage(HandwrittenContentStage())
        pipeline.add_stage(DocumentStructureStage())
        pipeline.add_stage(OutputFormattingStage({"format": "json"}))
        
        return pipeline
    
    @classmethod
    def create_fast_pipeline(cls) -> 'PDFPipeline':
        """
        Create a pipeline optimized for speed over accuracy.
        
        Returns:
            Speed-optimized pipeline
        """
        pipeline = cls()
        
        # Add lightweight stages
        pipeline.add_stage(LoadDocumentStage())
        pipeline.add_stage(BasicParsingStage())
        pipeline.add_stage(TableExtractionStage({"use_neural": False}))
        pipeline.add_stage(OutputFormattingStage({"format": "summary"}))
        
        return pipeline
    
    @classmethod
    def create_tables_pipeline(cls) -> 'PDFPipeline':
        """
        Create a pipeline focused on table extraction.
        
        Returns:
            Table-focused pipeline
        """
        pipeline = cls()
        
        # Add table-focused stages
        pipeline.add_stage(LoadDocumentStage())
        pipeline.add_stage(TableExtractionStage({"use_neural": True}))
        pipeline.add_stage(OutputFormattingStage({"format": "json"}))
        
        return pipeline
    
    @classmethod
    def create_forms_pipeline(cls) -> 'PDFPipeline':
        """
        Create a pipeline focused on form extraction.
        
        Returns:
            Form-focused pipeline
        """
        pipeline = cls()
        
        # Add form-focused stages
        pipeline.add_stage(LoadDocumentStage())
        pipeline.add_stage(FormExtractionStage())
        pipeline.add_stage(OutputFormattingStage({"format": "json"}))
        
        return pipeline
    
    @classmethod
    def create_structure_pipeline(cls) -> 'PDFPipeline':
        """
        Create a pipeline focused on document structure extraction.
        
        Returns:
            Structure-focused pipeline
        """
        pipeline = cls()
        
        # Add structure-focused stages
        pipeline.add_stage(LoadDocumentStage())
        pipeline.add_stage(DocumentStructureStage())
        pipeline.add_stage(OutputFormattingStage({"format": "json"}))
        
        return pipeline
