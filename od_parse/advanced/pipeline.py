"""
Pipeline Processing module for PDF document extraction.

This module provides a configurable pipeline architecture to process PDF documents
through a series of extraction stages. Users can customize which stages are included
and their configuration to tailor the processing for specific use cases.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Type
from pathlib import Path
import json
import datetime
import uuid

from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.utils.logging_utils import get_logger


class PipelineStage:
    """Base class for all pipeline stages."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize a pipeline stage with configuration.
        
        Args:
            config: Configuration dictionary for this stage
        """
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
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
    """Stage to load a document from a file path."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load a document from a file path.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with metadata
        """
        file_path = document.get("file_path")
        if not file_path:
            raise ValueError("Document must contain a file_path")
        
        # Ensure path is a Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Get file metadata
        stat = file_path.stat()
        
        # Update document with metadata
        document.update({
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_size": stat.st_size,
            "file_modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "pipeline_id": str(uuid.uuid4()),
            "processing_started": datetime.datetime.now().isoformat()
        })
        
        self.logger.info(f"Loaded document: {file_path.name} ({stat.st_size} bytes)")
        
        return document


class BasicParsingStage(PipelineStage):
    """Stage to perform basic PDF parsing."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform basic PDF parsing.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with basic parsing results
        """
        file_path = document.get("file_path")
        if not file_path:
            raise ValueError("Document must contain a file_path")
        
        try:
            # Use the UnifiedPDFParser with minimal configuration
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
        """
        Perform advanced PDF parsing.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with advanced parsing results
        """
        file_path = document.get("file_path")
        if not file_path:
            raise ValueError("Document must contain a file_path")
        
        try:
            # Use the UnifiedPDFParser with full configuration
            parser_config = {
                "use_deep_learning": True,
                "extract_handwritten": True, 
                "extract_tables": True,
                "extract_forms": True,
                "extract_structure": True
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
        """
        Extract tables from a document.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with extracted tables
        """
        file_path = document.get("file_path")
        if not file_path:
            raise ValueError("Document must contain a file_path")
        
        # Configure whether to use neural or standard table extraction
        use_neural = self.config.get("use_neural", True)
        
        try:
            # Create a focused parser just for table extraction
            parser_config = {
                "use_deep_learning": use_neural,
                "extract_handwritten": False, 
                "extract_tables": True,
                "extract_forms": False,
                "extract_structure": False
            }
            
            parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on tables
            results = parser.parse(file_path)
            
            # Extract just the table data
            tables = []
            for page in results.get("pages", []):
                page_tables = page.get("tables", [])
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
        """
        Extract form elements from a document.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with extracted form elements
        """
        file_path = document.get("file_path")
        if not file_path:
            raise ValueError("Document must contain a file_path")
        
        try:
            # Create a focused parser just for form extraction
            parser_config = {
                "use_deep_learning": True,
                "extract_handwritten": False, 
                "extract_tables": False,
                "extract_forms": True,
                "extract_structure": False
            }
            
            parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on forms
            results = parser.parse(file_path)
            
            # Extract just the form data
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
        """
        Extract handwritten content from a document.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with extracted handwritten content
        """
        file_path = document.get("file_path")
        if not file_path:
            raise ValueError("Document must contain a file_path")
        
        try:
            # Create a focused parser just for handwritten content
            parser_config = {
                "use_deep_learning": True,
                "extract_handwritten": True, 
                "extract_tables": False,
                "extract_forms": False,
                "extract_structure": False
            }
            
            parser = UnifiedPDFParser(parser_config)
            
            # Parse the document, focusing on handwritten content
            results = parser.parse(file_path)
            
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
        """
        Extract document structure.
        
        Args:
            document: Document data containing a file_path
            
        Returns:
            Document data with document structure
        """
        file_path = document.get("file_path")
        if not file_path:
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
            results = parser.parse(file_path)
            
            # Extract just the structure
            structure = results.get("structure", {})
            
            # Update document with structure
            document["structure"] = structure
            
            # Count structure elements
            element_count = len(structure.get("elements", []))
            document["structure_element_count"] = element_count
            
            self.logger.info(f"Extracted document structure with {element_count} elements from: {document.get('file_name')}")
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in structure extraction: {str(e)}")
            document["structure_extraction_error"] = str(e)
            return document


class OutputFormattingStage(PipelineStage):
    """Stage to format the output of the pipeline."""
    
    def process(self, document: Dict[str, Any]) -> Dict[str, Any]:
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
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error in output formatting: {str(e)}")
            document["output_formatting_error"] = str(e)
            return document
    
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
        # Initialize document with file path
        document = {"file_path": str(file_path) if isinstance(file_path, Path) else file_path}
        
        # If no stages are defined, add default stages
        if not self.stages:
            self.logger.info("No stages defined, using default pipeline configuration")
            self._configure_default_pipeline()
        
        # Process document through pipeline stages
        for i, stage in enumerate(self.stages):
            stage_name = str(stage)
            self.logger.info(f"Running pipeline stage {i+1}/{len(self.stages)}: {stage_name}")
            
            try:
                document = stage.process(document)
            except Exception as e:
                self.logger.error(f"Error in pipeline stage {stage_name}: {str(e)}")
                document[f"error_stage_{i+1}"] = {
                    "stage": stage_name,
                    "error": str(e)
                }
        
        return document
    
    def _configure_default_pipeline(self) -> None:
        """Configure the pipeline with default stages."""
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
