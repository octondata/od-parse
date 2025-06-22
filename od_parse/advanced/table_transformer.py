"""
Table Transformer Integration

This module provides Table Transformer integration for superior table detection
and extraction, with fallback to traditional table extraction methods.
"""

import os
import warnings
from typing import Union, Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from od_parse.config import get_advanced_config
from od_parse.utils.logging_utils import get_logger

logger = get_logger(__name__)


class TableTransformerEngine:
    """
    Table Transformer-based table detection and extraction engine.
    
    This class provides advanced table detection and extraction using Microsoft's
    Table Transformer models, with automatic fallback to traditional methods
    when Table Transformer dependencies are not available.
    """
    
    def __init__(self, 
                 detection_model: str = "microsoft/table-transformer-detection",
                 structure_model: str = "microsoft/table-transformer-structure-recognition",
                 device: str = "auto"):
        """
        Initialize the Table Transformer engine.
        
        Args:
            detection_model: Model for table detection
            structure_model: Model for table structure recognition
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.logger = get_logger(__name__)
        self.detection_model_name = detection_model
        self.structure_model_name = structure_model
        self.device = device
        
        # Model components
        self.detection_processor = None
        self.detection_model = None
        self.structure_processor = None
        self.structure_model = None
        
        self._is_available = False
        self._fallback_engine = None
        
        # Initialize Table Transformer
        self._initialize_table_transformer()
        
        # Initialize fallback engine if Table Transformer is not available
        if not self._is_available:
            self._initialize_fallback()
    
    def _initialize_table_transformer(self) -> bool:
        """Initialize Table Transformer models."""
        config = get_advanced_config()
        
        if not config.is_feature_enabled("table_transformer"):
            self.logger.info("Table Transformer feature is disabled. Use config.enable_feature('table_transformer') to enable.")
            return False
        
        if not config.is_feature_available("table_transformer"):
            self.logger.warning("Table Transformer dependencies not available. Install with: pip install od-parse[table_transformer]")
            return False
        
        try:
            from transformers import DetrImageProcessor, TableTransformerForObjectDetection
            import torch
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.logger.info(f"Loading Table Transformer models on {self.device}")
            
            # Load detection model
            self.detection_processor = DetrImageProcessor.from_pretrained(self.detection_model_name)
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(self.detection_model_name)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            
            # Load structure recognition model
            self.structure_processor = DetrImageProcessor.from_pretrained(self.structure_model_name)
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(self.structure_model_name)
            self.structure_model.to(self.device)
            self.structure_model.eval()
            
            self._is_available = True
            self.logger.info("Table Transformer initialized successfully")
            return True
            
        except ImportError as e:
            self.logger.warning(f"Table Transformer dependencies not available: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Table Transformer: {e}")
            return False
    
    def _initialize_fallback(self):
        """Initialize fallback table extraction engine."""
        try:
            import tabula
            self._fallback_engine = tabula
            self.logger.info("Initialized fallback table extraction engine (tabula-py)")
        except ImportError:
            self.logger.error("Neither Table Transformer nor tabula-py is available")
    
    def extract_tables(
        self, 
        image: Union[str, Path, Image.Image, np.ndarray],
        confidence_threshold: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Extract tables from an image using Table Transformer or fallback engine.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            confidence_threshold: Minimum confidence for table detection
            **kwargs: Additional arguments for table extraction
            
        Returns:
            Dictionary containing extracted tables and metadata
        """
        try:
            # Convert input to PIL Image
            pil_image = self._prepare_image(image)
            
            if self._is_available:
                return self._extract_with_table_transformer(pil_image, confidence_threshold, **kwargs)
            else:
                return self._extract_with_fallback(image, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error extracting tables: {e}")
            return {
                "tables": [],
                "confidence": 0.0,
                "engine": "error",
                "error": str(e)
            }
    
    def _prepare_image(self, image: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _extract_with_table_transformer(
        self, 
        image: Image.Image, 
        confidence_threshold: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Extract tables using Table Transformer."""
        try:
            import torch
            
            # Step 1: Detect tables
            table_detections = self._detect_tables(image, confidence_threshold)
            
            if not table_detections:
                return {
                    "tables": [],
                    "confidence": 0.0,
                    "engine": "table_transformer",
                    "detection_model": self.detection_model_name,
                    "structure_model": self.structure_model_name
                }
            
            # Step 2: Extract structure for each detected table
            extracted_tables = []
            for i, detection in enumerate(table_detections):
                try:
                    # Crop table region
                    bbox = detection["bbox"]
                    table_image = self._crop_table(image, bbox)
                    
                    # Extract table structure
                    structure = self._extract_table_structure(table_image)
                    
                    # Combine detection and structure information
                    table_data = {
                        "table_id": i,
                        "bbox": bbox,
                        "confidence": detection["confidence"],
                        "structure": structure,
                        "cells": structure.get("cells", []),
                        "rows": structure.get("rows", []),
                        "columns": structure.get("columns", [])
                    }
                    
                    extracted_tables.append(table_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing table {i}: {e}")
                    continue
            
            # Calculate overall confidence
            avg_confidence = sum(t["confidence"] for t in extracted_tables) / len(extracted_tables) if extracted_tables else 0
            
            return {
                "tables": extracted_tables,
                "confidence": avg_confidence,
                "engine": "table_transformer",
                "detection_model": self.detection_model_name,
                "structure_model": self.structure_model_name,
                "device": self.device
            }
            
        except Exception as e:
            self.logger.error(f"Table Transformer extraction failed: {e}")
            # Fallback to traditional table extraction
            return self._extract_with_fallback(image, **kwargs)
    
    def _detect_tables(self, image: Image.Image, confidence_threshold: float) -> List[Dict[str, Any]]:
        """Detect tables in the image."""
        import torch
        
        # Prepare inputs
        inputs = self.detection_processor(image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run detection
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)  # (height, width)
        results = self.detection_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )[0]
        
        # Convert to list of detections
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > confidence_threshold:
                detections.append({
                    "bbox": box.cpu().numpy().tolist(),  # [x1, y1, x2, y2]
                    "confidence": score.cpu().item(),
                    "label": label.cpu().item()
                })
        
        return detections
    
    def _crop_table(self, image: Image.Image, bbox: List[float]) -> Image.Image:
        """Crop table region from image."""
        x1, y1, x2, y2 = bbox
        return image.crop((x1, y1, x2, y2))
    
    def _extract_table_structure(self, table_image: Image.Image) -> Dict[str, Any]:
        """Extract table structure (cells, rows, columns)."""
        import torch
        
        # Prepare inputs
        inputs = self.structure_processor(table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run structure recognition
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        
        # Process outputs
        target_sizes = torch.tensor([table_image.size[::-1]]).to(self.device)
        results = self.structure_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.6
        )[0]
        
        # Organize structure elements
        cells = []
        rows = []
        columns = []
        
        # Map label IDs to structure types (this may vary by model)
        label_map = {
            0: "table",
            1: "table column",
            2: "table row", 
            3: "table column header",
            4: "table projected row header",
            5: "table spanning cell"
        }
        
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            element = {
                "bbox": box.cpu().numpy().tolist(),
                "confidence": score.cpu().item(),
                "label_id": label.cpu().item(),
                "label": label_map.get(label.cpu().item(), "unknown")
            }
            
            # Categorize by type
            if "column" in element["label"]:
                columns.append(element)
            elif "row" in element["label"]:
                rows.append(element)
            elif "cell" in element["label"]:
                cells.append(element)
        
        return {
            "cells": cells,
            "rows": rows,
            "columns": columns,
            "structure_confidence": float(torch.mean(results["scores"]).cpu()) if len(results["scores"]) > 0 else 0.0
        }
    
    def _extract_with_fallback(self, image: Union[str, Path, Image.Image], **kwargs) -> Dict[str, Any]:
        """Extract tables using fallback engine (tabula-py)."""
        if self._fallback_engine is None:
            return {
                "tables": [],
                "confidence": 0.0,
                "engine": "none",
                "error": "No table extraction engine available"
            }
        
        try:
            # For fallback, we need a PDF file path
            # This is a limitation of tabula-py
            if isinstance(image, (str, Path)):
                # Assume it's a PDF file
                tables = self._fallback_engine.read_pdf(str(image), pages='all')
                
                extracted_tables = []
                for i, df in enumerate(tables):
                    table_data = {
                        "table_id": i,
                        "data": df.to_dict('records'),
                        "columns": df.columns.tolist(),
                        "shape": df.shape,
                        "confidence": 0.8  # Default confidence for tabula
                    }
                    extracted_tables.append(table_data)
                
                return {
                    "tables": extracted_tables,
                    "confidence": 0.8,
                    "engine": "tabula",
                    "note": "Fallback engine used - limited to PDF files"
                }
            else:
                return {
                    "tables": [],
                    "confidence": 0.0,
                    "engine": "tabula",
                    "error": "Fallback engine requires PDF file path"
                }
                
        except Exception as e:
            self.logger.error(f"Fallback table extraction failed: {e}")
            return {
                "tables": [],
                "confidence": 0.0,
                "engine": "tabula",
                "error": str(e)
            }
    
    def visualize_detections(
        self, 
        image: Union[str, Path, Image.Image], 
        detections: List[Dict[str, Any]]
    ) -> Image.Image:
        """
        Visualize table detections on the image.
        
        Args:
            image: Input image
            detections: List of detection results
            
        Returns:
            Image with detection boxes drawn
        """
        pil_image = self._prepare_image(image)
        draw = ImageDraw.Draw(pil_image)
        
        for detection in detections:
            bbox = detection["bbox"]
            confidence = detection.get("confidence", 0)
            
            # Draw bounding box
            draw.rectangle(bbox, outline="red", width=2)
            
            # Draw confidence score
            draw.text((bbox[0], bbox[1] - 20), f"{confidence:.2f}", fill="red")
        
        return pil_image
    
    def is_available(self) -> bool:
        """Check if Table Transformer is available."""
        return self._is_available
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the current table extraction engine."""
        return {
            "table_transformer_available": self._is_available,
            "detection_model": self.detection_model_name if self._is_available else None,
            "structure_model": self.structure_model_name if self._is_available else None,
            "device": self.device if self._is_available else None,
            "fallback_available": self._fallback_engine is not None,
            "current_engine": "table_transformer" if self._is_available else "tabula"
        }


# Convenience function for easy usage
def extract_tables_with_transformer(
    image: Union[str, Path, Image.Image, np.ndarray],
    confidence_threshold: float = 0.7,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to extract tables using Table Transformer.
    
    Args:
        image: Input image
        confidence_threshold: Minimum confidence for detection
        **kwargs: Additional arguments
        
    Returns:
        Table extraction result dictionary
    """
    engine = TableTransformerEngine()
    return engine.extract_tables(image, confidence_threshold, **kwargs)
