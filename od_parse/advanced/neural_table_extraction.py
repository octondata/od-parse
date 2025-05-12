"""
Neural network-based table extraction module for complex tables in PDFs.

This module implements state-of-the-art deep learning approaches for detecting
and extracting tables from documents, including tables without clear borders,
tables with complex cell structures, and tables with spanning cells.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import os
import logging
import json

# Optional dependencies that degrade gracefully if not present
try:
    import torch
    import torchvision
    import cv2
    from transformers import TableTransformerForObjectDetection, TableTransformerProcessor
    HAVE_DL_DEPS = True
except ImportError:
    HAVE_DL_DEPS = False


@dataclass
class NeuralTableCell:
    """Represents a cell in a table detected by neural networks."""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    confidence: float = 0.0
    bbox: List[float] = None  # [x1, y1, x2, y2] normalized to 0-1


@dataclass
class NeuralTable:
    """Represents a table detected by neural networks."""
    bbox: List[float]  # [x1, y1, x2, y2] normalized to 0-1
    page: int
    confidence: float
    rows: int = 0
    cols: int = 0
    cells: List[NeuralTableCell] = None
    structure_type: str = "unknown"  # "bordered", "borderless", "partial_borders", etc.


class NeuralTableExtractor:
    """
    Advanced table extraction using neural networks.
    
    This class uses state-of-the-art deep learning models to detect and extract
    tables from PDF documents, handling complex cases including borderless tables,
    tables with spanning cells, and tables with complex structures.
    """
    
    def __init__(self, 
                 model_name: str = "microsoft/table-transformer-detection",
                 structure_model_name: str = "microsoft/table-transformer-structure-recognition",
                 use_gpu: bool = False,
                 confidence_threshold: float = 0.7):
        """
        Initialize the neural table extractor.
        
        Args:
            model_name: Pre-trained table detection model
            structure_model_name: Pre-trained table structure recognition model
            use_gpu: Whether to use GPU acceleration if available
            confidence_threshold: Minimum confidence for detection
        """
        self.model_name = model_name
        self.structure_model_name = structure_model_name
        self.use_gpu = use_gpu and torch.cuda.is_available() if HAVE_DL_DEPS else False
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if self.use_gpu else "cpu") if HAVE_DL_DEPS else None
        
        # Initialize models if dependencies are available
        self.detection_model = None
        self.detection_processor = None
        self.structure_model = None
        self.structure_processor = None
        
        if HAVE_DL_DEPS:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the table detection and structure recognition models."""
        try:
            # Initialize table detection model
            self.detection_processor = TableTransformerProcessor.from_pretrained(self.model_name)
            self.detection_model = TableTransformerForObjectDetection.from_pretrained(self.model_name)
            self.detection_model.to(self.device)
            self.detection_model.eval()
            
            # Initialize table structure recognition model
            self.structure_processor = TableTransformerProcessor.from_pretrained(self.structure_model_name)
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(self.structure_model_name)
            self.structure_model.to(self.device)
            self.structure_model.eval()
            
            logging.info("Successfully initialized neural table extraction models")
        except Exception as e:
            logging.error(f"Error initializing neural table extraction models: {str(e)}")
            self.detection_model = None
            self.structure_model = None
    
    def extract_tables(self, image_path: str, page_num: int = 0) -> List[NeuralTable]:
        """
        Extract tables from a document image using neural networks.
        
        Args:
            image_path: Path to the document image
            page_num: Page number in the original document
            
        Returns:
            List of detected tables with their structure
        """
        if not HAVE_DL_DEPS or self.detection_model is None:
            return self._fallback_extraction(image_path, page_num)
        
        try:
            # Step 1: Detect tables in the image
            tables = self._detect_tables(image_path, page_num)
            
            # Step 2: For each detected table, recognize its structure
            for table in tables:
                self._recognize_table_structure(image_path, table)
                
            # Step 3: Extract cell contents using OCR
            for table in tables:
                if table.cells:
                    self._extract_cell_contents(image_path, table)
            
            return tables
        
        except Exception as e:
            logging.error(f"Error in neural table extraction: {str(e)}")
            return self._fallback_extraction(image_path, page_num)
    
    def _detect_tables(self, image_path: str, page_num: int) -> List[NeuralTable]:
        """
        Detect tables in an image using the table detection model.
        
        Args:
            image_path: Path to the document image
            page_num: Page number in the original document
            
        Returns:
            List of detected tables (without structure)
        """
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Process image with table detection model
        inputs = self.detection_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.detection_model(**inputs)
        
        # Convert outputs to table objects
        tables = []
        
        # Process predictions (normalized bounding boxes, labels, scores)
        for box, label, score in zip(
            outputs.pred_boxes[0].cpu().numpy(),
            outputs.pred_labels[0].cpu().numpy(),
            outputs.scores[0].cpu().numpy()
        ):
            # Filter low confidence detections
            if score < self.confidence_threshold:
                continue
            
            # Get label string from model id
            label_str = self.detection_processor.tokenizer.decode(label)
            
            # Create table object
            table = NeuralTable(
                bbox=box.tolist(),  # Already normalized
                page=page_num,
                confidence=float(score),
                cells=[]
            )
            
            tables.append(table)
        
        return tables
    
    def _recognize_table_structure(self, image_path: str, table: NeuralTable) -> None:
        """
        Recognize the structure of a detected table.
        
        Args:
            image_path: Path to the document image
            table: Detected table object to update with structure
        """
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        # Crop to table bounding box
        x1, y1, x2, y2 = table.bbox
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # Ensure valid crop area
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        # Crop table region
        table_image = image.crop((x1, y1, x2, y2))
        
        # Process image with table structure recognition model
        inputs = self.structure_processor(images=table_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.structure_model(**inputs)
        
        # Process structure predictions
        rows = []
        columns = []
        cells = []
        
        # Extract row and column separators
        for box, label, score in zip(
            outputs.pred_boxes[0].cpu().numpy(),
            outputs.pred_labels[0].cpu().numpy(),
            outputs.scores[0].cpu().numpy()
        ):
            # Filter low confidence detections
            if score < self.confidence_threshold:
                continue
            
            # Get label string from model id
            label_str = self.structure_processor.tokenizer.decode(label)
            
            if "row" in label_str.lower():
                rows.append((box[1], box))  # y-coordinate, full box
            elif "column" in label_str.lower():
                columns.append((box[0], box))  # x-coordinate, full box
            elif "cell" in label_str.lower():
                cells.append((box, score))  # box, confidence
        
        # Sort rows and columns by position
        rows.sort()
        columns.sort()
        
        # Update table with structure information
        table.rows = len(rows) + 1  # +1 because n separators create n+1 rows
        table.cols = len(columns) + 1  # +1 because n separators create n+1 columns
        
        # Determine structure type based on detection pattern
        borders_detected = len(rows) > 0 and len(columns) > 0
        
        if borders_detected:
            if len(rows) >= table.rows - 1 and len(columns) >= table.cols - 1:
                table.structure_type = "bordered"
            else:
                table.structure_type = "partial_borders"
        else:
            table.structure_type = "borderless"
        
        # If cells were directly detected, create cell objects
        table.cells = []
        
        if cells:
            # When cells are directly detected
            for cell_box, confidence in cells:
                # Normalize cell coordinates to table coordinates
                cell_box_norm = [
                    cell_box[0],
                    cell_box[1],
                    cell_box[2],
                    cell_box[3]
                ]
                
                # Determine approximate row and column
                row = 0
                col = 0
                
                # Find which row this cell belongs to
                for i, (y, _) in enumerate(rows):
                    if cell_box[1] < y:
                        row = i
                        break
                else:
                    row = len(rows)
                
                # Find which column this cell belongs to
                for i, (x, _) in enumerate(columns):
                    if cell_box[0] < x:
                        col = i
                        break
                else:
                    col = len(columns)
                
                # Create cell object
                cell = NeuralTableCell(
                    row=row,
                    col=col,
                    bbox=cell_box_norm,
                    confidence=confidence
                )
                
                table.cells.append(cell)
        else:
            # When cells must be inferred from row/column separators
            # Create a grid of cells
            for row in range(table.rows):
                for col in range(table.cols):
                    # Calculate cell bounds from separators
                    x1 = 0 if col == 0 else columns[col-1][1][0]
                    y1 = 0 if row == 0 else rows[row-1][1][1]
                    x2 = 1 if col == table.cols-1 else columns[col][1][0]
                    y2 = 1 if row == table.rows-1 else rows[row][1][1]
                    
                    cell = NeuralTableCell(
                        row=row,
                        col=col,
                        bbox=[x1, y1, x2, y2],
                        confidence=0.9  # Default confidence for grid cells
                    )
                    
                    table.cells.append(cell)
    
    def _extract_cell_contents(self, image_path: str, table: NeuralTable) -> None:
        """
        Extract text content from table cells using OCR.
        
        Args:
            image_path: Path to the document image
            table: Table with structure to extract cell contents for
        """
        # Here we would use OCR to extract the text content of each cell
        # For simplicity, we'll assume this is done with a separate OCR module
        # In a real implementation, this would integrate with a proper OCR engine
        
        # This is a placeholder that would be replaced with actual OCR
        for cell in table.cells:
            cell.text = f"Cell content {cell.row},{cell.col}"
    
    def _fallback_extraction(self, image_path: str, page_num: int) -> List[NeuralTable]:
        """
        Fallback table extraction using classical computer vision.
        
        Args:
            image_path: Path to the document image
            page_num: Page number in the original document
            
        Returns:
            List of detected tables with basic structure
        """
        try:
            import cv2
            
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return []
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = img.shape[:2]
            
            # Threshold the image
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Detect lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Find table contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            
            for i, contour in enumerate(contours):
                if cv2.contourArea(contour) < 5000:  # Filter small areas
                    continue
                    
                x, y, w, h = cv2.boundingRect(contour)
                
                # Normalize coordinates
                bbox = [
                    float(x) / width,
                    float(y) / height,
                    float(x + w) / width,
                    float(y + h) / height
                ]
                
                # Create table object
                table = NeuralTable(
                    bbox=bbox,
                    page=page_num,
                    confidence=0.8,
                    structure_type="bordered"
                )
                
                # Detect cells
                self._detect_cells_cv(img, table, x, y, w, h)
                
                tables.append(table)
            
            # If no tables with lines were found, try to detect tables without explicit lines
            if not tables:
                tables = self._detect_borderless_tables_cv(img, page_num)
            
            return tables
            
        except Exception as e:
            logging.error(f"Error in fallback table extraction: {str(e)}")
            return []
    
    def _detect_cells_cv(self, img, table: NeuralTable, x: int, y: int, w: int, h: int) -> None:
        """
        Detect cells in a table using OpenCV.
        
        Args:
            img: Input image
            table: Table object to update
            x, y, w, h: Table bounding box in pixels
        """
        try:
            height, width = img.shape[:2]
            
            # Extract table region
            table_roi = img[y:y+h, x:x+w]
            
            # Convert to grayscale
            gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
            
            # Threshold
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            
            # Detect lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
            
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find intersection points
            joints = cv2.bitwise_and(horizontal_lines, vertical_lines)
            
            # Find contours of intersection points
            contours, _ = cv2.findContours(joints, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract intersection points
            points = []
            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    points.append((cx, cy))
            
            # Sort points by position
            points.sort(key=lambda p: (p[1], p[0]))  # Sort by y, then x
            
            # Find unique y-coordinates (rows)
            y_coords = sorted(set(p[1] for p in points))
            
            # Find unique x-coordinates (columns)
            x_coords = sorted(set(p[0] for p in points))
            
            # Update table structure
            table.rows = len(y_coords) - 1
            table.cols = len(x_coords) - 1
            
            # Create cells
            table.cells = []
            
            for i in range(len(y_coords) - 1):
                for j in range(len(x_coords) - 1):
                    # Cell coordinates in table roi
                    cell_x1 = x_coords[j]
                    cell_y1 = y_coords[i]
                    cell_x2 = x_coords[j + 1]
                    cell_y2 = y_coords[i + 1]
                    
                    # Normalize to 0-1 range for the whole image
                    norm_x1 = float(x + cell_x1) / width
                    norm_y1 = float(y + cell_y1) / height
                    norm_x2 = float(x + cell_x2) / width
                    norm_y2 = float(y + cell_y2) / height
                    
                    cell = NeuralTableCell(
                        row=i,
                        col=j,
                        bbox=[norm_x1, norm_y1, norm_x2, norm_y2],
                        confidence=0.9
                    )
                    
                    table.cells.append(cell)
                    
        except Exception as e:
            logging.error(f"Error detecting cells with OpenCV: {str(e)}")
    
    def _detect_borderless_tables_cv(self, img, page_num: int) -> List[NeuralTable]:
        """
        Detect borderless tables using text alignment.
        
        Args:
            img: Input image
            page_num: Page number
            
        Returns:
            List of detected borderless tables
        """
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find text
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find text contours
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter and group text contours
        text_blocks = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Filter very small contours
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            text_blocks.append((x, y, x+w, y+h))
        
        # If very few blocks, no tables
        if len(text_blocks) < 6:  # Need at least a small grid
            return []
        
        # Find aligned blocks
        # Group by similar y-coordinates
        y_tolerance = height * 0.01  # 1% of height
        row_groups = {}
        
        for block in text_blocks:
            _, y1, _, y2 = block
            center_y = (y1 + y2) / 2
            
            # Find or create row group
            assigned = False
            for group_y in list(row_groups.keys()):
                if abs(center_y - group_y) < y_tolerance:
                    row_groups[group_y].append(block)
                    assigned = True
                    break
            
            if not assigned:
                row_groups[center_y] = [block]
        
        # Filter to groups with multiple blocks (potential table rows)
        row_groups = {y: blocks for y, blocks in row_groups.items() if len(blocks) > 1}
        
        # Need multiple rows to form a table
        if len(row_groups) < 2:
            return []
        
        # Look for aligned columns
        x_tolerance = width * 0.01  # 1% of width
        column_groups = {}
        
        for y, blocks in row_groups.items():
            for block in blocks:
                x1, _, x2, _ = block
                center_x = (x1 + x2) / 2
                
                # Find or create column group
                assigned = False
                for group_x in list(column_groups.keys()):
                    if abs(center_x - group_x) < x_tolerance:
                        column_groups[group_x].append(block)
                        assigned = True
                        break
                
                if not assigned:
                    column_groups[center_x] = [block]
        
        # Filter to groups with multiple blocks (potential table columns)
        column_groups = {x: blocks for x, blocks in column_groups.items() if len(blocks) > 1}
        
        # Need multiple columns to form a table
        if len(column_groups) < 2:
            return []
        
        # Sort centers
        row_centers = sorted(row_groups.keys())
        column_centers = sorted(column_groups.keys())
        
        # Find table boundaries
        all_blocks = []
        for blocks in row_groups.values():
            all_blocks.extend(blocks)
        
        min_x = min(block[0] for block in all_blocks)
        max_x = max(block[2] for block in all_blocks)
        min_y = min(block[1] for block in all_blocks)
        max_y = max(block[3] for block in all_blocks)
        
        # Create table
        table = NeuralTable(
            bbox=[
                float(min_x) / width,
                float(min_y) / height,
                float(max_x) / width,
                float(max_y) / height
            ],
            page=page_num,
            confidence=0.7,
            rows=len(row_centers),
            cols=len(column_centers),
            structure_type="borderless"
        )
        
        # Create cells
        table.cells = []
        
        # Create cell grid
        for i in range(len(row_centers)):
            row_y = row_centers[i]
            next_row_y = row_centers[i+1] if i+1 < len(row_centers) else max_y
            
            for j in range(len(column_centers)):
                col_x = column_centers[j]
                next_col_x = column_centers[j+1] if j+1 < len(column_centers) else max_x
                
                # Find blocks in this cell
                cell_blocks = []
                for block in all_blocks:
                    block_x1, block_y1, block_x2, block_y2 = block
                    block_center_x = (block_x1 + block_x2) / 2
                    block_center_y = (block_y1 + block_y2) / 2
                    
                    if (abs(block_center_y - row_y) < y_tolerance and 
                        abs(block_center_x - col_x) < x_tolerance):
                        cell_blocks.append(block)
                
                # Use actual block bounds if found, otherwise use grid cell bounds
                if cell_blocks:
                    cell_x1 = min(block[0] for block in cell_blocks)
                    cell_y1 = min(block[1] for block in cell_blocks)
                    cell_x2 = max(block[2] for block in cell_blocks)
                    cell_y2 = max(block[3] for block in cell_blocks)
                else:
                    # Approximate cell bounds
                    prev_row_y = row_centers[i-1] if i > 0 else min_y
                    prev_col_x = column_centers[j-1] if j > 0 else min_x
                    
                    cell_x1 = prev_col_x + (col_x - prev_col_x) / 2
                    cell_y1 = prev_row_y + (row_y - prev_row_y) / 2
                    cell_x2 = col_x + (next_col_x - col_x) / 2
                    cell_y2 = row_y + (next_row_y - row_y) / 2
                
                # Normalize coordinates
                norm_x1 = float(cell_x1) / width
                norm_y1 = float(cell_y1) / height
                norm_x2 = float(cell_x2) / width
                norm_y2 = float(cell_y2) / height
                
                cell = NeuralTableCell(
                    row=i,
                    col=j,
                    bbox=[norm_x1, norm_y1, norm_x2, norm_y2],
                    confidence=0.7
                )
                
                table.cells.append(cell)
        
        return [table]
    
    def table_to_markdown(self, table: NeuralTable) -> str:
        """
        Convert a table to Markdown format.
        
        Args:
            table: Table object
            
        Returns:
            Markdown representation of the table
        """
        if not table.cells or table.rows <= 0 or table.cols <= 0:
            return ""
        
        # Create empty grid
        grid = [["" for _ in range(table.cols)] for _ in range(table.rows)]
        
        # Fill grid with cell content
        for cell in table.cells:
            row, col = cell.row, cell.col
            if 0 <= row < table.rows and 0 <= col < table.cols:
                grid[row][col] = cell.text or f"R{row+1}C{col+1}"
        
        # Generate Markdown
        md = []
        
        # Header row
        md.append('| ' + ' | '.join(grid[0]) + ' |')
        
        # Header separator
        md.append('| ' + ' | '.join(['---' for _ in range(table.cols)]) + ' |')
        
        # Data rows
        for row in range(1, table.rows):
            md.append('| ' + ' | '.join(grid[row]) + ' |')
        
        return '\n'.join(md)
    
    def table_to_html(self, table: NeuralTable) -> str:
        """
        Convert a table to HTML format.
        
        Args:
            table: Table object
            
        Returns:
            HTML representation of the table
        """
        if not table.cells or table.rows <= 0 or table.cols <= 0:
            return ""
        
        # Create empty grid
        grid = [["" for _ in range(table.cols)] for _ in range(table.rows)]
        
        # Fill grid with cell content
        for cell in table.cells:
            row, col = cell.row, cell.col
            if 0 <= row < table.rows and 0 <= col < table.cols:
                grid[row][col] = cell.text or f"R{row+1}C{col+1}"
        
        # Generate HTML
        html = ['<table border="1">']
        
        # Add thead
        html.append('<thead>')
        html.append('<tr>')
        for cell in grid[0]:
            html.append(f'<th>{cell}</th>')
        html.append('</tr>')
        html.append('</thead>')
        
        # Add tbody
        html.append('<tbody>')
        for row in range(1, table.rows):
            html.append('<tr>')
            for cell in grid[row]:
                html.append(f'<td>{cell}</td>')
            html.append('</tr>')
        html.append('</tbody>')
        
        html.append('</table>')
        return '\n'.join(html)
    
    def table_to_json(self, table: NeuralTable) -> str:
        """
        Convert a table to JSON format.
        
        Args:
            table: Table object
            
        Returns:
            JSON representation of the table
        """
        if not table.cells or table.rows <= 0 or table.cols <= 0:
            return "{}"
        
        # Create empty grid
        grid = [["" for _ in range(table.cols)] for _ in range(table.rows)]
        
        # Fill grid with cell content
        for cell in table.cells:
            row, col = cell.row, cell.col
            if 0 <= row < table.rows and 0 <= col < table.cols:
                grid[row][col] = cell.text or f"R{row+1}C{col+1}"
        
        # Headers
        headers = grid[0]
        
        # Data rows
        data = []
        for row in range(1, table.rows):
            row_data = {}
            for col in range(table.cols):
                row_data[headers[col]] = grid[row][col]
            data.append(row_data)
        
        return json.dumps(data, indent=2)
