"""
Table Extraction module for detecting and extracting tables from PDFs.

This module provides advanced capabilities for identifying tables in documents
and extracting their structure and content accurately.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class TableCell:
    """Represents a cell within a table."""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0
    confidence: float = 1.0
    is_header: bool = False


@dataclass
class Table:
    """Represents a table extracted from a document."""
    id: str
    page: int
    x1: float
    y1: float
    x2: float
    y2: float
    rows: int
    cols: int
    cells: List[TableCell]
    confidence: float = 1.0


class AdvancedTableExtractor:
    """
    Advanced class for detecting and extracting tables from PDFs.
    
    This class implements sophisticated algorithms for identifying tables,
    including those without explicit borders, and extracting their structure
    and content accurately.
    """
    
    def __init__(self, 
                 detect_borderless=True,
                 use_visual_features=True,
                 use_text_alignment=True,
                 detect_headers=True,
                 correct_ocr_errors=True):
        """
        Initialize the AdvancedTableExtractor.
        
        Args:
            detect_borderless: Whether to detect tables without borders
            use_visual_features: Whether to use visual features for detection
            use_text_alignment: Whether to use text alignment for detection
            detect_headers: Whether to detect table headers
            correct_ocr_errors: Whether to apply OCR error correction
        """
        self.detect_borderless = detect_borderless
        self.use_visual_features = use_visual_features
        self.use_text_alignment = use_text_alignment
        self.detect_headers = detect_headers
        self.correct_ocr_errors = correct_ocr_errors
        
    def extract_tables(self, pdf_data: Dict) -> List[Table]:
        """
        Extract tables from a PDF.
        
        Args:
            pdf_data: Parsed PDF data including text, images, and content
            
        Returns:
            List of extracted tables
        """
        tables = []
        
        # First try to detect tables using explicit borders
        if "lines" in pdf_data:
            bordered_tables = self._detect_bordered_tables(pdf_data)
            tables.extend(bordered_tables)
        
        # If borderless detection is enabled, try to detect tables using text alignment
        if self.detect_borderless and "text_blocks" in pdf_data:
            borderless_tables = self._detect_borderless_tables(pdf_data)
            
            # Only add borderless tables that don't overlap with existing tables
            for table in borderless_tables:
                if not self._overlaps_with_any(table, tables):
                    tables.append(table)
        
        # Extract content for each table
        tables_with_content = []
        for table in tables:
            table_with_content = self._extract_table_content(table, pdf_data)
            tables_with_content.append(table_with_content)
        
        # Detect headers if enabled
        if self.detect_headers:
            tables_with_headers = []
            for table in tables_with_content:
                table_with_headers = self._detect_table_headers(table)
                tables_with_headers.append(table_with_headers)
            tables_with_content = tables_with_headers
        
        # Correct OCR errors if enabled
        if self.correct_ocr_errors:
            tables_with_corrected_ocr = []
            for table in tables_with_content:
                table_with_corrected_ocr = self._correct_table_ocr_errors(table)
                tables_with_corrected_ocr.append(table_with_corrected_ocr)
            tables_with_content = tables_with_corrected_ocr
        
        return tables_with_content
    
    def _detect_bordered_tables(self, pdf_data: Dict) -> List[Table]:
        """
        Detect tables that have explicit borders.
        
        Uses line detection to identify table boundaries and cells.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected tables
        """
        tables = []
        table_id_counter = 0
        
        # Group lines by page
        lines_by_page = {}
        for line in pdf_data.get("lines", []):
            page = line.get("page", 0)
            if page not in lines_by_page:
                lines_by_page[page] = []
            lines_by_page[page].append(line)
        
        # Process each page
        for page, lines in lines_by_page.items():
            # Separate horizontal and vertical lines
            h_lines = [line for line in lines if self._is_horizontal(line)]
            v_lines = [line for line in lines if self._is_vertical(line)]
            
            # Skip if not enough lines to form a table
            if len(h_lines) < 2 or len(v_lines) < 2:
                continue
            
            # Find line intersections to identify potential tables
            tables_on_page = self._find_tables_from_lines(h_lines, v_lines, page)
            
            for table_data in tables_on_page:
                table = Table(
                    id=f"table_{table_id_counter}",
                    page=page,
                    x1=table_data["x1"],
                    y1=table_data["y1"],
                    x2=table_data["x2"],
                    y2=table_data["y2"],
                    rows=table_data["rows"],
                    cols=table_data["cols"],
                    cells=[],  # Will be filled later
                    confidence=0.9
                )
                
                tables.append(table)
                table_id_counter += 1
        
        return tables
    
    def _is_horizontal(self, line: Dict) -> bool:
        """Check if a line is horizontal."""
        x1, y1, x2, y2 = line.get("bbox", [0, 0, 0, 0])
        return abs(y2 - y1) < 0.01  # Very small vertical difference
    
    def _is_vertical(self, line: Dict) -> bool:
        """Check if a line is vertical."""
        x1, y1, x2, y2 = line.get("bbox", [0, 0, 0, 0])
        return abs(x2 - x1) < 0.01  # Very small horizontal difference
    
    def _find_tables_from_lines(self, h_lines: List[Dict], v_lines: List[Dict], page: int) -> List[Dict]:
        """
        Find tables from horizontal and vertical lines.
        
        Looks for rectangular areas formed by intersecting lines.
        
        Args:
            h_lines: Horizontal lines
            v_lines: Vertical lines
            page: Page number
            
        Returns:
            List of detected tables with basic information
        """
        # Extract line coordinates
        h_coords = [line.get("bbox", [0, 0, 0, 0])[1] for line in h_lines]  # y-coordinates
        v_coords = [line.get("bbox", [0, 0, 0, 0])[0] for line in v_lines]  # x-coordinates
        
        # Remove duplicates and sort
        h_coords = sorted(set(h_coords))
        v_coords = sorted(set(v_coords))
        
        # Skip if not enough unique coordinates to form a table
        if len(h_coords) < 2 or len(v_coords) < 2:
            return []
        
        # Check if we have a complete grid with intersecting lines
        # In a real implementation, this would be more sophisticated
        # Here we'll use a simplified approach
        
        # Find minimal rectangles that could be tables
        tables = []
        
        for i in range(len(h_coords) - 1):
            for j in range(len(v_coords) - 1):
                # Define boundaries of potential table
                y1 = h_coords[i]
                y2 = h_coords[i + 1]
                x1 = v_coords[j]
                x2 = v_coords[j + 1]
                
                # Check if this looks like a table:
                # 1. Is it large enough?
                if (x2 - x1) < 0.1 or (y2 - y1) < 0.05:
                    continue
                
                # 2. Does it have internal lines forming cells?
                internal_h_lines = [line for line in h_lines if x1 <= line.get("bbox", [0, 0, 0, 0])[0] <= x2 and 
                                    x1 <= line.get("bbox", [0, 0, 0, 0])[2] <= x2 and
                                    y1 < line.get("bbox", [0, 0, 0, 0])[1] < y2]
                                    
                internal_v_lines = [line for line in v_lines if y1 <= line.get("bbox", [0, 0, 0, 0])[1] <= y2 and 
                                    y1 <= line.get("bbox", [0, 0, 0, 0])[3] <= y2 and
                                    x1 < line.get("bbox", [0, 0, 0, 0])[0] < x2]
                
                if not internal_h_lines and not internal_v_lines:
                    continue
                
                # 3. Count rows and columns
                internal_h_coords = sorted(set([line.get("bbox", [0, 0, 0, 0])[1] for line in internal_h_lines]))
                internal_v_coords = sorted(set([line.get("bbox", [0, 0, 0, 0])[0] for line in internal_v_lines]))
                
                rows = len(internal_h_coords) + 1
                cols = len(internal_v_coords) + 1
                
                tables.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "rows": rows,
                    "cols": cols
                })
        
        return tables
    
    def _detect_borderless_tables(self, pdf_data: Dict) -> List[Table]:
        """
        Detect tables without explicit borders.
        
        Uses text alignment patterns to identify table structures.
        
        Args:
            pdf_data: Parsed PDF data
            
        Returns:
            List of detected borderless tables
        """
        tables = []
        table_id_counter = 0
        
        # Group text blocks by page
        blocks_by_page = {}
        for block in pdf_data.get("text_blocks", []):
            page = block.get("page", 0)
            if page not in blocks_by_page:
                blocks_by_page[page] = []
            blocks_by_page[page].append(block)
        
        # Process each page
        for page, blocks in blocks_by_page.items():
            # Skip if too few blocks to form a table
            if len(blocks) < 6:  # Assuming at least 2x3 table
                continue
            
            # Look for aligned text blocks that might form a table
            potential_tables = self._find_aligned_text_blocks(blocks, page)
            
            for table_data in potential_tables:
                table = Table(
                    id=f"table_{table_id_counter}",
                    page=page,
                    x1=table_data["x1"],
                    y1=table_data["y1"],
                    x2=table_data["x2"],
                    y2=table_data["y2"],
                    rows=table_data["rows"],
                    cols=table_data["cols"],
                    cells=[],  # Will be filled later
                    confidence=0.7  # Lower confidence for borderless tables
                )
                
                tables.append(table)
                table_id_counter += 1
        
        return tables
    
    def _find_aligned_text_blocks(self, blocks: List[Dict], page: int) -> List[Dict]:
        """
        Find aligned text blocks that might form a table.
        
        Args:
            blocks: Text blocks on a page
            page: Page number
            
        Returns:
            List of potential tables defined by aligned text blocks
        """
        # This is a simplified approach to detect borderless tables
        # In a real implementation, this would use more sophisticated text alignment analysis
        
        # Group blocks by y-coordinate (rows)
        row_groups = {}
        y_tolerance = 0.01  # Tolerance for y-coordinate alignment
        
        for block in blocks:
            y_center = (block.get("bbox", [0, 0, 0, 0])[1] + block.get("bbox", [0, 0, 0, 0])[3]) / 2
            rounded_y = round(y_center / y_tolerance) * y_tolerance
            
            if rounded_y not in row_groups:
                row_groups[rounded_y] = []
            
            row_groups[rounded_y].append(block)
        
        # Filter out rows with only one block
        row_groups = {y: blocks for y, blocks in row_groups.items() if len(blocks) > 1}
        
        # Skip if not enough rows
        if len(row_groups) < 2:
            return []
        
        # Find columns by aligning blocks across rows
        # In a real implementation, this would be more sophisticated
        col_groups = {}
        x_tolerance = 0.01  # Tolerance for x-coordinate alignment
        
        for row_blocks in row_groups.values():
            for block in row_blocks:
                x_center = (block.get("bbox", [0, 0, 0, 0])[0] + block.get("bbox", [0, 0, 0, 0])[2]) / 2
                rounded_x = round(x_center / x_tolerance) * x_tolerance
                
                if rounded_x not in col_groups:
                    col_groups[rounded_x] = []
                
                col_groups[rounded_x].append(block)
        
        # Filter out columns with only one block
        col_groups = {x: blocks for x, blocks in col_groups.items() if len(blocks) > 1}
        
        # Skip if not enough columns
        if len(col_groups) < 2:
            return []
        
        # Check if we have a dense enough grid of blocks
        # In a real implementation, this would involve more analysis
        
        # Find boundaries of potential table
        all_blocks = []
        for blocks in row_groups.values():
            all_blocks.extend(blocks)
        
        x1 = min(block.get("bbox", [1, 1, 1, 1])[0] for block in all_blocks)
        y1 = min(block.get("bbox", [1, 1, 1, 1])[1] for block in all_blocks)
        x2 = max(block.get("bbox", [0, 0, 0, 0])[2] for block in all_blocks)
        y2 = max(block.get("bbox", [0, 0, 0, 0])[3] for block in all_blocks)
        
        # Check if table has enough rows and columns
        if len(row_groups) < 2 or len(col_groups) < 2:
            return []
        
        # Return potential table
        return [{
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "rows": len(row_groups),
            "cols": len(col_groups)
        }]
    
    def _overlaps_with_any(self, table: Table, other_tables: List[Table]) -> bool:
        """
        Check if a table overlaps with any other table.
        
        Args:
            table: Table to check
            other_tables: List of other tables
            
        Returns:
            True if table overlaps with any other table, False otherwise
        """
        for other in other_tables:
            # Skip if tables are on different pages
            if table.page != other.page:
                continue
            
            # Check for overlap
            overlap_x = table.x1 < other.x2 and table.x2 > other.x1
            overlap_y = table.y1 < other.y2 and table.y2 > other.y1
            
            if overlap_x and overlap_y:
                return True
        
        return False
    
    def _extract_table_content(self, table: Table, pdf_data: Dict) -> Table:
        """
        Extract content for a table.
        
        Identifies cells and extracts text content.
        
        Args:
            table: Table to extract content for
            pdf_data: Parsed PDF data
            
        Returns:
            Table with extracted content
        """
        # Find cell boundaries
        cells = self._find_cells(table, pdf_data)
        
        # Extract text for each cell
        cells_with_content = []
        
        for cell in cells:
            cell_content = self._extract_cell_content(cell, table, pdf_data)
            cells_with_content.append(cell_content)
        
        # Update table with cells
        return Table(
            id=table.id,
            page=table.page,
            x1=table.x1,
            y1=table.y1,
            x2=table.x2,
            y2=table.y2,
            rows=table.rows,
            cols=table.cols,
            cells=cells_with_content,
            confidence=table.confidence
        )
    
    def _find_cells(self, table: Table, pdf_data: Dict) -> List[TableCell]:
        """
        Find cells within a table.
        
        Args:
            table: Table to find cells for
            pdf_data: Parsed PDF data
            
        Returns:
            List of cells within the table
        """
        cells = []
        
        # For a bordered table, we would use line intersections to find cells
        # For a borderless table, we would use text alignment
        
        # Simplified approach for demonstration purposes
        # In a real implementation, this would be more sophisticated
        
        # Create a uniform grid
        row_height = (table.y2 - table.y1) / table.rows
        col_width = (table.x2 - table.x1) / table.cols
        
        for row in range(table.rows):
            for col in range(table.cols):
                # Calculate cell boundaries
                x1 = table.x1 + col * col_width
                y1 = table.y1 + row * row_height
                x2 = x1 + col_width
                y2 = y1 + row_height
                
                cell = TableCell(
                    row=row,
                    col=col,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2
                )
                
                cells.append(cell)
        
        return cells
    
    def _extract_cell_content(self, cell: TableCell, table: Table, pdf_data: Dict) -> TableCell:
        """
        Extract text content for a cell.
        
        Args:
            cell: Cell to extract content for
            table: Table containing the cell
            pdf_data: Parsed PDF data
            
        Returns:
            Cell with extracted content
        """
        # Find text blocks that overlap with this cell
        cell_content = ""
        
        if "text_blocks" in pdf_data:
            for block in pdf_data.get("text_blocks", []):
                # Skip if not on the same page
                if block.get("page", 0) != table.page:
                    continue
                
                # Extract coordinates
                block_x1, block_y1, block_x2, block_y2 = block.get("bbox", [0, 0, 0, 0])
                
                # Check if block overlaps with cell
                overlap_x = block_x1 < cell.x2 and block_x2 > cell.x1
                overlap_y = block_y1 < cell.y2 and block_y2 > cell.y1
                
                if overlap_x and overlap_y:
                    # Add text to cell content
                    if cell_content:
                        cell_content += " "
                    cell_content += block.get("text", "")
        
        # Update cell with content
        return TableCell(
            row=cell.row,
            col=cell.col,
            row_span=cell.row_span,
            col_span=cell.col_span,
            text=cell_content,
            x1=cell.x1,
            y1=cell.y1,
            x2=cell.x2,
            y2=cell.y2,
            confidence=cell.confidence,
            is_header=cell.is_header
        )
    
    def _detect_table_headers(self, table: Table) -> Table:
        """
        Detect headers in a table.
        
        Args:
            table: Table to detect headers for
            
        Returns:
            Table with header information
        """
        cells_with_headers = []
        
        # Simple heuristic: assume first row is header
        first_row_cells = [cell for cell in table.cells if cell.row == 0]
        
        for cell in table.cells:
            is_header = (cell.row == 0)  # Simple heuristic
            
            # Update cell with header information
            updated_cell = TableCell(
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
                text=cell.text,
                x1=cell.x1,
                y1=cell.y1,
                x2=cell.x2,
                y2=cell.y2,
                confidence=cell.confidence,
                is_header=is_header
            )
            
            cells_with_headers.append(updated_cell)
        
        # Update table with cells
        return Table(
            id=table.id,
            page=table.page,
            x1=table.x1,
            y1=table.y1,
            x2=table.x2,
            y2=table.y2,
            rows=table.rows,
            cols=table.cols,
            cells=cells_with_headers,
            confidence=table.confidence
        )
    
    def _correct_table_ocr_errors(self, table: Table) -> Table:
        """
        Correct OCR errors in table text.
        
        Args:
            table: Table to correct OCR errors for
            
        Returns:
            Table with corrected text
        """
        # This would implement OCR error correction techniques
        # For example, using language models, dictionaries, or regex patterns
        
        # Simplified approach for demonstration purposes
        corrected_cells = []
        
        for cell in table.cells:
            # Apply simple corrections
            corrected_text = cell.text
            
            # Example corrections (in a real implementation, this would be more sophisticated)
            corrected_text = corrected_text.replace("l", "1").replace("O", "0")
            
            # Update cell with corrected text
            corrected_cell = TableCell(
                row=cell.row,
                col=cell.col,
                row_span=cell.row_span,
                col_span=cell.col_span,
                text=corrected_text,
                x1=cell.x1,
                y1=cell.y1,
                x2=cell.x2,
                y2=cell.y2,
                confidence=cell.confidence,
                is_header=cell.is_header
            )
            
            corrected_cells.append(corrected_cell)
        
        # Update table with corrected cells
        return Table(
            id=table.id,
            page=table.page,
            x1=table.x1,
            y1=table.y1,
            x2=table.x2,
            y2=table.y2,
            rows=table.rows,
            cols=table.cols,
            cells=corrected_cells,
            confidence=table.confidence
        )
    
    def table_to_markdown(self, table: Table) -> str:
        """
        Convert a table to Markdown format.
        
        Args:
            table: Table to convert
            
        Returns:
            Markdown representation of the table
        """
        if not table.cells:
            return ""
        
        # Get number of rows and columns
        rows = table.rows
        cols = table.cols
        
        # Create empty grid
        grid = [['' for _ in range(cols)] for _ in range(rows)]
        
        # Fill grid with cell content
        for cell in table.cells:
            grid[cell.row][cell.col] = cell.text
        
        # Generate Markdown
        md = []
        
        # Header row
        md.append('| ' + ' | '.join(grid[0]) + ' |')
        
        # Header separator
        md.append('| ' + ' | '.join(['---' for _ in range(cols)]) + ' |')
        
        # Data rows
        for row in range(1, rows):
            md.append('| ' + ' | '.join(grid[row]) + ' |')
        
        return '\n'.join(md)
