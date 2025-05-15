"""
Integration module for connecting advanced parsing capabilities with external systems.

This module provides connectors and adapters to integrate the advanced PDF parsing
capabilities with various external systems and formats.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.utils.logging_utils import get_logger


class DataConnector:
    """Base class for data connectors to external systems."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the connector with configuration."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
    
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data to external system.
        
        Args:
            data: Data to export
            
        Returns:
            Success status
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def import_data(self) -> Dict[str, Any]:
        """
        Import data from external system.
        
        Returns:
            Imported data
        """
        raise NotImplementedError("Subclasses must implement this method")


class JSONFileConnector(DataConnector):
    """Connector for exporting to JSON files."""
    
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data to a JSON file.
        
        Args:
            data: Data to export
            
        Returns:
            Success status
        """
        file_path = self.config.get("file_path")
        if not file_path:
            self.logger.error("No file path specified in configuration")
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Data exported to JSON file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON file: {str(e)}")
            return False
    
    def import_data(self) -> Dict[str, Any]:
        """
        Import data from a JSON file.
        
        Returns:
            Imported data
        """
        file_path = self.config.get("file_path")
        if not file_path:
            self.logger.error("No file path specified in configuration")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Data imported from JSON file: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error importing from JSON file: {str(e)}")
            return {}


class CSVConnector(DataConnector):
    """Connector for exporting to CSV files."""
    
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data to a CSV file.
        
        Args:
            data: Data to export
            
        Returns:
            Success status
        """
        import csv
        
        file_path = self.config.get("file_path")
        if not file_path:
            self.logger.error("No file path specified in configuration")
            return False
        
        try:
            # Extract tables from data
            tables = []
            
            # Look for tables in the data
            if "tables" in data:
                tables = data["tables"]
            else:
                # Check for tables in pages
                for page in data.get("pages", []):
                    if "tables" in page:
                        for table in page["tables"]:
                            table["page_number"] = page.get("page_number", 0)
                            tables.append(table)
            
            if not tables:
                self.logger.warning("No tables found in data")
                return False
            
            # Export each table to a separate CSV file
            for i, table in enumerate(tables):
                # Determine file path for this table
                if len(tables) > 1:
                    base, ext = os.path.splitext(file_path)
                    table_path = f"{base}_table{i+1}{ext}"
                else:
                    table_path = file_path
                
                # Extract table data
                cells = table.get("cells", [])
                if not cells:
                    continue
                
                # Organize cells by row and column
                rows = table.get("rows", 0)
                cols = table.get("cols", 0)
                
                grid = [[None for _ in range(cols)] for _ in range(rows)]
                
                for cell in cells:
                    row = cell.get("row", 0)
                    col = cell.get("col", 0)
                    text = cell.get("text", "")
                    
                    if 0 <= row < rows and 0 <= col < cols:
                        grid[row][col] = text
                
                # Write to CSV
                with open(table_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    for row in grid:
                        writer.writerow(row)
                
                self.logger.info(f"Table {i+1} exported to CSV file: {table_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV file: {str(e)}")
            return False
    
    def import_data(self) -> Dict[str, Any]:
        """
        Import data from a CSV file.
        
        Returns:
            Imported data
        """
        import csv
        
        file_path = self.config.get("file_path")
        if not file_path:
            self.logger.error("No file path specified in configuration")
            return {}
        
        try:
            # Read CSV file
            rows = []
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    rows.append(row)
            
            # Create table structure
            if not rows:
                return {}
            
            num_rows = len(rows)
            num_cols = max(len(row) for row in rows)
            
            cells = []
            for i, row in enumerate(rows):
                for j, cell_text in enumerate(row):
                    if cell_text:
                        cells.append({
                            "row": i,
                            "col": j,
                            "text": cell_text,
                            "bbox": [0, 0, 0, 0],  # Placeholder
                            "confidence": 1.0
                        })
            
            table = {
                "rows": num_rows,
                "cols": num_cols,
                "cells": cells,
                "bbox": [0, 0, 0, 0],  # Placeholder
                "structure_type": "grid",
                "confidence": 1.0
            }
            
            self.logger.info(f"Data imported from CSV file: {file_path}")
            return {"tables": [table]}
            
        except Exception as e:
            self.logger.error(f"Error importing from CSV file: {str(e)}")
            return {}


class DatabaseConnector(DataConnector):
    """Connector for exporting to and importing from databases."""
    
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data to a database.
        
        Args:
            data: Data to export
            
        Returns:
            Success status
        """
        db_type = self.config.get("db_type", "sqlite")
        
        if db_type == "sqlite":
            return self._export_sqlite(data)
        elif db_type == "postgres":
            return self._export_postgres(data)
        else:
            self.logger.error(f"Unsupported database type: {db_type}")
            return False
    
    def _export_sqlite(self, data: Dict[str, Any]) -> bool:
        """Export data to SQLite database."""
        import sqlite3
        
        db_path = self.config.get("db_path")
        if not db_path:
            self.logger.error("No database path specified in configuration")
            return False
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_size INTEGER,
                    page_count INTEGER,
                    created_at TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    page_number INTEGER,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    id TEXT PRIMARY KEY,
                    page_id TEXT,
                    rows INTEGER,
                    cols INTEGER,
                    structure_type TEXT,
                    confidence REAL,
                    FOREIGN KEY (page_id) REFERENCES pages (id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cells (
                    id TEXT PRIMARY KEY,
                    table_id TEXT,
                    row_num INTEGER,
                    col_num INTEGER,
                    text TEXT,
                    confidence REAL,
                    FOREIGN KEY (table_id) REFERENCES tables (id)
                )
            """)
            
            # Insert document data
            doc_id = data.get("pipeline_id", str(uuid.uuid4()))
            doc_info = data.get("document_info", {})
            
            cursor.execute(
                "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
                (
                    doc_id,
                    data.get("file_name", ""),
                    data.get("file_size", 0),
                    doc_info.get("page_count", 0),
                    data.get("processing_started", "")
                )
            )
            
            # Insert pages and tables
            for page in data.get("pages", []):
                page_id = str(uuid.uuid4())
                page_num = page.get("page_number", 0)
                
                cursor.execute(
                    "INSERT INTO pages VALUES (?, ?, ?)",
                    (page_id, doc_id, page_num)
                )
                
                # Insert tables
                for table in page.get("tables", []):
                    table_id = str(uuid.uuid4())
                    
                    cursor.execute(
                        "INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            table_id,
                            page_id,
                            table.get("rows", 0),
                            table.get("cols", 0),
                            table.get("structure_type", ""),
                            table.get("confidence", 0.0)
                        )
                    )
                    
                    # Insert cells
                    for cell in table.get("cells", []):
                        cell_id = str(uuid.uuid4())
                        
                        cursor.execute(
                            "INSERT INTO cells VALUES (?, ?, ?, ?, ?, ?)",
                            (
                                cell_id,
                                table_id,
                                cell.get("row", 0),
                                cell.get("col", 0),
                                cell.get("text", ""),
                                cell.get("confidence", 0.0)
                            )
                        )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Data exported to SQLite database: {db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to SQLite database: {str(e)}")
            return False
    
    def _export_postgres(self, data: Dict[str, Any]) -> bool:
        """Export data to PostgreSQL database."""
        try:
            import psycopg2
        except ImportError:
            self.logger.error("psycopg2 module not installed. Please install it to use PostgreSQL.")
            return False
        
        conn_string = self.config.get("conn_string")
        if not conn_string:
            self.logger.error("No connection string specified in configuration")
            return False
        
        try:
            # Implementation similar to SQLite but using psycopg2
            self.logger.info("PostgreSQL export not yet fully implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error exporting to PostgreSQL database: {str(e)}")
            return False
    
    def import_data(self) -> Dict[str, Any]:
        """
        Import data from a database.
        
        Returns:
            Imported data
        """
        db_type = self.config.get("db_type", "sqlite")
        
        if db_type == "sqlite":
            return self._import_sqlite()
        elif db_type == "postgres":
            return self._import_postgres()
        else:
            self.logger.error(f"Unsupported database type: {db_type}")
            return {}
    
    def _import_sqlite(self) -> Dict[str, Any]:
        """Import data from SQLite database."""
        import sqlite3
        
        db_path = self.config.get("db_path")
        doc_id = self.config.get("document_id")
        
        if not db_path:
            self.logger.error("No database path specified in configuration")
            return {}
        
        if not doc_id:
            self.logger.error("No document ID specified in configuration")
            return {}
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query document data
            cursor.execute(
                "SELECT * FROM documents WHERE id = ?",
                (doc_id,)
            )
            doc = cursor.fetchone()
            
            if not doc:
                self.logger.error(f"Document not found: {doc_id}")
                return {}
            
            # Build document data
            data = {
                "pipeline_id": doc[0],
                "file_name": doc[1],
                "file_size": doc[2],
                "document_info": {
                    "page_count": doc[3]
                },
                "processing_started": doc[4],
                "pages": []
            }
            
            # Query pages
            cursor.execute(
                "SELECT * FROM pages WHERE document_id = ? ORDER BY page_number",
                (doc_id,)
            )
            pages = cursor.fetchall()
            
            for page in pages:
                page_id = page[0]
                page_num = page[2]
                
                page_data = {
                    "page_number": page_num,
                    "tables": []
                }
                
                # Query tables
                cursor.execute(
                    "SELECT * FROM tables WHERE page_id = ?",
                    (page_id,)
                )
                tables = cursor.fetchall()
                
                for table in tables:
                    table_id = table[0]
                    
                    table_data = {
                        "rows": table[2],
                        "cols": table[3],
                        "structure_type": table[4],
                        "confidence": table[5],
                        "cells": []
                    }
                    
                    # Query cells
                    cursor.execute(
                        "SELECT * FROM cells WHERE table_id = ? ORDER BY row_num, col_num",
                        (table_id,)
                    )
                    cells = cursor.fetchall()
                    
                    for cell in cells:
                        cell_data = {
                            "row": cell[2],
                            "col": cell[3],
                            "text": cell[4],
                            "confidence": cell[5]
                        }
                        
                        table_data["cells"].append(cell_data)
                    
                    page_data["tables"].append(table_data)
                
                data["pages"].append(page_data)
            
            conn.close()
            
            self.logger.info(f"Data imported from SQLite database: {db_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error importing from SQLite database: {str(e)}")
            return {}
    
    def _import_postgres(self) -> Dict[str, Any]:
        """Import data from PostgreSQL database."""
        try:
            import psycopg2
        except ImportError:
            self.logger.error("psycopg2 module not installed. Please install it to use PostgreSQL.")
            return {}
        
        conn_string = self.config.get("conn_string")
        if not conn_string:
            self.logger.error("No connection string specified in configuration")
            return {}
        
        try:
            # Implementation similar to SQLite but using psycopg2
            self.logger.info("PostgreSQL import not yet fully implemented")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error importing from PostgreSQL database: {str(e)}")
            return {}


class APIConnector(DataConnector):
    """Connector for integrating with APIs."""
    
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data to an API.
        
        Args:
            data: Data to export
            
        Returns:
            Success status
        """
        import requests
        
        endpoint = self.config.get("endpoint")
        if not endpoint:
            self.logger.error("No API endpoint specified in configuration")
            return False
        
        api_key = self.config.get("api_key")
        headers = self.config.get("headers", {})
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            response = requests.post(endpoint, json=data, headers=headers)
            response.raise_for_status()
            
            self.logger.info(f"Data exported to API: {endpoint}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to API: {str(e)}")
            return False
    
    def import_data(self) -> Dict[str, Any]:
        """
        Import data from an API.
        
        Returns:
            Imported data
        """
        import requests
        
        endpoint = self.config.get("endpoint")
        if not endpoint:
            self.logger.error("No API endpoint specified in configuration")
            return {}
        
        api_key = self.config.get("api_key")
        headers = self.config.get("headers", {})
        params = self.config.get("params", {})
        
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            response = requests.get(endpoint, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            self.logger.info(f"Data imported from API: {endpoint}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error importing from API: {str(e)}")
            return {}


class VectorDBConnector(DataConnector):
    """Connector for integrating with vector databases."""
    
    def export(self, data: Dict[str, Any]) -> bool:
        """
        Export data to a vector database.
        
        Args:
            data: Data to export
            
        Returns:
            Success status
        """
        db_type = self.config.get("db_type", "pgvector")
        
        if db_type == "pgvector":
            return self._export_pgvector(data)
        else:
            self.logger.error(f"Unsupported vector database type: {db_type}")
            return False
    
    def _export_pgvector(self, data: Dict[str, Any]) -> bool:
        """Export data to pgvector database."""
        try:
            import psycopg2
            from psycopg2 import sql
        except ImportError:
            self.logger.error("psycopg2 module not installed. Please install it to use pgvector.")
            return False
        
        conn_string = self.config.get("conn_string")
        if not conn_string:
            self.logger.error("No connection string specified in configuration")
            return False
        
        # Check for document embeddings
        if "embeddings" not in data:
            self.logger.error("No embeddings found in data")
            return False
        
        try:
            # Connect to database
            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()
            
            # Create tables if they don't exist
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_embeddings (
                    id TEXT PRIMARY KEY,
                    file_name TEXT,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    id TEXT PRIMARY KEY,
                    document_id TEXT,
                    text TEXT,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES document_embeddings (id)
                )
            """)
            
            # Insert document embedding
            doc_id = data.get("pipeline_id", str(uuid.uuid4()))
            doc_embedding = data["embeddings"].get("document", [])
            doc_metadata = {
                "file_size": data.get("file_size", 0),
                "page_count": data.get("document_info", {}).get("page_count", 0),
                "processing_started": data.get("processing_started", ""),
                "processing_completed": data.get("processing_completed", "")
            }
            
            cursor.execute(
                sql.SQL("INSERT INTO document_embeddings (id, file_name, embedding, metadata) VALUES (%s, %s, %s, %s)"),
                (
                    doc_id,
                    data.get("file_name", ""),
                    doc_embedding,
                    json.dumps(doc_metadata)
                )
            )
            
            # Insert chunk embeddings
            chunk_embeddings = data["embeddings"].get("chunks", [])
            
            for i, chunk in enumerate(chunk_embeddings):
                chunk_id = str(uuid.uuid4())
                chunk_text = chunk.get("text", "")
                chunk_embedding = chunk.get("embedding", [])
                chunk_metadata = chunk.get("metadata", {})
                
                cursor.execute(
                    sql.SQL("INSERT INTO chunk_embeddings (id, document_id, text, embedding, metadata) VALUES (%s, %s, %s, %s, %s)"),
                    (
                        chunk_id,
                        doc_id,
                        chunk_text,
                        chunk_embedding,
                        json.dumps(chunk_metadata)
                    )
                )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Data exported to pgvector database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting to pgvector database: {str(e)}")
            return False
    
    def import_data(self) -> Dict[str, Any]:
        """
        Import data from a vector database.
        
        Returns:
            Imported data
        """
        # Not implemented for vector databases
        self.logger.error("Import not supported for vector databases")
        return {}
