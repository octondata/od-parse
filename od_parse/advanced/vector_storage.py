"""
Vector Storage module for document embeddings.

This module provides a flexible interface for generating and storing
document embeddings with support for various vector databases and embedding models.
"""

import os
import logging
import json
from typing import Dict, List, Any, Optional, Union
import importlib.util
from pathlib import Path

from od_parse.utils.logging_utils import get_logger
from od_parse.config.settings import get_config, load_config


class VectorStorage:
    """
    Flexible vector storage for document embeddings.
    
    This class provides methods for generating embeddings from parsed documents
    and storing them in various vector databases. It supports configurable
    embedding models and vector database backends.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the vector storage with configuration options.
        
        Args:
            config: Configuration dictionary with the following options:
                - embedding_model: The embedding model to use (default: "openai")
                - embedding_model_name: The specific model name (default: "text-embedding-3-small")
                - embedding_dimension: The dimension of embeddings (default: 1536)
                - chunk_size: Size of text chunks for embeddings (default: 1000)
                - chunk_overlap: Overlap between chunks (default: 100)
                - api_key: API key for the embedding service
                - vector_db: Vector database type (default: "pgvector")
                - connection_string: Database connection string
        """
        self.logger = get_logger(__name__)
        self.config = config or {}
        
        # Load configuration if not already loaded
        if not get_config():
            load_config()
            
        # Set default configuration values
        embedding_provider = self.config.get("embedding_model", "openai")
        self.embedding_model = embedding_provider
        self.embedding_model_name = self.config.get(
            "embedding_model_name", 
            get_config(f"embedding_models.{embedding_provider}")
        )
        self.embedding_dimension = self.config.get(
            "embedding_dimension", 
            get_config(f"vector_db.pgvector.default_dimension")
        )
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        
        # Get API key from config, with fallbacks
        self.api_key = self.config.get(
            "api_key",
            get_config(f"api_keys.{embedding_provider}") or os.environ.get(f"{embedding_provider.upper()}_API_KEY")
        )
        
        self.vector_db = self.config.get("vector_db", "pgvector")
        self.connection_string = self.config.get("connection_string", "")
        
        # Initialize embedding model
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        self.embedding_client = None
        
        if self.embedding_model == "openai":
            try:
                from openai import OpenAI
                if not self.api_key:
                    self.logger.warning("OpenAI API key not provided. Embeddings will not work.")
                    return
                
                self.embedding_client = OpenAI(api_key=self.api_key)
                self.logger.info(f"Initialized OpenAI embedding model: {self.embedding_model_name}")
            except ImportError:
                self.logger.error("OpenAI package not installed. Please install it with: pip install openai")
        
        elif self.embedding_model == "huggingface":
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_client = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Initialized HuggingFace embedding model: {self.embedding_model_name}")
            except ImportError:
                self.logger.error("sentence-transformers package not installed. Please install it with: pip install sentence-transformers")
        
        elif self.embedding_model == "cohere":
            try:
                import cohere
                if not self.api_key:
                    self.logger.warning("Cohere API key not provided. Embeddings will not work.")
                    return
                
                self.embedding_client = cohere.Client(self.api_key)
                self.logger.info(f"Initialized Cohere embedding model: {self.embedding_model_name}")
            except ImportError:
                self.logger.error("Cohere package not installed. Please install it with: pip install cohere")
        
        elif self.embedding_model == "custom":
            # For custom embedding models, users need to provide their own implementation
            # through the config or environment
            custom_module_path = self.config.get("custom_embedding_module")
            custom_class_name = self.config.get("custom_embedding_class")
            
            if custom_module_path and custom_class_name:
                try:
                    spec = importlib.util.spec_from_file_location("custom_embeddings", custom_module_path)
                    custom_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(custom_module)
                    
                    custom_class = getattr(custom_module, custom_class_name)
                    self.embedding_client = custom_class(**self.config.get("custom_embedding_params", {}))
                    self.logger.info(f"Initialized custom embedding model from {custom_module_path}")
                except Exception as e:
                    self.logger.error(f"Error loading custom embedding model: {str(e)}")
            else:
                self.logger.error("Custom embedding model specified but module path or class name not provided")
        
        else:
            self.logger.error(f"Unsupported embedding model: {self.embedding_model}")
    
    def create_embeddings(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create embeddings from parsed document data.
        
        Args:
            parsed_data: The parsed document data
            
        Returns:
            Dictionary containing the document and chunk embeddings
        """
        if not self.embedding_client:
            self.logger.error("Embedding client not initialized")
            return {"document": [], "chunks": []}
        
        # Extract text from parsed data
        document_text = self._extract_text(parsed_data)
        
        # Create document-level embedding
        document_embedding = self._get_embedding(document_text)
        
        # Create chunk embeddings
        chunks = self._create_chunks(document_text)
        chunk_embeddings = []
        
        for i, chunk in enumerate(chunks):
            embedding = self._get_embedding(chunk)
            chunk_embeddings.append({
                "id": f"chunk_{i}",
                "text": chunk,
                "embedding": embedding,
                "metadata": {
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })
        
        # Return embeddings
        result = {
            "document": {
                "embedding": document_embedding,
                "metadata": {
                    "file_name": parsed_data.get("metadata", {}).get("file_name", "unknown"),
                    "page_count": parsed_data.get("metadata", {}).get("page_count", 0),
                }
            },
            "chunks": chunk_embeddings
        }
        
        return result
    
    def _extract_text(self, parsed_data: Dict[str, Any]) -> str:
        """Extract text content from parsed data."""
        text = ""
        
        # Extract text from content sections
        if "content" in parsed_data:
            for item in parsed_data["content"]:
                if isinstance(item, dict) and "text" in item:
                    text += item["text"] + "\n\n"
        
        # Extract text from tables
        if "tables" in parsed_data:
            for table in parsed_data["tables"]:
                if "data" in table:
                    for row in table["data"]:
                        text += " | ".join([str(cell) for cell in row]) + "\n"
                    text += "\n"
        
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """Split text into overlapping chunks for embedding."""
        chunks = []
        
        if not text:
            return chunks
        
        # Simple chunking by character count
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a sentence or paragraph boundary
            if end < len(text):
                for boundary in ["\n\n", ".\n", ". ", "\n"]:
                    boundary_pos = text.rfind(boundary, start, end)
                    if boundary_pos > start:
                        end = boundary_pos + len(boundary)
                        break
            
            chunks.append(text[start:end])
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text using the configured model."""
        if not text:
            return []
        
        try:
            if self.embedding_model == "openai":
                response = self.embedding_client.embeddings.create(
                    input=text,
                    model=self.embedding_model_name
                )
                return response.data[0].embedding
            
            elif self.embedding_model == "huggingface":
                return self.embedding_client.encode(text).tolist()
            
            elif self.embedding_model == "cohere":
                response = self.embedding_client.embed(
                    texts=[text],
                    model=self.embedding_model_name
                )
                return response.embeddings[0]
            
            elif self.embedding_model == "custom":
                # Assume the custom embedding model has an embed method
                return self.embedding_client.embed(text)
            
            else:
                self.logger.error(f"Unsupported embedding model: {self.embedding_model}")
                return []
        
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            return []
    
    def store_embeddings(self, embeddings: Dict[str, Any]) -> bool:
        """
        Store embeddings in the configured vector database.
        
        Args:
            embeddings: The document and chunk embeddings
            
        Returns:
            True if storage was successful, False otherwise
        """
        if self.vector_db == "pgvector":
            return self._store_pgvector(embeddings)
        elif self.vector_db == "qdrant":
            return self._store_qdrant(embeddings)
        elif self.vector_db == "pinecone":
            return self._store_pinecone(embeddings)
        elif self.vector_db == "milvus":
            return self._store_milvus(embeddings)
        elif self.vector_db == "weaviate":
            return self._store_weaviate(embeddings)
        elif self.vector_db == "chroma":
            return self._store_chroma(embeddings)
        elif self.vector_db == "json":
            return self._store_json(embeddings)
        else:
            self.logger.error(f"Unsupported vector database: {self.vector_db}")
            return False
    
    def _store_pgvector(self, embeddings: Dict[str, Any]) -> bool:
        """Store embeddings in a PostgreSQL database with pgvector extension."""
        try:
            import psycopg2
            from psycopg2 import sql
            import json
        except ImportError:
            self.logger.error("psycopg2 module not installed. Please install it with: pip install psycopg2-binary")
            return False
        
        if not self.connection_string:
            self.logger.error("PostgreSQL connection string not provided")
            return False
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cursor:
                    # Create extension and tables if they don't exist
                    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS document_embeddings (
                            id TEXT PRIMARY KEY,
                            file_name TEXT,
                            embedding vector({self.embedding_dimension}),
                            metadata JSONB
                        )
                    """)
                    
                    cursor.execute(f"""
                        CREATE TABLE IF NOT EXISTS chunk_embeddings (
                            id TEXT PRIMARY KEY,
                            document_id TEXT,
                            text TEXT,
                            embedding vector({self.embedding_dimension}),
                            metadata JSONB,
                            FOREIGN KEY (document_id) REFERENCES document_embeddings (id)
                        )
                    """)
                    
                    # Insert document embedding
                    doc_data = embeddings.get("document", {})
                    doc_embedding = doc_data.get("embedding", [])
                    doc_metadata = doc_data.get("metadata", {})
                    doc_id = doc_metadata.get("file_name", "").replace(".", "_") + "_" + str(hash(str(doc_metadata)))
                    
                    cursor.execute(
                        sql.SQL("INSERT INTO document_embeddings (id, file_name, embedding, metadata) VALUES (%s, %s, %s, %s)"),
                        (
                            doc_id,
                            doc_metadata.get("file_name", "unknown"),
                            doc_embedding,
                            json.dumps(doc_metadata)
                        )
                    )
                    
                    # Insert chunk embeddings
                    chunk_embeddings = embeddings.get("chunks", [])
                    
                    for chunk in chunk_embeddings:
                        chunk_id = chunk.get("id", "")
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
                    
                    self.logger.info(f"Stored embeddings in pgvector database")
                    return True
        
        except Exception as e:
            self.logger.error(f"Error storing embeddings in pgvector database: {str(e)}")
            return False
    
    def _store_qdrant(self, embeddings: Dict[str, Any]) -> bool:
        """Store embeddings in a Qdrant vector database."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
        except ImportError:
            self.logger.error("qdrant-client not installed. Please install it with: pip install qdrant-client")
            return False
        
        connection_params = self.config.get("qdrant", {})
        host = connection_params.get("host", "localhost")
        port = connection_params.get("port", 6333)
        
        try:
            client = QdrantClient(host=host, port=port)
            
            # Get document data
            doc_data = embeddings.get("document", {})
            doc_embedding = doc_data.get("embedding", [])
            doc_metadata = doc_data.get("metadata", {})
            doc_id = doc_metadata.get("file_name", "").replace(".", "_") + "_" + str(hash(str(doc_metadata)))
            
            # Create collections if they don't exist
            try:
                client.get_collection("documents")
            except:
                client.create_collection(
                    collection_name="documents",
                    vectors_config=models.VectorParams(size=self.embedding_dimension, distance=models.Distance.COSINE)
                )
            
            try:
                client.get_collection("chunks")
            except:
                client.create_collection(
                    collection_name="chunks",
                    vectors_config=models.VectorParams(size=self.embedding_dimension, distance=models.Distance.COSINE)
                )
            
            # Store document embedding
            client.upsert(
                collection_name="documents",
                points=[
                    models.PointStruct(
                        id=doc_id,
                        vector=doc_embedding,
                        payload={
                            "file_name": doc_metadata.get("file_name", "unknown"),
                            "metadata": doc_metadata
                        }
                    )
                ]
            )
            
            # Store chunk embeddings
            chunk_embeddings = embeddings.get("chunks", [])
            points = []
            
            for chunk in chunk_embeddings:
                chunk_id = chunk.get("id", "")
                chunk_text = chunk.get("text", "")
                chunk_embedding = chunk.get("embedding", [])
                chunk_metadata = chunk.get("metadata", {})
                
                points.append(
                    models.PointStruct(
                        id=chunk_id,
                        vector=chunk_embedding,
                        payload={
                            "document_id": doc_id,
                            "text": chunk_text,
                            "metadata": chunk_metadata
                        }
                    )
                )
            
            client.upsert(
                collection_name="chunks",
                points=points
            )
            
            self.logger.info(f"Stored embeddings in Qdrant database")
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing embeddings in Qdrant database: {str(e)}")
            return False
    
    def _store_json(self, embeddings: Dict[str, Any]) -> bool:
        """Store embeddings in a JSON file."""
        output_path = self.config.get("json_output_path", "embeddings.json")
        
        try:
            with open(output_path, "w") as f:
                json.dump(embeddings, f, indent=2)
            
            self.logger.info(f"Stored embeddings in JSON file: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error storing embeddings in JSON file: {str(e)}")
            return False
    
    # Placeholder methods for other vector databases
    def _store_pinecone(self, embeddings: Dict[str, Any]) -> bool:
        self.logger.info("Pinecone storage not yet implemented")
        return False
    
    def _store_milvus(self, embeddings: Dict[str, Any]) -> bool:
        self.logger.info("Milvus storage not yet implemented")
        return False
    
    def _store_weaviate(self, embeddings: Dict[str, Any]) -> bool:
        self.logger.info("Weaviate storage not yet implemented")
        return False
    
    def _store_chroma(self, embeddings: Dict[str, Any]) -> bool:
        self.logger.info("Chroma storage not yet implemented")
        return False
