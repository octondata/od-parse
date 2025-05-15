# RAG Integration Guide: Using OctonData Parse with Vector Databases

This guide demonstrates how to leverage the advanced PDF parsing capabilities of `od-parse` to build powerful Retrieval-Augmented Generation (RAG) systems using OctonData's infrastructure.

## Table of Contents

- [Introduction to RAG with Document Parsing](#introduction-to-rag-with-document-parsing)
- [System Architecture](#system-architecture)
- [Step-by-Step Integration](#step-by-step-integration)
- [Advanced Configuration](#advanced-configuration)
- [Performance Optimization](#performance-optimization)
- [Example Implementation](#example-implementation)
- [Troubleshooting](#troubleshooting)

## Introduction to RAG with Document Parsing

Retrieval-Augmented Generation (RAG) combines the power of large language models with a retrieval system that fetches relevant information from a knowledge base. For enterprise applications, PDF documents often contain critical information that needs to be made available to RAG systems. The `od-parse` library provides the critical first step in this process by extracting rich, structured content from complex PDFs.

### Benefits of Using OctonData Parse for RAG

1. **Enhanced Document Understanding**: Deep learning-based extraction captures complex document structures.
2. **Structured Data Extraction**: Tables, forms, and semantic structure are properly extracted and preserved.
3. **Rich Context Preservation**: Layout information and relationships between document elements are maintained.
4. **Seamless Vector Database Integration**: Direct integration with pgvector for efficient storage and retrieval.
5. **Enterprise-Ready**: Works with OctonData's existing infrastructure components.

## System Architecture

A typical RAG system using `od-parse` consists of the following components:

1. **Document Processing Pipeline**:
   - Document loading and preprocessing
   - Advanced content extraction using UnifiedPDFParser
   - Semantic chunking based on document structure
   - Text embedding generation

2. **Vector Storage Layer**:
   - pgvector on AWS RDS for vector storage
   - Metadata storage for document properties
   - Index management for efficient retrieval

3. **Retrieval System**:
   - Semantic search capabilities
   - Relevance scoring
   - Context assembly for LLM prompting

4. **Generation Layer**:
   - Prompt construction with retrieved context
   - LLM integration for response generation
   - Output refinement based on document structure

## Step-by-Step Integration

### 1. Document Processing and Extraction

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.unified_parser import UnifiedPDFParser

# Create a pipeline optimized for RAG
pipeline = PDFPipeline()
pipeline.add_stage(LoadDocumentStage())
pipeline.add_stage(AdvancedParsingStage())
pipeline.add_stage(TableExtractionStage({"use_neural": True}))
pipeline.add_stage(DocumentStructureStage())
pipeline.add_stage(OutputFormattingStage({"format": "json"}))

# Process document
result = pipeline.process("enterprise_document.pdf")
```

### 2. Semantic Chunking

```python
def create_semantic_chunks(parsed_document):
    """Create semantically meaningful chunks from the parsed document."""
    chunks = []
    
    # Extract structure elements
    structure = parsed_document.get("structure", {})
    elements = structure.get("elements", [])
    
    current_section = {"title": "", "content": [], "metadata": {}}
    
    for element in elements:
        element_type = element.get("type")
        
        if element_type == "heading":
            # Save previous section if it has content
            if current_section["content"]:
                chunks.append({
                    "text": "\n".join(current_section["content"]),
                    "metadata": {
                        "title": current_section["title"],
                        "document": parsed_document.get("file_name"),
                        "page_numbers": current_section.get("page_numbers", [])
                    }
                })
            
            # Start new section
            current_section = {
                "title": element.get("text", ""),
                "content": [element.get("text", "")],
                "metadata": {"level": element.get("level", 1)},
                "page_numbers": set([element.get("page", 0)])
            }
        else:
            # Add content to current section
            if element.get("text"):
                current_section["content"].append(element.get("text"))
            if "page" in element:
                current_section["page_numbers"].add(element.get("page"))
    
    # Add final section
    if current_section["content"]:
        chunks.append({
            "text": "\n".join(current_section["content"]),
            "metadata": {
                "title": current_section["title"],
                "document": parsed_document.get("file_name"),
                "page_numbers": list(current_section.get("page_numbers", []))
            }
        })
    
    # Process tables separately
    for page in parsed_document.get("pages", []):
        for table in page.get("tables", []):
            table_markdown = table.get("markdown", "")
            if table_markdown:
                chunks.append({
                    "text": table_markdown,
                    "metadata": {
                        "content_type": "table",
                        "document": parsed_document.get("file_name"),
                        "page_number": page.get("page_number")
                    }
                })
    
    return chunks

# Create semantic chunks
chunks = create_semantic_chunks(result)
```

### 3. Generating Embeddings

```python
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for text chunks using a sentence transformer model."""
    model = SentenceTransformer(model_name)
    
    for chunk in chunks:
        # Generate embedding for text
        text = chunk["text"]
        embedding = model.encode(text)
        
        # Add embedding to chunk
        chunk["embedding"] = embedding.tolist()
    
    return chunks

# Generate embeddings for chunks
chunks_with_embeddings = generate_embeddings(chunks)
```

### 4. Storing in pgvector

```python
import psycopg2
from psycopg2 import sql
import json

def store_in_pgvector(chunks, conn_string):
    """Store chunks with embeddings in pgvector database."""
    # Connect to database
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    # Ensure pgvector extension is installed
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    
    # Create table for chunks if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS document_chunks (
        id SERIAL PRIMARY KEY,
        document_name TEXT,
        chunk_text TEXT,
        embedding vector(384),
        metadata JSONB
    )
    """)
    
    # Insert chunks
    for chunk in chunks:
        cursor.execute(
            sql.SQL("""
            INSERT INTO document_chunks (document_name, chunk_text, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            """),
            (
                chunk["metadata"].get("document", ""),
                chunk["text"],
                chunk["embedding"],
                json.dumps(chunk["metadata"])
            )
        )
    
    # Commit transaction
    conn.commit()
    conn.close()
    
    return len(chunks)

# Store chunks in pgvector
conn_string = "postgresql://pgvector_admin:Od-Vector2025@octondata-dev-pgvector.c70okyo8mtsw.us-west-1.rds.amazonaws.com:5432/pgvector"
num_stored = store_in_pgvector(chunks_with_embeddings, conn_string)
print(f"Stored {num_stored} chunks in pgvector database")
```

### 5. Retrieving Context for RAG

```python
def retrieve_context(query, conn_string, model_name="all-MiniLM-L6-v2", top_k=5):
    """Retrieve relevant context for a query from pgvector database."""
    # Generate embedding for query
    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query).tolist()
    
    # Connect to database
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    # Query for most similar chunks
    cursor.execute(
        sql.SQL("""
        SELECT chunk_text, metadata, 1 - (embedding <=> %s) AS similarity
        FROM document_chunks
        ORDER BY embedding <=> %s
        LIMIT %s
        """),
        (query_embedding, query_embedding, top_k)
    )
    
    results = cursor.fetchall()
    conn.close()
    
    # Format results
    context_chunks = []
    for text, metadata, similarity in results:
        context_chunks.append({
            "text": text,
            "metadata": json.loads(metadata),
            "similarity": similarity
        })
    
    return context_chunks

# Retrieve context for a query
query = "What are the revenue projections for Q3 2025?"
context = retrieve_context(query, conn_string)

# Format context for LLM prompt
formatted_context = "\n\n".join([
    f"Source: {chunk['metadata'].get('document')} - {chunk['metadata'].get('title')}\n{chunk['text']}"
    for chunk in context
])

print(f"Retrieved {len(context)} context chunks for query: {query}")
```

## Advanced Configuration

### Customizing Chunk Size and Strategy

The default semantic chunking strategy is based on document structure. However, depending on your specific use case, you may want to adjust the chunking strategy:

```python
def chunk_by_token_count(parsed_document, target_token_count=512):
    """Chunk document by target token count rather than semantic structure."""
    import nltk
    from nltk.tokenize import word_tokenize
    
    # Download tokenizer if needed
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    chunks = []
    current_chunk = []
    current_token_count = 0
    current_metadata = {"document": parsed_document.get("file_name"), "page_numbers": set()}
    
    # Extract all text from document
    for page in parsed_document.get("pages", []):
        page_num = page.get("page_number")
        
        for text_block in page.get("text_blocks", []):
            text = text_block.get("text", "")
            tokens = word_tokenize(text)
            
            if current_token_count + len(tokens) > target_token_count and current_chunk:
                # Save current chunk
                chunks.append({
                    "text": " ".join(current_chunk),
                    "metadata": {
                        "document": current_metadata["document"],
                        "page_numbers": list(current_metadata["page_numbers"])
                    }
                })
                
                # Start new chunk
                current_chunk = []
                current_token_count = 0
                current_metadata = {"document": parsed_document.get("file_name"), "page_numbers": set()}
            
            # Add to current chunk
            current_chunk.append(text)
            current_token_count += len(tokens)
            current_metadata["page_numbers"].add(page_num)
    
    # Add final chunk
    if current_chunk:
        chunks.append({
            "text": " ".join(current_chunk),
            "metadata": {
                "document": current_metadata["document"],
                "page_numbers": list(current_metadata["page_numbers"])
            }
        })
    
    return chunks
```

### Using Different Embedding Models

For different use cases, you may want to use different embedding models:

```python
def generate_embeddings_custom(chunks, model_name="BAAI/bge-large-en-v1.5"):
    """Generate embeddings using a custom model."""
    model = SentenceTransformer(model_name)
    
    for chunk in chunks:
        # Generate embedding for text
        text = chunk["text"]
        embedding = model.encode(text, normalize_embeddings=True)
        
        # Add embedding to chunk
        chunk["embedding"] = embedding.tolist()
    
    return chunks
```

## Performance Optimization

### Batch Processing for Large Document Sets

When processing many documents, use batch processing for efficiency:

```python
def process_document_batch(file_paths, output_dir, conn_string):
    """Process a batch of documents and store in pgvector."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    def process_single_document(file_path):
        try:
            # Create filename for output
            base_name = os.path.basename(file_path)
            output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}.json")
            
            # Process document
            pipeline = PDFPipeline.create_full_pipeline()
            result = pipeline.process(file_path)
            
            # Save result to file
            with open(output_path, 'w') as f:
                json.dump(result, f)
            
            # Create chunks and embeddings
            chunks = create_semantic_chunks(result)
            chunks_with_embeddings = generate_embeddings(chunks)
            
            # Store in pgvector
            store_in_pgvector(chunks_with_embeddings, conn_string)
            
            return {"status": "success", "file": file_path, "chunks": len(chunks)}
        except Exception as e:
            return {"status": "error", "file": file_path, "error": str(e)}
    
    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_single_document, file_paths))
    
    return results
```

### Optimizing Vector Queries with Indexes

For faster queries on large vector databases, create appropriate indexes:

```python
def create_vector_index(conn_string):
    """Create an index on the vector column for faster queries."""
    conn = psycopg2.connect(conn_string)
    cursor = conn.cursor()
    
    # Create index
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks
    USING ivfflat (embedding vector_l2_ops)
    WITH (lists = 100)
    """)
    
    conn.commit()
    conn.close()
```

## Example Implementation

Here's a complete example that ties everything together:

```python
import os
import json
import argparse
from od_parse.advanced.pipeline import PDFPipeline, LoadDocumentStage, AdvancedParsingStage, TableExtractionStage, DocumentStructureStage, OutputFormattingStage
from sentence_transformers import SentenceTransformer
import psycopg2
from psycopg2 import sql

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Process documents for RAG")
    parser.add_argument("--input", help="Input document or directory", required=True)
    parser.add_argument("--output", help="Output directory", default="./output")
    parser.add_argument("--conn_string", help="Connection string for pgvector database", required=True)
    args = parser.parse_args()
    
    # Check if input is a directory or file
    if os.path.isdir(args.input):
        file_paths = [os.path.join(args.input, f) for f in os.listdir(args.input) if f.lower().endswith('.pdf')]
        print(f"Processing {len(file_paths)} documents...")
        results = process_document_batch(file_paths, args.output, args.conn_string)
        
        # Print results
        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "error"]
        
        print(f"Successfully processed {len(successes)} documents with {sum(r['chunks'] for r in successes)} chunks")
        if failures:
            print(f"Failed to process {len(failures)} documents:")
            for failure in failures:
                print(f"  - {failure['file']}: {failure['error']}")
    else:
        # Process single document
        print(f"Processing document: {args.input}")
        
        try:
            # Process document
            pipeline = PDFPipeline.create_full_pipeline()
            result = pipeline.process(args.input)
            
            # Save result
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(args.output, f"{os.path.splitext(os.path.basename(args.input))[0]}.json")
            
            with open(output_path, 'w') as f:
                json.dump(result, f)
            
            # Create chunks and embeddings
            chunks = create_semantic_chunks(result)
            chunks_with_embeddings = generate_embeddings(chunks)
            
            # Store in pgvector
            num_stored = store_in_pgvector(chunks_with_embeddings, args.conn_string)
            
            print(f"Successfully processed document with {num_stored} chunks")
            print(f"Results saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing document: {str(e)}")
    
    print("Document processing complete")

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues and Solutions

1. **Issue**: Poor retrieval quality for complex tables or charts.
   **Solution**: Ensure that tables are properly extracted and formatted. For charts, consider extracting the image and providing a textual description.

2. **Issue**: Vector database connection errors.
   **Solution**: Verify connection string and ensure that the pgvector extension is installed in your PostgreSQL database.

3. **Issue**: Out of memory errors when processing large documents.
   **Solution**: Adjust the pipeline configuration to use less memory-intensive options, such as `use_deep_learning: False` for very large documents.

4. **Issue**: Missing context in retrieved chunks.
   **Solution**: Adjust your chunking strategy to ensure that related content stays together. Consider using overlapping chunks.

5. **Issue**: Slow query performance on large databases.
   **Solution**: Create appropriate indexes and consider sharding your vector database for improved performance.

### Monitoring and Logging

For production deployments, implement comprehensive logging:

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='rag_processing.log'
)

# Add logging to your processing functions
def process_document_with_logging(file_path, *args, **kwargs):
    logger = logging.getLogger('rag_processor')
    logger.info(f"Processing document: {file_path}")
    
    try:
        # Process document
        result = process_single_document(file_path, *args, **kwargs)
        logger.info(f"Successfully processed {file_path} with {result['chunks']} chunks")
        return result
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        raise
```

This guide provides a comprehensive approach to integrating OctonData Parse with pgvector for building powerful RAG systems. By following these steps and patterns, you can create enterprise-grade document understanding capabilities that enhance the performance of AI applications.
