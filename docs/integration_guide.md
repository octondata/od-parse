# OctonData Parse Integration Guide

This guide demonstrates how to integrate the advanced PDF parsing capabilities of `od-parse` with various enterprise systems and workflows. The library has been designed to seamlessly connect with databases, APIs, and vector stores that power modern AI applications.

## Table of Contents

- [Integration with Vector Databases](#integration-with-vector-databases)
- [Integration with Relational Databases](#integration-with-relational-databases)
- [Integration with REST APIs](#integration-with-rest-apis)
- [Integration with RAG Systems](#integration-with-rag-systems)
- [Integration with Document Management Systems](#integration-with-document-management-systems)
- [Custom Integrations](#custom-integrations)

## Integration with Vector Databases

The `od-parse` library includes built-in support for pgvector, making it easy to store document embeddings for semantic search and RAG applications.

### PostgreSQL with pgvector

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import VectorDBConnector
import os

# Process the document using the full pipeline
pipeline = PDFPipeline.create_full_pipeline()
result = pipeline.process("financial_report.pdf")

# Configure connection to pgvector database
vector_connector = VectorDBConnector({
    "db_type": "pgvector",
    "conn_string": "postgresql://pgvector_admin:Od-Vector2025@octondata-dev-pgvector.c70okyo8mtsw.us-west-1.rds.amazonaws.com:5432/pgvector"
})

# Export document with embeddings to pgvector
vector_connector.export(result)
```

This will store both document-level and chunk-level embeddings in the pgvector database, making them available for semantic search and RAG applications.

## Integration with Relational Databases

### SQLite

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import DatabaseConnector

# Process the document
pipeline = PDFPipeline.create_full_pipeline()
result = pipeline.process("invoice.pdf")

# Configure connection to SQLite database
db_connector = DatabaseConnector({
    "db_type": "sqlite",
    "db_path": "documents.db"
})

# Export document data to SQLite
db_connector.export(result)
```

### PostgreSQL

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import DatabaseConnector
import os

# Process the document
# Use the UnifiedPDFParser directly
from od_parse.advanced.unified_parser import UnifiedPDFParser

# Configure the parser with the desired options
parser = UnifiedPDFParser({
    "use_deep_learning": True,
    "extract_handwritten": True,
    "extract_tables": True,
    "extract_forms": True,
    "extract_structure": True
})

# Parse the document
result = parser.parse("contract.pdf")

# Configure connection to PostgreSQL database
db_connector = DatabaseConnector({
    "db_type": "postgres",
    "conn_string": "postgresql://username:password@hostname:5432/database"
})

# Export document data to PostgreSQL
db_connector.export(result)
```

## Integration with REST APIs

The `od-parse` library can be used with external APIs for further processing or storage.

```python
from od_parse.advanced.unified_parser import UnifiedPDFParser
from od_parse.advanced.integrations import APIConnector
import os

# Configure and use the parser
parser = UnifiedPDFParser({
    "use_deep_learning": True,
    "extract_handwritten": True,
    "extract_tables": True
})

# Process the document
result = parser.parse("medical_report.pdf")

# Configure connection to API
api_connector = APIConnector({
    "endpoint": "https://api.example.com/documents",
    "api_key": os.environ.get("API_KEY"),
    "headers": {
        "Content-Type": "application/json",
        "X-Custom-Header": "value"
    }
})

# Export document data to API
api_connector.export(result)
```

## Integration with RAG Systems

The advanced parsing capabilities of `od-parse` make it ideal for building RAG (Retrieval-Augmented Generation) systems. Here's how to integrate it with a typical RAG pipeline:

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import VectorDBConnector
import os
import uuid

# Step 1: Process documents
pipeline = PDFPipeline.create_full_pipeline()
documents = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
results = []

for doc in documents:
    result = pipeline.process(doc)
    results.append(result)

# Step 2: Store in vector database
vector_connector = VectorDBConnector({
    "db_type": "pgvector",
    "conn_string": "postgresql://pgvector_admin:Od-Vector2025@octondata-dev-pgvector.c70okyo8mtsw.us-west-1.rds.amazonaws.com:5432/pgvector"
})

for result in results:
    vector_connector.export(result)

# Step 3: Example of querying the vector database for RAG
# (Using psycopg2 and pgvector)
import psycopg2
from psycopg2 import sql

# Connect to database
conn = psycopg2.connect("postgresql://pgvector_admin:Od-Vector2025@octondata-dev-pgvector.c70okyo8mtsw.us-west-1.rds.amazonaws.com:5432/pgvector")
cursor = conn.cursor()

# Query chunks most similar to a question
question = "What is the revenue forecast for Q3?"
question_embedding = get_embedding(question)  # Function to get embedding for question

cursor.execute(
    sql.SQL("SELECT text, metadata FROM chunk_embeddings ORDER BY embedding <-> %s LIMIT 5"),
    (question_embedding,)
)

chunks = cursor.fetchall()
context = "\n".join([chunk[0] for chunk in chunks])

# Use the retrieved context with an LLM
# response = query_llm(question, context)
```

## Integration with Document Management Systems

### Saving Parsed Documents

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import JSONFileConnector, CSVConnector
from od_parse.advanced.unified_parser import UnifiedPDFParser
import os
import json

# Process the document
pipeline = PDFPipeline.create_full_pipeline()
result = pipeline.process("legal_document.pdf")

# Save as JSON
json_connector = JSONFileConnector({
    "file_path": "legal_document.json"
})
json_connector.export(result)

# Save tables as CSV
csv_connector = CSVConnector({
    "file_path": "legal_document_tables.csv"
})
csv_connector.export(result)

# Save markdown for human-readable version
markdown = UnifiedPDFParser().to_markdown(result)
with open("legal_document.md", "w") as f:
    f.write(markdown)
```

## Custom Integrations

You can create custom integrations by extending the `DataConnector` base class:

```python
from od_parse.advanced.integrations import DataConnector
import uuid

class CustomConnector(DataConnector):
    """Custom connector for your specific use case."""
    
    def export(self, data):
        """Export data to your custom system."""
        # Your custom implementation here
        print(f"Exporting data to custom system")
        # Process the data as needed
        return True
    
    def import_data(self):
        """Import data from your custom system."""
        # Your custom implementation here
        return {"custom_data": "value"}

# Use your custom connector
custom_connector = CustomConnector({"custom_config": "value"})
custom_connector.export(result)
```

## Real-World Example: Building a Document Processing Pipeline with OctonData Platform

This example shows how to integrate the advanced PDF parsing capabilities with the OctonData platform's existing components:

```python
from od_parse.advanced.pipeline import PDFPipeline, LoadDocumentStage, AdvancedParsingStage
from od_parse.advanced.pipeline import TableExtractionStage, FormExtractionStage, DocumentStructureStage, OutputFormattingStage
from od_parse.advanced.integrations import VectorDBConnector, DatabaseConnector
from od_parse.advanced.unified_parser import UnifiedPDFParser
import os
import json
import uuid
from datetime import datetime

# Step 1: Configure the pipeline for document processing
pipeline = PDFPipeline()
pipeline.add_stage(LoadDocumentStage())
pipeline.add_stage(AdvancedParsingStage())
pipeline.add_stage(TableExtractionStage({"use_neural": True}))
pipeline.add_stage(FormExtractionStage())
pipeline.add_stage(DocumentStructureStage())
pipeline.add_stage(OutputFormattingStage({"format": "json"}))

# Step 2: Process the document
result = pipeline.process("quarterly_report.pdf")

# Step 3: Store structured data in PostgreSQL
db_connector = DatabaseConnector({
    "db_type": "postgres",
    "conn_string": "postgresql://username:password@hostname:5432/database"
})
db_connector.export(result)

# Step 4: Store vector embeddings in pgvector for RAG
vector_connector = VectorDBConnector({
    "db_type": "pgvector",
    "conn_string": "postgresql://pgvector_admin:Od-Vector2025@octondata-dev-pgvector.c70okyo8mtsw.us-west-1.rds.amazonaws.com:5432/pgvector"
})
vector_connector.export(result)

# Step 5: Generate human-readable markdown
markdown = UnifiedPDFParser().to_markdown(result)
with open("quarterly_report.md", "w") as f:
    f.write(markdown)

# Step 6: Log the processing
with open("processing_log.json", "w") as f:
    log_data = {
        "file": "quarterly_report.pdf",
        "timestamp": datetime.now().isoformat(),
        "duration": result.get("processing_duration_seconds"),
        "stats": result.get("summary", {}).get("extraction_statistics", {})
    }
    json.dump(log_data, f, indent=2)

print("Document processing pipeline completed successfully!")
```

## AWS Integration Example

This example shows how to integrate the PDF parser with AWS services for a scalable document processing pipeline:

```python
from od_parse.advanced.pipeline import PDFPipeline
from od_parse.advanced.integrations import VectorDBConnector, DatabaseConnector
import boto3
import os
import json
import uuid

# Step 1: Configure S3 client
s3 = boto3.client('s3')
s3_bucket = 'airbyte-docs-20250505102840'  # Example bucket name
s3_prefix = 'processed-documents/'

# Step 2: Configure the PDF processing pipeline
pipeline = PDFPipeline.create_full_pipeline()

# Step 3: Process a document from S3
file_key = 'incoming-documents/report.pdf'
local_file = '/tmp/report.pdf'

# Download file from S3
s3.download_file(s3_bucket, file_key, local_file)

# Process the document
result = pipeline.process(local_file)

# Step 4: Store extraction results back in S3
output_key = f"{s3_prefix}{uuid.uuid4()}.json"
with open('/tmp/output.json', 'w') as f:
    json.dump(result, f)

s3.upload_file('/tmp/output.json', s3_bucket, output_key)

# Step 5: Store in pgvector for semantic search
vector_connector = VectorDBConnector({
    "db_type": "pgvector",
    "conn_string": "postgresql://pgvector_admin:Od-Vector2025@octondata-dev-pgvector.c70okyo8mtsw.us-west-1.rds.amazonaws.com:5432/pgvector"
})
vector_connector.export(result)

# Step 6: Send processing notification
sns = boto3.client('sns')
sns_topic = 'arn:aws:sns:us-west-1:123456789012:document-processing'

sns.publish(
    TopicArn=sns_topic,
    Message=json.dumps({
        'status': 'completed',
        'document': file_key,
        'output': output_key,
        'timestamp': datetime.now().isoformat(),
        'stats': result.get('summary', {}).get('extraction_statistics', {})
    }),
    Subject='Document Processing Completed'
)

print(f"Document processing completed for {file_key}")
```

This guide demonstrates the versatility of the `od-parse` library for integration with various enterprise systems and workflows. For additional assistance or custom integration needs, please reach out to the OctonData team.
