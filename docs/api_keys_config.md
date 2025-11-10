# API Keys Configuration Guide

This guide explains how to configure API keys for od-parse, including passing them as configuration parameters.

## Configuration Methods

There are three ways to configure API keys:

1. **Pass as configuration parameter** (Recommended for programmatic use)
2. **Environment variables** (Recommended for security)
3. **Configuration files** (For development)

## Method 1: Pass as Configuration Parameter

You can pass API keys directly to `parse_pdf()` function:

```python
from od_parse import parse_pdf

# Parse with API keys passed as configuration
result = parse_pdf(
    "document.pdf",
    api_keys={
        "google": "AIzaSy...",  # Gemini API key
        "openai": "sk-...",     # OpenAI API key (optional)
        "anthropic": "sk-ant-...",  # Anthropic API key (optional)
        "vllm_server_url": "http://localhost:8000",  # vLLM server URL (optional)
        "vllm_api_key": "optional-key"  # vLLM API key (optional)
    }
)
```

### Using with LLMDocumentProcessor

```python
from od_parse.llm import LLMDocumentProcessor
from od_parse.parser import parse_pdf as core_parse

# Parse PDF first
parsed_data = core_parse("document.pdf")

# Process with LLM using API keys from config
processor = LLMDocumentProcessor(
    model_id="gemini-2.0-flash",
    api_keys={
        "google": "AIzaSy..."
    }
)

enhanced_data = processor.process_document(parsed_data)
```

## Method 2: Environment Variables

Set environment variables (recommended for security):

```bash
export GOOGLE_API_KEY="AIzaSy..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export VLLM_SERVER_URL="http://localhost:8000"
export VLLM_API_KEY="optional-key"
```

Then use normally:
```python
from od_parse import parse_pdf

# Automatically uses environment variables
result = parse_pdf("document.pdf")
```

## Method 3: Configuration Files

Create a configuration file (e.g., `od-parse.yaml`):

```yaml
api_keys:
  google: "AIzaSy..."
  openai: "sk-..."
  anthropic: "sk-ant-..."
```

**Note**: Configuration files are less secure. Use environment variables or pass as parameters for production.

## API Keys Dictionary Format

The `api_keys` parameter accepts a dictionary with the following keys:

| Key | Description | Example |
|-----|-------------|---------|
| `"google"` | Google/Gemini API key | `"AIzaSy..."` |
| `"openai"` | OpenAI API key | `"sk-..."` |
| `"anthropic"` | Anthropic/Claude API key | `"sk-ant-..."` |
| `"azure_openai"` | Azure OpenAI API key | `"your-key"` |
| `"cohere"` | Cohere API key | `"your-key"` |
| `"huggingface"` | HuggingFace API key | `"hf_..."` |
| `"vllm_server_url"` | vLLM server URL | `"http://localhost:8000"` |
| `"vllm_api_key"` | vLLM API key (optional) | `"optional-key"` |

## Priority Order

API keys are resolved in the following priority order:

1. **Provided in `api_keys` parameter** (highest priority)
2. **Environment variables**
3. **Configuration files** (if supported)

## Examples

### Example 1: Basic Usage with Google API Key

```python
from od_parse import parse_pdf

result = parse_pdf(
    "document.pdf",
    api_keys={"google": "AIzaSy..."}
)
```

### Example 2: Multiple API Keys

```python
from od_parse import parse_pdf

result = parse_pdf(
    "document.pdf",
    api_keys={
        "google": "AIzaSy...",
        "openai": "sk-...",
        "anthropic": "sk-ant-..."
    }
)
```

### Example 3: With vLLM Server

```python
from od_parse import parse_pdf

result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b",
    api_keys={
        "vllm_server_url": "http://localhost:8000",
        "vllm_api_key": "optional-key"
    }
)
```

### Example 4: Using LLMDocumentProcessor

```python
from od_parse.llm import LLMDocumentProcessor
from od_parse.parser import parse_pdf as core_parse

# Parse PDF
parsed_data = core_parse("document.pdf")

# Process with LLM using API keys
processor = LLMDocumentProcessor(
    model_id="gemini-2.0-flash",
    api_keys={"google": "AIzaSy..."}
)

enhanced_data = processor.process_document(parsed_data)
```

### Example 5: Loading from Secrets Manager

```python
import os
from od_parse import parse_pdf

# Load from secrets manager (example)
def get_api_key(provider: str) -> str:
    # Your secrets manager logic here
    return os.getenv(f"{provider.upper()}_API_KEY")

# Use in parse_pdf
result = parse_pdf(
    "document.pdf",
    api_keys={
        "google": get_api_key("google"),
        "openai": get_api_key("openai")
    }
)
```

## Security Best Practices

1. **Never commit API keys to version control**
2. **Use environment variables for production**
3. **Pass keys as parameters only when necessary**
4. **Rotate keys regularly**
5. **Use secrets managers** (AWS Secrets Manager, Azure Key Vault, etc.)

## Troubleshooting

### API Key Not Found

**Error**: "No LLM API keys found"

**Solution**: 
- Pass `api_keys` parameter to `parse_pdf()`
- Or set environment variables
- Or check key names match expected format

### Invalid API Key

**Error**: "Invalid API key" or authentication errors

**Solution**:
- Verify API key is correct
- Check key format matches provider requirements
- Ensure key has necessary permissions

### Key Priority Issues

**Problem**: Wrong key being used

**Solution**:
- Check priority order: provided keys > environment variables
- Explicitly pass `api_keys` parameter to override environment variables

## Next Steps

- See [README.md](../README.md) for general usage
- See [gemini_2_setup.md](gemini_2_setup.md) for Gemini setup
- See [vllm_setup_guide.md](vllm_setup_guide.md) for vLLM setup

