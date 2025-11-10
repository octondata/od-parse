# Choosing Between Cloud LLM APIs and vLLM

This guide explains how to differentiate and choose between cloud LLM APIs (OpenAI, Google, Anthropic) and vLLM (local/cloud-hosted inference) when using od-parse.

## Quick Answer

**Differentiation is done by model name:**

- **Cloud LLM APIs**: Use model names like `gemini-2.0-flash`, `gpt-4o`, `claude-3-5-sonnet`
- **vLLM**: Use model names starting with `vllm-`, like `vllm-llama-3.1-8b`, `vllm-qwen-2.5-7b`

## Model Naming Convention

### Cloud LLM API Models

These models use cloud APIs (OpenAI, Google, Anthropic):

| Model ID | Provider | API Key Required |
|----------|----------|------------------|
| `gemini-2.0-flash` | Google | `GOOGLE_API_KEY` |
| `gemini-2.0-flash-exp` | Google | `GOOGLE_API_KEY` |
| `gemini-1.5-pro` | Google | `GOOGLE_API_KEY` |
| `gemini-1.5-flash` | Google | `GOOGLE_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4-turbo` | OpenAI | `OPENAI_API_KEY` |
| `claude-3-5-sonnet` | Anthropic | `ANTHROPIC_API_KEY` |
| `claude-3-haiku` | Anthropic | `ANTHROPIC_API_KEY` |
| `azure-gpt-4o` | Azure OpenAI | `AZURE_OPENAI_API_KEY` |

### vLLM Models

These models use vLLM (local or cloud-hosted):

| Model ID | Provider | Setup Required |
|----------|----------|----------------|
| `vllm-llama-3.1-8b` | vLLM | vLLM server running |
| `vllm-llama-3.1-70b` | vLLM | vLLM server running |
| `vllm-qwen-2.5-7b` | vLLM | vLLM server running |
| `vllm-mistral-7b` | vLLM | vLLM server running |

## Usage Examples

### Using Cloud LLM API (Google Gemini)

```python
from od_parse import parse_pdf

# Use Google Gemini (cloud API)
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",  # Cloud LLM API
    api_keys={"google": "AIzaSy..."}
)
```

### Using vLLM (Local/Cloud-Hosted)

```python
from od_parse import parse_pdf

# Use vLLM (local or cloud-hosted)
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b",  # vLLM model (starts with "vllm-")
    api_keys={
        "vllm_server_url": "http://localhost:8000"  # Local vLLM server
        # Or "https://api.runpod.ai/v1" for cloud-hosted
    }
)
```

## How to Choose

### Choose Cloud LLM API When:

✅ **You want simplicity** - Just need API key, no setup  
✅ **You want best models** - Access to latest GPT-4, Claude, Gemini  
✅ **You want vision support** - Most cloud APIs support images  
✅ **You don't have GPU** - No hardware requirements  
✅ **You want managed service** - No server maintenance  

**Example:**
```python
# Simple - just API key
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",  # Cloud API
    api_keys={"google": "AIzaSy..."}
)
```

### Choose vLLM When:

✅ **You want privacy** - Data stays local  
✅ **You want cost control** - Free (local) or pay-per-use (cloud)  
✅ **You have GPU** - For local inference  
✅ **You want open-source models** - Llama, Qwen, Mistral  
✅ **You want no API costs** - Free local inference  

**Example:**
```python
# Local vLLM - requires server running
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b",  # vLLM (starts with "vllm-")
    api_keys={"vllm_server_url": "http://localhost:8000"}
)
```

## Model Selection Logic

The library automatically detects the provider based on model name:

```python
# Model name determines provider
"gemini-2.0-flash"  → Google API (cloud)
"gpt-4o"            → OpenAI API (cloud)
"claude-3-5-sonnet" → Anthropic API (cloud)
"vllm-llama-3.1-8b" → vLLM (local/cloud-hosted)
```

## Helper Functions

The library provides helper functions to easily differentiate and work with models:

### List Available Models

```python
from od_parse import get_available_models, get_cloud_llm_models, get_vllm_models

# Get all available models
all_models = get_available_models()
print("All available models:", all_models)

# Get only cloud LLM API models
cloud_models = get_cloud_llm_models()
print("Cloud LLM API models:", cloud_models)
# Output: ['gemini-2.0-flash', 'gemini-2.0-flash-exp', 'gpt-4o', ...]

# Get only vLLM models
vllm_models = get_vllm_models()
print("vLLM models:", vllm_models)
# Output: ['vllm-llama-3.1-8b', 'vllm-qwen-2.5-7b', ...]
```

### Check Model Type

```python
from od_parse import is_vllm_model, is_cloud_llm_model

# Check if model is vLLM
if is_vllm_model("vllm-llama-3.1-8b"):
    print("This is a vLLM model")

# Check if model is cloud LLM API
if is_cloud_llm_model("gemini-2.0-flash"):
    print("This is a cloud LLM API model")
```

### With API Keys

```python
from od_parse import get_cloud_llm_models, get_vllm_models

# Get models with provided API keys
cloud_models = get_cloud_llm_models(api_keys={"google": "AIzaSy..."})
vllm_models = get_vllm_models(api_keys={"vllm_server_url": "http://localhost:8000"})
```

## Comparison Table

| Feature | Cloud LLM API | vLLM |
|---------|---------------|------|
| **Model Names** | `gemini-*`, `gpt-*`, `claude-*` | `vllm-*` |
| **Setup** | Just API key | Server setup required |
| **Cost** | Pay per token | Free (local) or cloud costs |
| **Privacy** | Data sent to cloud | 100% local (if local) |
| **Hardware** | No GPU needed | GPU needed (local) |
| **Models** | Latest GPT, Claude, Gemini | Open-source (Llama, Qwen, etc.) |
| **Vision Support** | Yes (most models) | Limited (not yet) |
| **Speed** | Fast (cloud) | Fast (local) or depends on cloud |

## Examples

### Example 1: Use Cloud LLM (Default)

```python
from od_parse import parse_pdf

# Uses cloud LLM API (Gemini 2.0 Flash)
result = parse_pdf(
    "document.pdf",
    api_keys={"google": "AIzaSy..."}
    # llm_model defaults to "gemini-2.0-flash" (cloud API)
)
```

### Example 2: Explicitly Use Cloud LLM

```python
from od_parse import parse_pdf

# Explicitly use cloud LLM API
result = parse_pdf(
    "document.pdf",
    llm_model="gemini-2.0-flash",  # Cloud API model
    api_keys={"google": "AIzaSy..."}
)
```

### Example 3: Use vLLM (Local)

```python
from od_parse import parse_pdf

# Use vLLM (local server)
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b",  # vLLM model (starts with "vllm-")
    api_keys={"vllm_server_url": "http://localhost:8000"}
)
```

### Example 4: Use vLLM (Cloud-Hosted)

```python
from od_parse import parse_pdf

# Use vLLM (cloud-hosted, e.g., RunPod)
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b",  # vLLM model
    api_keys={
        "vllm_server_url": "https://api.runpod.ai/v1",
        "vllm_api_key": "your-runpod-key"
    }
)
```

## Quick Reference

### Cloud LLM API Models

```python
# Google/Gemini
"gemini-2.0-flash"
"gemini-2.0-flash-exp"
"gemini-1.5-pro"
"gemini-1.5-flash"

# OpenAI
"gpt-4o"
"gpt-4o-mini"
"gpt-4-turbo"

# Anthropic/Claude
"claude-3-5-sonnet"
"claude-3-haiku"
```

### vLLM Models

```python
# vLLM models (all start with "vllm-")
"vllm-llama-3.1-8b"
"vllm-llama-3.1-70b"
"vllm-qwen-2.5-7b"
"vllm-mistral-7b"
```

## Troubleshooting

### Wrong Provider Selected

**Problem**: Library using wrong provider

**Solution**: 
- Check model name - cloud APIs don't start with "vllm-"
- vLLM models must start with "vllm-"
- Verify API keys match the provider

### vLLM Not Working

**Problem**: vLLM model not available

**Solution**:
- Check vLLM server is running
- Verify `vllm_server_url` is correct
- Check server health: `curl http://localhost:8000/health`

### Cloud API Not Working

**Problem**: Cloud API model not working

**Solution**:
- Verify API key is correct
- Check API key matches provider (Google key for Gemini, etc.)
- Verify model name is correct

## Next Steps

- See [api_keys_config.md](api_keys_config.md) for API key configuration
- See [vllm_setup_guide.md](vllm_setup_guide.md) for vLLM setup
- See [cloud_vllm_guide.md](cloud_vllm_guide.md) for cloud-hosted vLLM
- See [README.md](../README.md) for general usage

