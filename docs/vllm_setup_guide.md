# vLLM Setup Guide for Local Inference (Optional)

This guide explains how to optionally set up and use vLLM (fast LLM inference library) for local document parsing with od-parse.

**Note**: vLLM is **optional**. od-parse works perfectly fine with cloud APIs (OpenAI, Anthropic, Google). vLLM is only needed if you want local inference for privacy or cost reasons.

## What is vLLM?

vLLM is a fast and easy-to-use library for LLM inference and serving. It provides:
- **High throughput**: Optimized inference engine
- **Local or cloud processing**: Run locally or on cloud infrastructure
- **OpenAI-compatible API**: Easy integration
- **Multiple model support**: Llama, Qwen, Mistral, and more

**Note**: You can use vLLM either:
- **Locally**: Run on your own machine (requires GPU)
- **Cloud-hosted**: Use cloud providers like RunPod, Together AI, etc. (see [cloud_vllm_guide.md](cloud_vllm_guide.md))

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (recommended: 16GB+ VRAM for 7B models, 40GB+ for 70B models)
- **CPU**: Modern CPU (can run on CPU but much slower)
- **RAM**: 16GB+ recommended

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- pip

## Installation Options

### Option 1: Cloud-Hosted vLLM (Easiest - No Setup Required)

Use cloud-hosted vLLM servers (RunPod, Together AI, etc.). No local installation needed!

**See [cloud_vllm_guide.md](cloud_vllm_guide.md) for detailed setup.**

Quick example:
```python
import os
os.environ["VLLM_SERVER_URL"] = "https://api.runpod.ai/v1"
os.environ["VLLM_API_KEY"] = "your-api-key"

from od_parse import parse_pdf
result = parse_pdf("document.pdf", llm_model="vllm-llama-3.1-8b")
```

### Option 2: Local vLLM Server (Recommended for Local Use)

This is the best way to use vLLM locally. Run vLLM as a server with OpenAI-compatible API.

#### Step 1: Install vLLM

```bash
pip install vllm
```

Or with specific CUDA version:
```bash
# For CUDA 11.8
pip install vllm

# For CUDA 12.1
VLLM_USE_CUDA_12_1=1 pip install vllm
```

#### Step 2: Start vLLM Server

```bash
# For Llama 3.1 8B (recommended for most users)
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000

# For Qwen 2.5 7B (good alternative)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --port 8000

# For Mistral 7B
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --port 8000
```

The server will:
- Download the model on first run (can take time)
- Start serving on `http://localhost:8000`
- Provide OpenAI-compatible API at `/v1/chat/completions`

#### Step 3: Configure od-parse

Set environment variable:
```bash
export VLLM_SERVER_URL="http://localhost:8000"
```

Or use in code:
```python
from od_parse import parse_pdf

# Use vLLM model
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"  # or vllm-qwen-2.5-7b, vllm-mistral-7b
)
```

### Option 2: Direct vLLM Client

Use vLLM directly in Python (loads model in memory).

```bash
pip install vllm
```

Then use in code:
```python
from od_parse import parse_pdf

# vLLM will be used automatically if server is not available
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"
)
```

**Note**: Direct client loads the model into memory, which can be slow on first use.

## Available vLLM Models

The following vLLM models are pre-configured:

| Model ID | Model Name | Size | Context Window | Best For |
|----------|-----------|------|----------------|----------|
| `vllm-llama-3.1-8b` | Meta-Llama-3.1-8B-Instruct | 8B | 8K | Simple to moderate documents |
| `vllm-llama-3.1-70b` | Meta-Llama-3.1-70B-Instruct | 70B | 8K | Complex documents (requires 40GB+ VRAM) |
| `vllm-qwen-2.5-7b` | Qwen2.5-7B-Instruct | 7B | 32K | Long documents, multilingual |
| `vllm-mistral-7b` | Mistral-7B-Instruct-v0.2 | 7B | 32K | General purpose |

## Usage Examples

### Basic Usage

```python
from od_parse import parse_pdf

# Parse with vLLM (auto-detects if server is running)
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"
)

print(result['parsed_data'])
```

### With Custom Server URL

```python
import os
from od_parse import parse_pdf

# Set custom server URL
os.environ["VLLM_SERVER_URL"] = "http://localhost:8001"

result = parse_pdf(
    "document.pdf",
    llm_model="vllm-qwen-2.5-7b"
)
```

### Using LLMDocumentProcessor Directly

```python
from od_parse.llm import LLMDocumentProcessor
from od_parse.parser import parse_pdf as core_parse

# Parse PDF first
parsed_data = core_parse("document.pdf")

# Process with vLLM
processor = LLMDocumentProcessor(model_id="vllm-llama-3.1-8b")
enhanced_data = processor.process_document(parsed_data)

print(enhanced_data['llm_analysis'])
```

## Configuration

### Environment Variables

- `VLLM_SERVER_URL`: vLLM server URL 
  - **Local**: `http://localhost:8000` (default)
  - **Cloud**: `https://api.runpod.ai/v1`, `https://api.together.xyz/v1`, or your custom URL
- `VLLM_API_KEY`: Optional API key for vLLM server (required for most cloud providers)

### Cloud-Hosted vLLM

For cloud-hosted vLLM servers, set the server URL to your cloud provider's endpoint:

```bash
# RunPod
export VLLM_SERVER_URL="https://api.runpod.ai/v1"
export VLLM_API_KEY="your-runpod-api-key"

# Together AI
export VLLM_SERVER_URL="https://api.together.xyz/v1"
export VLLM_API_KEY="your-together-api-key"

# Custom cloud server
export VLLM_SERVER_URL="https://vllm.yourdomain.com/v1"
export VLLM_API_KEY="your-api-key"
```

**See [cloud_vllm_guide.md](cloud_vllm_guide.md) for detailed cloud setup.**

### Model Configuration

You can customize vLLM models in `od_parse/config/llm_config.py`:

```python
"vllm-custom-model": LLMModelConfig(
    provider=LLMProvider.VLLM,
    model_name="your-model/name",
    api_key_env="",
    max_tokens=4096,
    temperature=0.1,
    supports_vision=False,
    cost_per_1k_tokens=0.0,
    context_window=8192,
    vllm_server_url="http://localhost:8000",  # Optional custom URL
    vllm_api_key=None  # Optional API key
)
```

## Troubleshooting

### Server Not Accessible

If you get "vLLM server not accessible" error:

1. **Check if server is running**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check server logs** for errors

3. **Verify port**: Make sure port 8000 is not in use

4. **Check firewall**: Ensure localhost connections are allowed

### CUDA Out of Memory

If you get CUDA OOM errors:

1. **Use smaller model**: Switch to 7B/8B models instead of 70B
2. **Reduce batch size**: Add `--max-num-batched-tokens` flag
3. **Use CPU**: vLLM can run on CPU (much slower)

### Model Download Issues

If model download fails:

1. **Check internet connection**
2. **Use HuggingFace token**: Set `HF_TOKEN` environment variable
3. **Download manually**: Download model to local path and use `--model /path/to/model`

### Performance Issues

- **Use GPU**: CPU inference is 10-100x slower
- **Use tensor parallelism**: For multi-GPU setups
- **Optimize batch size**: Adjust based on your hardware

## Advanced Configuration

### Multi-GPU Setup

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --tensor-parallel-size 2 \
    --port 8000
```

### Custom Model Path

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/local/model \
    --port 8000
```

### With API Key Authentication

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --api-key your-api-key \
    --port 8000
```

Then set in code:
```python
os.environ["VLLM_API_KEY"] = "your-api-key"
```

## Comparison: vLLM vs Cloud APIs

| Feature | vLLM (Local) | Cloud APIs (OpenAI/Anthropic) |
|---------|--------------|------------------------------|
| **Cost** | Free (after hardware) | Pay per token |
| **Privacy** | 100% local | Data sent to cloud |
| **Speed** | Fast (local network) | Depends on internet |
| **Setup** | Requires GPU | Just API key |
| **Maintenance** | You manage | Managed service |
| **Model Choice** | Any open model | Provider's models |

## Best Practices

1. **Start with server mode**: Easier to manage and restart
2. **Use appropriate model size**: Match model to your GPU
3. **Monitor GPU usage**: Use `nvidia-smi` to check utilization
4. **Cache models**: Models are cached after first download
5. **Use health checks**: Monitor server health in production

## Next Steps

- See [README.md](../README.md) for general usage
- See [technical_reference.md](technical_reference.md) for API details
- Check [vLLM documentation](https://docs.vllm.ai/) for advanced features

