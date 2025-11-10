# Using Cloud-Hosted vLLM Servers

This guide explains how to use cloud-hosted vLLM servers with od-parse.

## What is Cloud-Hosted vLLM?

Cloud-hosted vLLM is a vLLM server running on cloud infrastructure (AWS, GCP, Azure, etc.) that you can access via API. This gives you:
- **No local GPU needed**: Run on cloud GPUs
- **Scalability**: Handle more requests
- **Cost-effective**: Pay only for what you use
- **Easy setup**: No local installation required

## Supported Cloud vLLM Providers

### 1. RunPod (Recommended)
- **URL**: `https://api.runpod.ai/v1`
- **Setup**: Deploy vLLM on RunPod serverless GPUs
- **Cost**: Pay per second of GPU usage
- **Documentation**: https://docs.runpod.io/

### 2. Together AI
- **URL**: `https://api.together.xyz/v1`
- **Setup**: Use Together AI's hosted vLLM
- **Cost**: Pay per token
- **Documentation**: https://docs.together.ai/

### 3. Replicate
- **URL**: Custom endpoint per deployment
- **Setup**: Deploy vLLM model on Replicate
- **Cost**: Pay per request
- **Documentation**: https://replicate.com/docs

### 4. AWS SageMaker / GCP Vertex AI / Azure ML
- **URL**: Custom endpoint from your deployment
- **Setup**: Deploy vLLM on your cloud infrastructure
- **Cost**: Pay for compute resources
- **Documentation**: Cloud provider specific

### 5. Self-Hosted Cloud Instance
- **URL**: Your cloud server URL (e.g., `https://vllm.yourdomain.com`)
- **Setup**: Deploy vLLM on your own cloud server
- **Cost**: Pay for cloud instance
- **Documentation**: vLLM deployment guides

## Configuration

### Method 1: Environment Variable (Recommended)

Set the `VLLM_SERVER_URL` environment variable:

```bash
# For RunPod
export VLLM_SERVER_URL="https://api.runpod.ai/v1"

# For Together AI
export VLLM_SERVER_URL="https://api.together.xyz/v1"

# For custom cloud server
export VLLM_SERVER_URL="https://vllm.yourdomain.com/v1"

# Optional: Set API key if required
export VLLM_API_KEY="your-api-key"
```

Then use in code:
```python
from od_parse import parse_pdf

# Uses cloud-hosted vLLM automatically
result = parse_pdf("document.pdf", llm_model="vllm-llama-3.1-8b")
```

### Method 2: In Code Configuration

```python
import os
from od_parse import parse_pdf

# Set cloud vLLM server URL
os.environ["VLLM_SERVER_URL"] = "https://api.runpod.ai/v1"
os.environ["VLLM_API_KEY"] = "your-api-key"  # If required

# Use with cloud vLLM
result = parse_pdf("document.pdf", llm_model="vllm-llama-3.1-8b")
```

### Method 3: Custom Model Configuration

You can create a custom vLLM model configuration:

```python
from od_parse.config.llm_config import LLMConfig, LLMProvider, LLMModelConfig, DocumentComplexity

# Get LLM config
llm_config = LLMConfig()

# Add custom cloud vLLM model
llm_config.models["vllm-cloud-llama"] = LLMModelConfig(
    provider=LLMProvider.VLLM,
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_key_env="",
    max_tokens=4096,
    temperature=0.1,
    supports_vision=False,
    cost_per_1k_tokens=0.0,
    context_window=8192,
    recommended_complexity=[DocumentComplexity.SIMPLE, DocumentComplexity.MODERATE],
    vllm_server_url="https://api.runpod.ai/v1",  # Cloud server URL
    vllm_api_key="your-api-key"  # Optional API key
)

# Use custom model
from od_parse.llm import LLMDocumentProcessor
processor = LLMDocumentProcessor(model_id="vllm-cloud-llama")
```

## Examples

### Example 1: RunPod

```python
import os
from od_parse import parse_pdf

# Configure RunPod vLLM
os.environ["VLLM_SERVER_URL"] = "https://api.runpod.ai/v1"
os.environ["VLLM_API_KEY"] = "your-runpod-api-key"

# Parse with cloud vLLM
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"
)

print(result['parsed_data'])
```

### Example 2: Together AI

```python
import os
from od_parse import parse_pdf

# Configure Together AI vLLM
os.environ["VLLM_SERVER_URL"] = "https://api.together.xyz/v1"
os.environ["VLLM_API_KEY"] = "your-together-api-key"

# Parse with cloud vLLM
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"
)
```

### Example 3: Custom Cloud Server

```python
import os
from od_parse import parse_pdf

# Configure custom cloud vLLM server
os.environ["VLLM_SERVER_URL"] = "https://vllm.yourdomain.com/v1"
os.environ["VLLM_API_KEY"] = "your-api-key"  # If required

# Parse with cloud vLLM
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"
)
```

### Example 4: AWS SageMaker Endpoint

```python
import os
from od_parse import parse_pdf

# Configure AWS SageMaker vLLM endpoint
os.environ["VLLM_SERVER_URL"] = "https://your-endpoint.sagemaker.aws.amazon.com/v1"
os.environ["VLLM_API_KEY"] = "your-aws-key"  # If required

# Parse with cloud vLLM
result = parse_pdf(
    "document.pdf",
    llm_model="vllm-llama-3.1-8b"
)
```

## Setting Up Cloud vLLM Servers

### RunPod Setup

1. **Create RunPod Account**: https://www.runpod.io/
2. **Deploy vLLM Template**:
   - Go to Templates
   - Search for "vLLM"
   - Deploy template
3. **Get Endpoint URL**: Copy the API endpoint URL
4. **Get API Key**: Copy your RunPod API key
5. **Configure**:
   ```bash
   export VLLM_SERVER_URL="https://api.runpod.ai/v1"
   export VLLM_API_KEY="your-runpod-api-key"
   ```

### Together AI Setup

1. **Create Together AI Account**: https://www.together.ai/
2. **Get API Key**: From dashboard
3. **Configure**:
   ```bash
   export VLLM_SERVER_URL="https://api.together.xyz/v1"
   export VLLM_API_KEY="your-together-api-key"
   ```

### Self-Hosted Cloud Setup

1. **Deploy vLLM on Cloud Server** (AWS EC2, GCP Compute, Azure VM):
   ```bash
   # On your cloud server
   pip install vllm
   python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Meta-Llama-3.1-8B-Instruct \
       --host 0.0.0.0 \
       --port 8000
   ```

2. **Configure Firewall**: Open port 8000 (or your chosen port)

3. **Get Public IP/URL**: Note your server's public IP or domain

4. **Configure**:
   ```bash
   export VLLM_SERVER_URL="http://your-server-ip:8000"
   # Or with domain
   export VLLM_SERVER_URL="https://vllm.yourdomain.com"
   ```

## Authentication

Most cloud vLLM providers require API keys:

```python
import os

# Set API key
os.environ["VLLM_API_KEY"] = "your-api-key"

# Or in code
from od_parse.config.llm_config import get_llm_config
llm_config = get_llm_config()
model_config = llm_config.models["vllm-llama-3.1-8b"]
model_config.vllm_api_key = "your-api-key"
```

## Testing Connection

Test if your cloud vLLM server is accessible:

```python
import requests

server_url = "https://api.runpod.ai/v1"  # Your cloud server URL
health_url = f"{server_url.rstrip('/')}/health"

try:
    response = requests.get(health_url, timeout=5)
    if response.status_code == 200:
        print("✅ Cloud vLLM server is accessible")
    else:
        print(f"❌ Server returned status {response.status_code}")
except Exception as e:
    print(f"❌ Cannot connect to cloud vLLM server: {e}")
```

## Troubleshooting

### Connection Errors

**Problem**: Cannot connect to cloud vLLM server

**Solutions**:
1. Check server URL is correct
2. Verify API key is set (if required)
3. Check firewall/network settings
4. Test with `curl`:
   ```bash
   curl https://api.runpod.ai/v1/health
   ```

### Authentication Errors

**Problem**: 401 Unauthorized

**Solutions**:
1. Verify API key is correct
2. Check API key format
3. Ensure API key is set in environment variable

### Model Not Found

**Problem**: Model not available on cloud server

**Solutions**:
1. Check model name matches cloud provider's model name
2. Verify model is deployed on cloud server
3. Use model name from cloud provider's documentation

### Timeout Errors

**Problem**: Requests timing out

**Solutions**:
1. Check network connection
2. Increase timeout in code
3. Verify cloud server is running
4. Check cloud provider status

## Cost Comparison

| Provider | Cost Model | Best For |
|----------|-----------|----------|
| **RunPod** | Pay per second | Sporadic usage, cost-effective |
| **Together AI** | Pay per token | High throughput |
| **Replicate** | Pay per request | Simple deployments |
| **Self-Hosted** | Pay for instance | High volume, predictable usage |
| **AWS/GCP/Azure** | Pay for compute | Enterprise, compliance |

## Best Practices

1. **Use Environment Variables**: Keep API keys secure
2. **Test Connection First**: Verify server is accessible
3. **Monitor Costs**: Track usage on cloud provider
4. **Use Appropriate Model Size**: Match model to your needs
5. **Handle Errors Gracefully**: Implement retry logic
6. **Cache Results**: Reduce API calls when possible

## Next Steps

- See [vllm_setup_guide.md](vllm_setup_guide.md) for local vLLM setup
- See [README.md](../README.md) for general usage
- Check cloud provider documentation for specific setup

