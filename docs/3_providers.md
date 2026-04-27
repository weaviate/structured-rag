# Providers

StructuredRAG supports 6 LLM providers. Each provider is implemented as an adapter behind the `LLMPort` interface, so adding new ones is straightforward.

## Supported Providers

### OpenAI

```yaml
provider: openai
model: gpt-5.4-nano
api_key_env: OPENAI_API_KEY
```

Supports structured outputs via `fstring_structured` strategy, which uses OpenAI's `response_format` parameter for schema-constrained generation.

### Anthropic

```yaml
provider: anthropic
model: claude-sonnet-4-20250514
api_key_env: ANTHROPIC_API_KEY
```

### Google Gemini

```yaml
provider: google
model: gemini-1.5-pro
api_key_env: GOOGLE_API_KEY
```

Supports structured outputs via `fstring_structured` strategy, which uses `response_mime_type="application/json"` with a `response_schema`.

### Ollama (local)

```yaml
provider: ollama
model: llama3:8b-instruct-q4_0
```

No API key needed. Requires a running Ollama instance on localhost.

### Ollama Cloud

```yaml
provider: ollama_cloud
model: llama3:8b-instruct-q4_0
api_key_env: OLLAMA_API_KEY
```

Hosted Ollama at `ollama.com`.

### Modal vLLM

```yaml
provider: modal_vllm
model: https://YOUR_MODAL_ENDPOINT_URL
api_key_env: MODAL_API_KEY
use_outlines: true
use_llama3_template: true
```

Calls a self-deployed Modal endpoint running vLLM with optional Outlines structured decoding.

**Setup:**

1. Deploy the Modal endpoint:
   ```bash
   cd structured_rag/adapters/llm/modal_deployment
   modal run download_llama.py
   modal deploy vllm_outlines_setup.py
   ```
2. Set the endpoint URL as the `model` field in your config.

**Provider-specific options:**

| Option | Description | Default |
|---|---|---|
| `use_outlines` | Enable Outlines JSON schema-constrained decoding | `true` |
| `use_llama3_template` | Wrap prompts in Llama 3 chat template | `true` |

## Adding a New Provider

1. Create `structured_rag/adapters/llm/your_adapter.py` implementing `LLMPort`:

   ```python
   from structured_rag.core.ports.llm import LLMPort

   class YourAdapter(LLMPort):
       def __init__(self, model_name: str, api_key: str):
           ...

       def generate(self, prompt, output_model=None):
           ...

       def test_connection(self):
           ...
   ```

2. Register it in `structured_rag/adapters/llm/registry.py`:

   ```python
   elif provider == "your_provider":
       from structured_rag.adapters.llm.your_adapter import YourAdapter
       return YourAdapter(model_name, api_key)
   ```

3. Use it in your config:

   ```yaml
   provider: your_provider
   model: your-model-name
   api_key_env: YOUR_API_KEY
   ```
