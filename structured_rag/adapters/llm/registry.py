from structured_rag.core.ports.llm import LLMPort


def get_llm_adapter(provider: str, model_name: str, api_key: str = "",
                     structured_outputs: bool = False, **kwargs) -> LLMPort:
    """Factory: create the right LLM adapter for a given provider."""

    if provider == "openai":
        from structured_rag.adapters.llm.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(model_name, api_key, structured_outputs=structured_outputs)

    elif provider == "anthropic":
        from structured_rag.adapters.llm.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(model_name, api_key)

    elif provider == "google":
        from structured_rag.adapters.llm.google_adapter import GoogleAdapter
        return GoogleAdapter(model_name, api_key, structured_outputs=structured_outputs)

    elif provider == "ollama":
        from structured_rag.adapters.llm.ollama_adapter import OllamaAdapter
        return OllamaAdapter(model_name)

    elif provider == "ollama_cloud":
        from structured_rag.adapters.llm.ollama_adapter import OllamaCloudAdapter
        return OllamaCloudAdapter(model_name, api_key)

    elif provider == "modal_vllm":
        from structured_rag.adapters.llm.modal_vllm_adapter import ModalVLLMAdapter
        return ModalVLLMAdapter(
            endpoint_url=model_name,  # config passes the URL as "model"
            api_key=api_key,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. "
            f"Supported: openai, anthropic, google, ollama, ollama_cloud, modal_vllm"
        )
