from typing import Optional

from pydantic import BaseModel

from structured_rag.core.ports.llm import LLMPort


class OllamaAdapter(LLMPort):
    """Adapter for local Ollama instances."""

    def __init__(self, model_name: str):
        import ollama
        self.ollama = ollama
        self.model_name = model_name

    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        response = self.ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    def test_connection(self) -> str:
        response = self.ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": "say hello"}]
        )
        return response['message']['content']


class OllamaCloudAdapter(LLMPort):
    """Adapter for Ollama Cloud (hosted at ollama.com)."""

    def __init__(self, model_name: str, api_key: str):
        import ollama
        self.model_name = model_name
        self.client = ollama.Client(
            host="https://ollama.com",
            headers={"Authorization": f"Bearer {api_key}"}
        )

    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']

    def test_connection(self) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": "say hello"}]
        )
        return response['message']['content']
