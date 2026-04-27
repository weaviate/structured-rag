from typing import Optional

from pydantic import BaseModel

from structured_rag.core.ports.llm import LLMPort


class AnthropicAdapter(LLMPort):
    def __init__(self, model_name: str, api_key: str):
        import anthropic
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    def test_connection(self) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            messages=[{"role": "user", "content": "say hello"}]
        )
        return response.content[0].text
