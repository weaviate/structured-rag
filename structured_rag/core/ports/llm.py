from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class LLMPort(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        """Send prompt to LLM, return raw text response.

        output_model is passed for providers that support structured outputs
        (e.g. OpenAI response_format, Google response_schema).
        Adapters that don't support structured outputs simply ignore it.
        """
        ...

    @abstractmethod
    def test_connection(self) -> str:
        """Quick health check — send 'say hello' and return the response."""
        ...
