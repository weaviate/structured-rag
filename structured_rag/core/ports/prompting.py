from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class PromptingStrategy(ABC):
    """Abstract interface for prompting strategies (f-string, DSPy, etc.)."""

    @abstractmethod
    def run(self, task_name: str, task_params: dict,
            output_model: Optional[type[BaseModel]] = None,
            context: str = "", question: str = "", answer: str = "") -> str:
        """Execute the prompting strategy for a given task, return raw LLM output string."""
        ...
