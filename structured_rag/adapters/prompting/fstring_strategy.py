from typing import Optional, Dict

from pydantic import BaseModel

from structured_rag.core.ports.llm import LLMPort
from structured_rag.core.ports.prompting import PromptingStrategy


def _build_prompt(task_name: str, references: Dict[str, str], task_params: Dict[str, str]) -> str:
    """Build an f-string prompt from task params and references."""
    references_str = ' | '.join(f"{k}: {v}" for k, v in references.items())
    return f"""Instructions: {task_params['task_instructions']}
    References: {references_str}
    Output the result as a JSON string with the following format: {task_params['response_format']}
    IMPORTANT!! Do not start the JSON with ```json or end it with ```."""


class FStringStrategy(PromptingStrategy):
    """f-String prompting: builds a prompt with inline instructions and calls an LLMPort."""

    def __init__(self, llm: LLMPort):
        self.llm = llm

    def run(self, task_name: str, task_params: dict,
            output_model: Optional[type[BaseModel]] = None,
            context: str = "", question: str = "", answer: str = "") -> str:

        if task_name == "ParaphraseQuestions":
            references = {"question": question}
        elif task_name == "RAGAS":
            references = {"context": context, "question": question, "answer": answer}
        else:
            references = {"context": context, "question": question}

        prompt = _build_prompt(task_name, references, task_params)
        return self.llm.generate(prompt, output_model=output_model)
