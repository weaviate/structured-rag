from typing import Optional

import dspy
from pydantic import BaseModel

from structured_rag.core.ports.prompting import PromptingStrategy


class GenerateResponse(dspy.Signature):
    """Follow the task_instructions (Input Field) and generate the response (Output Field) according to the output format given by response_format (Input Field). You will be given references from (Task-Specific Input Field)."""
    task_instructions = dspy.InputField(desc="(Input Field)")
    response_format = dspy.InputField(desc="(Input Field)")
    references = dspy.InputField(desc="Task-Specific Input Field")
    response = dspy.OutputField(desc="(Output Field)")


class OPRO_JSON(dspy.Signature):
    """Carefully interpret the task_instructions provided in the Input Field, synthesizing the necessary information from the Task-Specific Input Field to construct a response. Your response should be formatted exclusively in JSON and must conform precisely to the structure dictated by the response_format Input Field. Ensure that your JSON-formatted response is devoid of extraneous characters or elements, such as markdown code block ticks (```), and includes only the keys specified by the response_format. Your attention to detail in following these instructions is paramount for the accuracy and relevance of your output."""
    task_instructions = dspy.InputField(desc="(Input Field)")
    response_format = dspy.InputField(desc="(Input Field)")
    references = dspy.InputField(desc="Task-Specific Input Field")
    response = dspy.OutputField(desc="(Output Field)")


def _configure_dspy_lm(model_provider: str, model_name: str, api_key: Optional[str] = None):
    """Configure DSPy's global LM setting for the given provider."""
    if model_provider == "ollama":
        lm = dspy.OllamaLocal(model=model_name, max_tokens=4000, timeout_s=480)
    elif model_provider == "ollama_cloud":
        lm = dspy.OpenAI(model=model_name, api_key=api_key,
                          api_base="https://ollama.com/v1/", model_type="chat", max_tokens=4000)
    elif model_provider == "google":
        lm = dspy.Google(model=model_name, api_key=api_key)
    elif model_provider == "openai":
        import openai
        openai.api_key = api_key
        lm = dspy.OpenAI(model=model_name)
    elif model_provider == "anthropic":
        lm = dspy.Claude(model=model_name, api_key=api_key)
    else:
        raise ValueError(f"Unsupported DSPy provider: {model_provider}")

    dspy.settings.configure(lm=lm)
    return lm


class DSpyStrategy(PromptingStrategy):
    """DSPy prompting strategy. Manages its own LM configuration internally."""

    def __init__(self, model_name: str, model_provider: str,
                 api_key: Optional[str] = None, use_opro: bool = False):
        self.model_name = model_name
        self.model_provider = model_provider
        self.use_opro = use_opro

        lm = _configure_dspy_lm(model_provider, model_name, api_key)
        print("Running LLM connection test (say hello)...")
        print(lm("say hello"))

        if use_opro:
            self.predictor = dspy.Predict(OPRO_JSON)
        else:
            self.predictor = dspy.ChainOfThought(GenerateResponse)

    def run(self, task_name: str, task_params: dict,
            output_model: Optional[type[BaseModel]] = None,
            context: str = "", question: str = "", answer: str = "") -> str:

        references = {"context": context, "question": question, "answer": answer}
        references_str = "".join(f"{k}: {v}" for k, v in references.items())

        response = self.predictor(
            task_instructions=task_params['task_instructions'],
            response_format=task_params['response_format'],
            references=references_str
        ).response

        return response
