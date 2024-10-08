import dspy
from typing import Optional, Any, Dict
from structured_rag.mock_gfl.dspy_signatures import GenerateResponse, OPRO_JSON
from pydantic import BaseModel

class dspy_Program(dspy.Module):
    def __init__(self, 
                 test_params: Dict[str, str],
                 model_name: str, model_provider: str, api_key: Optional[str] = None,
                 use_OPRO_JSON: bool = False) -> None:
        super().__init__()
        self.test_params = test_params
        self.model_name = model_name
        self.model_provider = model_provider
        self.use_OPRO_JSON = use_OPRO_JSON
        self.configure_llm(api_key)
        # ToDo, Interface `TypedPredictor` here
        if self.use_OPRO_JSON:
            self.generate_response = dspy.Predict(OPRO_JSON)
        else:
            self.generate_response = dspy.ChainOfThought(GenerateResponse)
        
    def configure_llm(self, api_key: Optional[str] = None):
        if self.model_provider == "ollama":
            llm = dspy.OllamaLocal(model=self.model_name, max_tokens=4000, timeout_s=480)
        elif self.model_provider == "google":
            llm = dspy.Google(model=self.model_name, api_key=api_key)
        elif self.model_provider == "openai":
            import openai

            openai.api_key = api_key
            llm = dspy.OpenAI(model=self.model_name)
        elif self.model_provider == "anthropic":
            import anthropic
            llm = dspy.Claude(model=self.model_name, api_key=api_key)
        # ToDo, add Cohere
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        print("Running LLM connection test (say hello)...")
        print(llm("say hello"))
        dspy.settings.configure(lm=llm)

    # Note, this needs to be cleaned up with the abstraction around DSPy / LLM APIs
    def forward(self, output_model: Optional[BaseModel], test: str, question: str, context: Optional[str] = "", answer: Optional[str] = "") -> Any:
        references = {"context": context, "question": question, "answer": answer}
        references = "".join(f"{k}: {v}" for k, v in references.items())
        response = self.generate_response(
            task_instructions=self.test_params['task_instructions'],
            response_format=self.test_params['response_format'],
            references=references
        ).response

        return response