from typing import Optional

from pydantic import BaseModel

from structured_rag.core.ports.llm import LLMPort


class GoogleAdapter(LLMPort):
    def __init__(self, model_name: str, api_key: str, structured_outputs: bool = False):
        import google.generativeai as genai
        self.genai = genai
        self.model_name = model_name
        self.structured_outputs = structured_outputs
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        if self.structured_outputs and output_model is not None:
            response = self.model.generate_content(
                prompt,
                generation_config=self.genai.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=output_model
                ),
            )
        else:
            response = self.model.generate_content(prompt)
        return response.text

    def test_connection(self) -> str:
        response = self.model.generate_content("say hello")
        return response.text
