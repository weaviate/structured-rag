import json
from typing import Optional

from pydantic import BaseModel

from structured_rag.core.ports.llm import LLMPort


class OpenAIAdapter(LLMPort):
    def __init__(self, model_name: str, api_key: str, structured_outputs: bool = False):
        import openai
        self.model_name = model_name
        self.structured_outputs = structured_outputs
        self.client = openai.OpenAI(api_key=api_key)

    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        if self.structured_outputs and output_model is not None:
            response = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Follow the response format instructions."},
                    {"role": "user", "content": prompt}
                ],
                response_format=output_model
            )
            parsed_response = response.choices[0].message.parsed
            return json.dumps({key: value for key, value in parsed_response.__dict__.items()})
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

    def test_connection(self) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "say hello"}
            ]
        )
        return response.choices[0].message.content
