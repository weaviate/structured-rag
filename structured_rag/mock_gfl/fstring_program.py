from typing import Optional, Dict
import ollama
import google.generativeai as genai
import openai
from structured_rag.mock_gfl.fstring_prompts import get_prompt
from pydantic import BaseModel

class fstring_Program():
    def __init__(self,
                 test_params: Dict[str, str],
                 model_name: str, model_provider: str, api_key: Optional[str]) -> None:
        self.test_params = test_params
        self.model_name = model_name
        self.model_provider = model_provider
        self.structured_outputs = True # change to constructor argument
        if self.model_provider == "google":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif self.model_provider == "openai":
            import openai
            self.model = openai.OpenAI(api_key=api_key)
        elif self.model_provider == "anthropic":
            import anthropic
            self.model = anthropic.Anthropic(api_key=api_key)
        print("Running LLM connection test (say hello)...")
        print(self.test_connection())

    def test_connection(self) -> str:
        # For now this tests without structured outputs, could be an idea to add this
        connection_prompt = "say hello"
        if self.model_provider == "google":
            # how to add a BaseModel to this?
            response = self.model.generate_content(connection_prompt)
            return response.text
        elif self.model_provider == "ollama":
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": connection_prompt}])
            return response['message']['content']
        elif self.model_provider == "openai":
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": connection_prompt}
                ]
            )
            return response.choices[0].message.content
        elif self.model_provider == "anthropic":
            response = self.model.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": connection_prompt}
                ]
            )
            return response.content[0].text

    def forward(self, output_model: Optional[BaseModel], test: str, 
                context: str = "", question: str = "", answer: str = "", 
                ) -> str:
        references: Dict[str, str] = {}
        if test != "ParaphraseQuestions":
            references = {"context": context, "question": question}
        elif test == "RAGAS":
            references = {"context": context, "question": question, "answer": answer}
        else:
            references = {"question": question}

        prompt = get_prompt(test, references, self.test_params)

        if self.model_provider == "ollama":
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']
        elif self.model_provider == "google":
            if self.structured_outputs:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json", response_schema=output_model
                    ),
                )
            else:
                response = self.model.generate_content(prompt)
            return response.text
        elif self.model_provider == "openai":
            response = self.model.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        elif self.model_provider == "anthropic":
            response = self.model.messages.create(
                model=self.model_name,
                max_tokens=2048,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text