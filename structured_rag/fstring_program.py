from typing import Optional, Dict
import ollama
import google.generativeai as genai
import openai
from structured_rag.fstring_prompts import get_prompt

class fstring_Program():
    def __init__(self,
                 test_params: Dict[str, str],
                 model_name: str, model_provider: str, api_key: Optional[str]) -> None:
        self.test_params = test_params
        self.model_name = model_name
        self.model_provider = model_provider
        if self.model_provider == "google":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
        elif self.model_provider == "openai":
            import openai
            self.model = openai.OpenAI(api_key=api_key)
        print("Running LLM connection test (say hello)...")
        print(self.test_connection())

    def test_connection(self) -> str:
        connection_prompt = "say hello"
        if self.model_provider == "google":
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

    def forward(self, test: str, context: str = "", question: str = "") -> str:
        references: Dict[str, str] = {}
        if test != "ParaphraseQuestions":
            references = {"context": context, "question": question}
        else:
            references = {"question": question}

        prompt = get_prompt(test, references, self.test_params)

        if self.model_provider == "ollama":
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']
        elif self.model_provider == "google":
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