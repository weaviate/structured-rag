from typing import Optional, Dict
import ollama
import google.generativeai as genai
from fstring_prompts import get_prompt

class fstring_Program():
    def __init__(self, model_name: str, model_provider: str, api_key: Optional[str]) -> None:
        self.model_name = model_name
        self.model_provider = model_provider
        if self.model_provider == "google":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
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

    def forward(self, test: str, context: str = "", question: str = "") -> str:
        references: Dict[str, str] = {}
        if test != "ParaphraseQuestions":
            references = {"context": context, "question": question}
        else:
            references = {"question": question}

        prompt = get_prompt(test, references)

        if self.model_provider == "ollama":
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']
        elif self.model_provider == "google":
            response = self.model.generate_content(prompt)
            return response.text