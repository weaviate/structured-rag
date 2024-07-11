from typing import Optional
import ollama
import google.generativeai as genai
from fstring_prompts import *

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
        if test == "GenerateAnswer":
            prompt = get_generate_answer_prompt(context, question)
        elif test == "RateContext":
            prompt = get_rate_context_prompt(context, question)
        elif test == "AssessAnswerability":
            prompt = get_assess_answerability_prompt(context, question)
        elif test == "ParaphraseQuestions":
            prompt = get_paraphrase_questions_prompt(question)
        elif test == "GenerateAnswerWithConfidence":
            prompt = get_generate_answer_with_confidence_prompt(context, question)
        elif test == "GenerateAnswersWithConfidence":
            prompt = get_generate_answers_with_confidence_prompt(context, question)
        else:
            raise ValueError(f"Unsupported test: {test}")

        if self.model_provider == "ollama":
            response = ollama.chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
            return response['message']['content']
        elif self.model_provider == "google":
            response = self.model.generate_content(prompt)
            return response.text