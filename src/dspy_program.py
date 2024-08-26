import dspy
from typing import Optional, Any, Dict
from src.dspy_signatures import GenerateResponse

class dspy_Program(dspy.Module):
    def __init__(self, model_name: str, model_provider: str, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_provider = model_provider
        self.configure_llm(api_key)
        self.generate_response = dspy.ChainOfThought(GenerateResponse)

    def configure_llm(self, api_key: Optional[str] = None):
        if self.model_provider == "ollama":
            llm = dspy.OllamaLocal(model=self.model_name, max_tokens=4000, timeout_s=480)
        elif self.model_provider == "google":
            llm = dspy.Google(model=self.model_name, api_key=api_key)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")

        print("Running LLM connection test (say hello)...")
        print(llm("say hello"))
        dspy.settings.configure(lm=llm)

    def forward(self, test: str, question: str, context: Optional[str] = "") -> Any:
        test_params = self.get_test_parameters(test)
        references = {"context": context, "question": question}
        references = "".join(f"{k}: {v}" for k, v in references.items())
        response = self.generate_response(
            task_instructions=test_params['task_instructions'],
            response_format=test_params['response_format'],
            references=references
        ).response

        return response

    def get_test_parameters(self, test: str) -> Dict[str, str]:
        test_params = {
            "GenerateAnswer": {
                "task_instructions": "Assess the context and answer the question. If the context does not contain sufficient information to answer the question, respond with \"NOT ENOUGH CONTEXT\".",
                "response_format": '{"answer": "string"}'
            },
            "RateContext": {
                "task_instructions": "Assess how well the context helps answer the question.",
                "response_format": '{"context_score": "int (0-5)"}'
            },
            "AssessAnswerability": {
                "task_instructions": "Determine if the question is answerable based on the context.",
                "response_format": '{"answerable_question": "bool"}'
            },
            "ParaphraseQuestions": {
                "task_instructions": "Generate 3 paraphrased versions of the given question.",
                "response_format": '{"paraphrased_questions": ["string", "string", "string"]}'
            },
            "GenerateAnswerWithConfidence": {
                "task_instructions": "Generate an answer with a confidence score.",
                "response_format": '{"Answer": "string", "Confidence": "int (0-5)"}'
            },
            "GenerateAnswersWithConfidence": {
                "task_instructions": "Generate multiple answers with confidence scores.",
                "response_format": '[{"Answer": "string", "Confidence": "int (0-5)"}, ...]'
            }
        }
        
        if test not in test_params:
            raise ValueError(f"Unsupported test: {test}")
        
        return test_params[test]