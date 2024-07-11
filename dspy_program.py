import dspy
from typing import Optional, Any
from dspy_signatures import *

class dspy_Program(dspy.Module):
    def __init__(self, model_name: str, model_provider: str, api_key: Optional[str] = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.model_provider = model_provider
        self.configure_llm(api_key)
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.rate_context = dspy.Predict(RateContext)
        self.assess_answerability = dspy.Predict(AssessAnswerability)
        self.paraphrase_questions = dspy.Predict(ParaphraseQuestions)
        self.generate_answer_with_confidence = dspy.Predict(GenerateAnswerWithConfidence)
        self.generate_answers_with_confidence = dspy.Predict(GenerateAnswersWithConfidence)

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
        if test == "GenerateAnswer":
            return self.answer_question(context=context, question=question).answer
        elif test == "RateContext":
            return self.rate_context(context=context, question=question).context_score
        elif test == "AssessAnswerability":
            return self.assess_answerability(context=context, question=question).answerable_question
        elif test == "ParaphraseQuestions":
            return self.paraphrase_questions(question=question).paraphrased_questions
        elif test == "GenerateAnswerWithConfidence":
            return self.generate_answer_with_confidence(context=context, question=question).answer_with_confidence
        elif test == "GenerateAnswersWithConfidence":
            return self.generate_answers_with_confidence(context=context, question=question).answers_with_confidence
        else:
            raise ValueError(f"Unsupported test: {test}")