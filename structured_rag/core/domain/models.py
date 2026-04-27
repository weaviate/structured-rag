from pydantic import BaseModel, create_model
from enum import Enum
from typing import Type, List


class PromptWithResponse(BaseModel):
    prompt: str
    response: str


class PromptingMethod(str, Enum):
    dspy = "dspy"
    fstring = "fstring"


class Experiment(BaseModel):
    test_name: str
    model_name: str
    prompting_method: PromptingMethod
    num_successes: int
    total_task_performance: int
    num_attempts: int
    success_rate: float
    average_task_performance: float
    total_time: int
    all_responses: list[PromptWithResponse]
    failed_responses: list[PromptWithResponse]

    class Config:
        protected_namespaces = ()


class SingleTestResult(BaseModel):
    prompt_with_response: PromptWithResponse
    is_valid: bool
    task_metric: int


# --- Output models for each task ---

class GenerateAnswer(BaseModel):
    answer: str


class RateContext(BaseModel):
    context_score: int


class AssessAnswerability(BaseModel):
    answerable_question: bool


class ParaphraseQuestions(BaseModel):
    paraphrased_questions: List[str]


class RAGAS(BaseModel):
    faithfulness_score: float
    answer_relevance_score: float
    context_relevance_score: float


class GenerateAnswerWithConfidence(BaseModel):
    answer: str
    confidence: int


class GenerateAnswersWithConfidence(BaseModel):
    answers: List[GenerateAnswerWithConfidence]


class ClassifyDocument(BaseModel):
    category: Enum

    class Config:
        arbitrary_types_allowed = True


class ClassifyDocumentWithRationale(BaseModel):
    rationale: str
    category: Enum

    class Config:
        arbitrary_types_allowed = True


def create_enum(enum_name: str, enum_values: List[str]) -> Type[Enum]:
    """Dynamically create an Enum class with given values."""
    return Enum(enum_name, {value: value for value in enum_values})


def _ClassifyDocument(categories: List[str]) -> Type[BaseModel]:
    CategoriesEnum = create_enum("CategoriesEnum", categories)
    return create_model(
        'ClassifyDocument',
        category=(CategoriesEnum, ...)
    )


def _ClassifyDocumentWithRationale(categories: List[str]) -> Type[BaseModel]:
    CategoriesEnum = create_enum("CategoriesEnum", categories)
    return create_model(
        'ClassifyDocumentWithRationale',
        rationale=(str, ...),
        category=(CategoriesEnum, ...)
    )
