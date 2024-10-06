from pydantic import BaseModel, create_model
from enum import Enum
from typing import Type, List

class PromptWithResponse(BaseModel):
    prompt: str
    response: str

class PromptingMethod(str, Enum):
    dspy = "dspy"
    fstring = "fstring"

# ToDo rename to `JSON_success_rate`
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
    # Dynamically create the Enum for categories
    CategoriesEnum = create_enum("CategoriesEnum", categories)
    
    # Dynamically create the Pydantic model with the Enum field
    ClassifyDocument = create_model(
        'ClassifyDocument',
        category=(CategoriesEnum, ...)
    )
    
    return ClassifyDocument

def _ClassifyDocumentWithRationale(categories: List[str]) -> Type[BaseModel]:
    # Dynamically create the Enum for categories
    CategoriesEnum = create_enum("CategoriesEnum", categories)
    
    # Dynamically create the Pydantic model with the Enum field and rationale
    ClassifyDocumentWithRationale = create_model(
        'ClassifyDocumentWithRationale',
        rationale=(str, ...),
        category=(CategoriesEnum, ...)
    )
    
    return ClassifyDocumentWithRationale



# ToDo, get `test_params` from here instead of hardcoded in `run_test.py`
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
    "RAGAS": {
        "task_instructions": "Assess the faithfulness, answer relevance, and context relevance given a question, context, and answer.",
        "response_format": '{"faithfulness_score": "float (0-5)", "answer_relevance_score": "float (0-5)", "context_relevance_score": "float (0-5)"}'
    },
    "GenerateAnswerWithConfidence": {
        "task_instructions": "Generate an answer with a confidence score.",
        "response_format": '{"Answer": "string", "Confidence": "int (0-5)"}'
    },
    "GenerateAnswersWithConfidence": {
        "task_instructions": "Generate multiple answers with confidence scores.",
        "response_format": '[{"Answer": "string", "Confidence": "int (0-5)"}, ...]'
    },
    "ClassifyDocument": {
        "task_instructions": "Classify the document into one of the provided classes.",
        "response_format": '{"classification": "Enum"}'
    },
    "ClassifyDocumentWithRationale": {
        "task_instructions": "Classify the document into one of the provided classes and provide a rationale explaining why the document belongs in this class.",
        "response_format": '{"rationale": "string", "classification": "Enum"}'
    }
}

test_to_output_model = {
    "GenerateAnswer": GenerateAnswer,
    "RateContext": RateContext,
    "AssessAnswerability": AssessAnswerability,
    "ParaphraseQuestions": ParaphraseQuestions,
    "RAGAS": RAGAS,
    "GenerateAnswerWithConfidence": GenerateAnswerWithConfidence,
    "GenerateAnswersWithConfidence": GenerateAnswersWithConfidence,
    "ClassifyDocument": ClassifyDocument,
    "ClassifyDocumentWithRationale": ClassifyDocumentWithRationale
}