from structured_rag.core.domain.models import (
    GenerateAnswer, RateContext, AssessAnswerability, ParaphraseQuestions,
    RAGASMetrics, GenerateAnswerWithConfidence, GenerateAnswersWithConfidence,
    ClassifyDocument, ClassifyDocumentWithRationale,
)

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
    "RAGAS": RAGASMetrics,
    "GenerateAnswerWithConfidence": GenerateAnswerWithConfidence,
    "GenerateAnswersWithConfidence": GenerateAnswersWithConfidence,
    "ClassifyDocument": ClassifyDocument,
    "ClassifyDocumentWithRationale": ClassifyDocumentWithRationale
}

ALL_TASKS = [
    "GenerateAnswer",
    "RateContext",
    "AssessAnswerability",
    "ParaphraseQuestions",
    "GenerateAnswerWithConfidence",
    "GenerateAnswersWithConfidence",
    "RAGAS",
]
