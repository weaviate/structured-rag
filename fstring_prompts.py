from typing import Dict

def get_prompt(test: str, references: Dict[str, str]) -> str:
    test_params = {
        "GenerateAnswer": {
            "instructions": "Assess the context and answer the question. If the context does not contain sufficient information to answer the question, respond with \"NOT ENOUGH CONTEXT\".",
            "response_format": '{"answer": "string"}'
        },
        "RateContext": {
            "instructions": "Assess how well the context helps answer the question. Rate the relevance of the context to the question on a scale of 0 to 5.",
            "response_format": '{"context_score": "int (0-5)"}'
        },
        "AssessAnswerability": {
            "instructions": "Determine if the question is answerable based on the context.",
            "response_format": '{"answerable_question": "bool"}'
        },
        "ParaphraseQuestions": {
            "instructions": "Generate 3 paraphrased versions of the given question.",
            "response_format": '{"paraphrased_questions": ["string", "string", "string"]}'
        },
        "GenerateAnswerWithConfidence": {
            "instructions": "Generate an answer to the question based on the context, and provide a confidence score (0-5).",
            "response_format": '{"Answer": "string", "Confidence": "int (0-5)"}'
        },
        "GenerateAnswersWithConfidence": {
            "instructions": "Generate 3 possible answers to the question based on the context, each with a confidence score (0-5).",
            "response_format": '[{"Answer": "string", "Confidence": "int (0-5)"}, ...]'
        }
    }

    if test not in test_params:
        raise ValueError(f"Unsupported test: {test}")

    params = test_params[test]
    references_str = ' | '.join(f"{k}: {v}" for k, v in references.items())

    return f"""
    Instructions: {params['instructions']}
    References: {references_str}
    Output the result as a JSON string with the following format: {params['response_format']}
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """