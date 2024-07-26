from typing import Dict

def get_prompt(test: str, references: Dict[str, str]) -> str:
    test_params = {
            "GenerateAnswer": {
                "task_instructions": "Assess the context and answer the question. If the context does not contain sufficient information to answer the question, respond with \"NOT ENOUGH CONTEXT\".",
                "response_format": '{"answer": string}'
            },
            "RateContext": {
                "task_instructions": "Assess how well the context helps answer the question.",
                "response_format": '{"context_score": int (0-5)}'
            },
            "AssessAnswerability": {
                "task_instructions": "Determine if the question is answerable based on the context.",
                "response_format": '{"answerable_question": bool}'
            },
            "ParaphraseQuestions": {
                "task_instructions": "Generate 3 paraphrased versions of the given question.",
                "response_format": '{"paraphrased_questions": List[string]}'
            },
            "GenerateAnswerWithConfidence": {
                "task_instructions": "Generate an answer with a confidence score.",
                "response_format": '{"answer": string, "confidence": int (0-5)}'
            },
            "GenerateAnswersWithConfidence": {
                "task_instructions": "Generate multiple answers with confidence scores.",
                "response_format": '[{"answer": string, "confidence": int (0-5)}, ...]'
            }
    }

    if test not in test_params:
        raise ValueError(f"Unsupported test: {test}")

    params = test_params[test]
    references_str = ' | '.join(f"{k}: {v}" for k, v in references.items())

    return f"""
    Task Instructions: {params['instructions']}
    References: {references_str}
    Output the result as a JSON string with the following format: {params['response_format']}
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """