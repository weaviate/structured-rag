def get_generate_answer_prompt(context: str, question: str) -> str:
    return f"""
    Assess the context: {context} and answer the question {question}.
    Output the answer as a JSON string with the key "answer".
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """

def get_rate_context_prompt(context: str, question: str) -> str:
    return f"""
    Assess how well the context helps answer the question.
    Context: {context}
    Question: {question}
    Rate the relevance of the context to the question on a scale of 0 to 5.
    Output the rating as a JSON string with the key "context_score".
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """

def get_assess_answerability_prompt(context: str, question: str) -> str:
    return f"""
    Determine if the question is answerable based on the context.
    Context: {context}
    Question: {question}
    Output the result as a JSON string with the key "answerable_question" (boolean value).
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """

def get_paraphrase_questions_prompt(question: str) -> str:
    return f"""
    Generate 3 paraphrased versions of the given question: {question}
    Output the result as a JSON string with the key "paraphrased_questions" (list of strings).
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """

def get_generate_answer_with_confidence_prompt(context: str, question: str) -> str:
    return f"""
    Generate an answer to the question based on the context, and provide a confidence score (0-5).
    Context: {context}
    Question: {question}
    Output the result as a JSON string with keys "Answer" and "Confidence".
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """

def get_generate_answers_with_confidence_prompt(context: str, question: str) -> str:
    return f"""
    Generate 3 possible answers to the question based on the context, each with a confidence score (0-5).
    Context: {context}
    Question: {question}
    Output the result as a JSON string with a list of objects containing "Answer" and "Confidence" keys.
    IMPORTANT!! Do not start the JSON with ```json or end it with ```.
    """