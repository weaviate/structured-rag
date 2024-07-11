import dspy
from typing import List

class AnswerQuestion(dspy.Signature):
    """Assess the context and answer the question.
    VERY IMPORTANT!! Respond with a JSON template with the output key "answer" with a string value. Do not start your response with ```json or end it with ```.
    """
    context = dspy.InputField(desc="(Input Field)")
    question = dspy.InputField(desc="(Input Field)")
    answer = dspy.OutputField(desc="(Output Field)")

class RateContext(dspy.Signature):
    """Assess how well the context helps answer the question.
    VERY IMPORTANT!! Respond with a JSON template with the output key "context_score" with an int value. Do not start your response with ```json or end it with ```.
    """
    context = dspy.InputField(desc="(Input Field)")
    question = dspy.InputField(desc="(Input Field)")
    context_score = dspy.OutputField(desc="(Output Field)")

class AssessAnswerability(dspy.Signature):
    """Determine if the question is answerable based on the context.
    VERY IMPORTANT!! Respond with a JSON template with the output key "answerable_question" with a bool value. Do not start your response with ```json or end it with ```.
    """
    context = dspy.InputField(desc="(Input Field)")
    question = dspy.InputField(desc="(Input Field)")
    answerable_question = dspy.OutputField(desc="(Output Field)")

class ParaphraseQuestions(dspy.Signature):
    """Generate 3 paraphrased versions of the given question.
    VERY IMPORTANT!! Respond with a JSON template with the output key "paraphrased_questions" wiht a List[str] value. Do not start your response with ```json or end it with ```.
    """
    question = dspy.InputField(desc="(Input Field)")
    paraphrased_questions = dspy.OutputField(desc="(Output Field)")

class GenerateAnswerWithConfidence(dspy.Signature):
    """Generate an answer with a confidence score.
    VERY IMPORTANT!! Respond with a JSON template with the keys "Answer" and "Confidence", "Answer" has a str value and "Confidence" has an int value on a scale of 0 to 5. Do not start your response with ```json or end it with ```.
    """
    context = dspy.InputField(desc="(Input Field)")
    question = dspy.InputField(desc="(Input Field)")
    answer_with_confidence = dspy.OutputField(desc="(Output Field)")

class GenerateAnswersWithConfidence(dspy.Signature):
    """Generate multiple answers with confidence scores.
    VERY IMPORTANT!! Respond with a JSON template with a list of objects containing "Answer" and "Confidence" keys, "Answer" has a str value and "Confidence" has an int value on a scale of 0 to 5. Do not start your response with ```json or end it with ```.
    """
    context = dspy.InputField(desc="(Input Field)")
    question = dspy.InputField(desc="(Input Field)")
    answers_with_confidence = dspy.OutputField(desc="(Output Field)")