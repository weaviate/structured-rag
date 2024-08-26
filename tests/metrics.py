import json

def is_valid_json_output(output, test_type):
    try:
        parsed = json.loads(output)
        if test_type == "GenerateAnswer":
            return isinstance(parsed.get("answer"), str)
        elif test_type == "RateContext":
            score = parsed.get("context_score")
            return isinstance(score, int) and 0 <= score <= 5
        elif test_type == "AssessAnswerability":
            return isinstance(parsed.get("answerable_question"), bool)
        elif test_type == "ParaphraseQuestions":
            questions = parsed.get("paraphrased_questions")
            return isinstance(questions, list) and all(isinstance(q, str) for q in questions)
        elif test_type == "GenerateAnswerWithConfidence":
            return isinstance(parsed.get("Answer"), str) and isinstance(parsed.get("Confidence"), int) and 0 <= parsed["Confidence"] <= 5
        elif test_type == "GenerateAnswersWithConfidence":
            answers = parsed
            return isinstance(answers, list) and all(isinstance(a.get("Answer"), str) and isinstance(a.get("Confidence"), int) and 0 <= a["Confidence"] <= 5 for a in answers)
        else:
            return False
    except json.JSONDecodeError:
        return False