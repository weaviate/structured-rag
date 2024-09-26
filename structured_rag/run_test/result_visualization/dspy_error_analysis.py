import dspy

class ErrorAnalyzer(dspy.Signature):
    """An AI system has been tasked with following a particular response format in its output. Please review the example and output why the response failed to follow the response format."""

    system_output = dspy.InputField(description="The output from the AI system.")
    why_it_failed = dspy.OutputField(description="Why the AI system failed to follow the response format.")

error_analyzer = dspy.Predict(ErrorAnalyzer)

# loop through failed_responses