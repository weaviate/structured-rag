import dspy

from structured_rag.run_test.utils_and_metrics.helpers import load_experiments
from structured_rag.run_test.utils_and_metrics.helpers import Colors

class ErrorAnalyzer(dspy.Signature):
    """An AI system has been tasked with following a particular response format in its output. 
    Please review the example and output why the response failed to follow the response format."""

    system_output = dspy.InputField(description="The output from the AI system.")
    why_it_failed = dspy.OutputField(description="Why the AI system failed to follow the response format.")

class SummarizeErrors(dspy.Signature):
    """An AI System has been tasked with following a particular response format in its output.
    Another AI system has reviewed the AI system's output and provided an error analysis for each error.
    Please summarize the provided list of error analyses into a single error analysis."""

    errors = dspy.InputField(description="The list of errors.")
    error_analysis_report = dspy.OutputField(description="The summary of the errors.")

error_analyzer = dspy.Predict(ErrorAnalyzer)

# loop through failed_responses

# ToDo reorganize results to move gemini results
experiments = load_experiments("../results/")

failed_responses = experiments["failed_responses"]

error_analyses = []
for failed_response in failed_responses:
    error_analysis = error_analyzer.predict(failed_response).why_it_failed
    error_analyses.append(error_analysis)
    print(f"{Colors.GREEN}Error analysis: {error_analysis}{Colors.ENDC}")

summary = SummarizeErrors.predict(error_analyses).error_analysis_report

print(summary)
