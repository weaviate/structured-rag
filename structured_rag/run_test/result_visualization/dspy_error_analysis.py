import dspy

from structured_rag.run_test.utils_and_metrics.helpers import load_experiments
from structured_rag.run_test.utils_and_metrics.helpers import Colors

import openai
openai.api_key = "sk-foobar"

gpt4 = dspy.OpenAI(model="gpt-4o-mini")

dspy.settings.configure(lm=gpt4)

class ErrorAnalyzer(dspy.Signature):
    """An AI system has been tasked with following a particular response format in its output. 
    Please review the example and output why the response failed to follow the response format."""

    system_output = dspy.InputField(description="The output from the AI system.")
    why_it_failed = dspy.OutputField(description="Why the AI system failed to follow the response format.")

class SummarizeErrors(dspy.Signature):
    """An AI System has been tasked with following a particular response format in its output.
    Another AI system has reviewed the AI system's output and provided an error analysis for each error.
    Please summarize the provided list of error analyses into a single error analysis."""

    error_analyses = dspy.InputField(description="The list of errors.")
    error_analysis_report = dspy.OutputField(description="The summary of the errors.")

error_analyzer = dspy.Predict(ErrorAnalyzer)

# loop through failed_responses

# ToDo reorganize results to move gemini results
experiments = load_experiments("../results/Gemini-1.5-Pro-9-11-24")

print(experiments.info())

# Need to add the task_instruction and response_format to these Results in order to parse it here

failed_responses_per_experiments = experiments["failed_responses"].tolist()
test_names = experiments["test_name"].tolist()

for idx, failed_responses in enumerate(failed_responses_per_experiments):
    print(f"{Colors.GREEN}Analyzing Failures for Experiment: {test_names[idx]}\n{Colors.ENDC}")
    error_analyses = []
    for idx, failed_response in enumerate(failed_responses):
        print(f"{Colors.BOLD}Analyzing Failure {idx}: {failed_response.response}\n{Colors.ENDC}")
        error_analysis = error_analyzer(system_output=failed_response.response).why_it_failed
        error_analyses.append(error_analysis)
        print(f"{Colors.GREEN}Error analysis: {error_analysis}\n{Colors.ENDC}")

    error_analyses = "\n".join([f"[{i+1}] {item}" for i, item in enumerate(error_analyses)])
    summary = SummarizeErrors.predict(error_analyses=error_analyses).error_analysis_report

    print(f"{Colors.BOLD}Summary of Errors:{Colors.ENDC}\n{summary}")
