import json
import os
import datetime
import time

from typing import Optional
from pydantic import BaseModel

from structured_rag.mock_gfl.dspy_program import dspy_Program
from structured_rag.mock_gfl.fstring_program import fstring_Program

from structured_rag.run_test.utils_and_metrics.helpers import Colors, load_json_from_file
from structured_rag.run_test.utils_and_metrics.metrics import is_valid_json_output, assess_answerability_metric

from structured_rag.models import Experiment, PromptWithResponse, PromptingMethod, SingleTestResult
from structured_rag.models import test_params, test_to_output_model

# Configuration variables
MODEL_NAME = "gpt-4o"
MODEL_PROVIDER = "openai" # one of: "ollama", "google", "openai", "anthropic"
API_KEY = "your-api-key-here"
TEST_TYPE = "AssessAnswerability" # one of: "GenerateAnswer", "RateContext", "AssessAnswerability", "ParaphraseQuestions", "RAGAS", "RateMultipleAspects", "GenerateAnswerWithConfidence", "GenerateAnswersWithConfidence"
SAVE_DIR = "test_results"

def run_single_test(output_model: Optional[BaseModel],
                    program, test_type, title, context, question, answer, task_specific_ground_truth) -> SingleTestResult:
    try:
        if test_type == "ParaphraseQuestions":
            output = program.forward(output_model, test_type, question=question)
        elif test_type == "RAGAS":
            output = program.forward(output_model, test_type, context, question, answer)
        else:
            output = program.forward(output_model, test_type, context, question)

        print(f"{Colors.CYAN}{program.__class__.__name__} Output: {output}{Colors.ENDC}\n")

        task_metric = 0

        parsed_output, is_valid = is_valid_json_output(output, test_type)

        if is_valid:
            print(f"{Colors.GREEN}Valid output for {test_type}{Colors.ENDC}")
            is_valid = True
            if test_type == "AssessAnswerability":
                answerable_question_response = parsed_output # not necessary, but lazy
                # print(f"{Colors.BOLD}Assess Answerability Response: {answerable_question_response}{Colors.ENDC}")
                # print(f"{Colors.CYAN}Ground truth answerability: {task_specific_ground_truth}{Colors.ENDC}\n")
                # print(f"Predicted type {type(answerable_question_response)}\n")
                # print(f"Ground truth type {type(task_specific_ground_truth)}\n")
                task_metric = assess_answerability_metric(answerable_question_response, task_specific_ground_truth)
                print(f"{Colors.BOLD}Task Metric: {task_metric}{Colors.ENDC}")
        else:
            print(f"{Colors.RED}Invalid output for {test_type}{Colors.ENDC}")

        return SingleTestResult(prompt_with_response=PromptWithResponse(prompt=f"Title: {title}\nContext: {context}\nQuestion: {question}", response=output), is_valid=is_valid, task_metric=task_metric)

    except Exception as e:
        print(f"{Colors.YELLOW}Error occurred: {str(e)}{Colors.ENDC}")
        print(f"{Colors.RED}Skipping this test due to error.{Colors.ENDC}")
        return SingleTestResult(prompt_with_response=PromptWithResponse(prompt=f"Title: {title}\nContext: {context}\nQuestion: {question}", response="Error"), is_valid=False, task_metric=0)

def run_test():
    filename = "../../../data/WikiQuestions.json"
    json_data = load_json_from_file(filename)

    print(f"{Colors.BOLD}Number of samples in the dataset: {len(json_data)}{Colors.ENDC}")

    if TEST_TYPE not in test_params:
        raise ValueError(f"Unsupported test: {TEST_TYPE}")

    test_to_run = test_params[TEST_TYPE]
    output_model = test_to_output_model[TEST_TYPE]

    # Define program configurations
    program_configs = [
        # DSPy Programs
        {
            'name': 'dspy_NO_OPRO_JSON',
            'type': 'dspy',
            'params': {
                'use_OPRO_JSON': False,
                'test_params': test_to_run,
                'model_name': MODEL_NAME,
                'model_provider': MODEL_PROVIDER,
                'api_key': API_KEY
            }
        },
        {
            'name': 'dspy_WITH_OPRO_JSON',
            'type': 'dspy',
            'params': {
                'use_OPRO_JSON': True,
                'test_params': test_to_run,
                'model_name': MODEL_NAME,
                'model_provider': MODEL_PROVIDER,
                'api_key': API_KEY
            }
        },
        # f-string Programs
        {
            'name': 'fstring_without_structured_outputs',
            'type': 'fstring',
            'params': {
                'structured_outputs': False,
                'test_params': test_to_run,
                'model_name': MODEL_NAME,
                'model_provider': MODEL_PROVIDER,
                'api_key': API_KEY
            }
        },
        {
            'name': 'fstring_with_structured_outputs',
            'type': 'fstring',
            'params': {
                'structured_outputs': True,
                'test_params': test_to_run,
                'model_name': MODEL_NAME,
                'model_provider': MODEL_PROVIDER,
                'api_key': API_KEY
            }
        }
    ]

    total_inference_count = 0  # Total inferences across all programs

    # For each program configuration
    for program_config in program_configs:
        print(f"\n{Colors.BOLD}Running tests for program: {program_config['name']}{Colors.ENDC}")
        # Initialize the program
        if program_config['type'] == 'dspy':
            program = dspy_Program(**program_config['params'])
            prompting_method = PromptingMethod.dspy
        elif program_config['type'] == 'fstring':
            program = fstring_Program(**program_config['params'])
            prompting_method = PromptingMethod.fstring
        else:
            raise ValueError(f"Unknown program type: {program_config['type']}")

        # Initialize the Experiment
        experiment = Experiment(
            test_name=TEST_TYPE,
            model_name=MODEL_NAME,
            prompting_method=prompting_method,
            num_successes=0,
            total_task_performance=0,
            num_attempts=0,
            success_rate=0,
            average_task_performance=0,
            total_time=0,
            all_responses=[],
            failed_responses=[]
        )

        total_start_time = time.time()
        inference_count = 0  # Inferences for this program

        # Loop over dataset entries
        for entry in json_data:
            title = entry.get('title', '')
            context = entry.get('context', '')
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            answerable = entry.get('answerable', '')

            print(f"{Colors.UNDERLINE}Title: {title}{Colors.ENDC}")
            print(f"{Colors.UNDERLINE}Question: {question}{Colors.ENDC}\n")

            single_test_result = run_single_test(
                output_model=output_model,
                program=program,
                test_type=TEST_TYPE,
                title=title,
                context=context,
                question=question,
                answer=answer,
                task_specific_ground_truth=answerable
            )
            inference_count += 1

            # Record the result
            if single_test_result:
                experiment.all_responses.append(single_test_result.prompt_with_response)
                experiment.num_attempts += 1
                if single_test_result.is_valid:
                    experiment.num_successes += 1
                    experiment.total_task_performance += single_test_result.task_metric
                else:
                    experiment.failed_responses.append(single_test_result.prompt_with_response)

            print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")

        total_time = time.time() - total_start_time
        experiment.total_time = int(total_time)

        # Calculate success rate and average task performance
        if experiment.num_attempts > 0:
            experiment.success_rate = experiment.num_successes / experiment.num_attempts
            experiment.average_task_performance = experiment.total_task_performance / experiment.num_attempts
        else:
            experiment.success_rate = 0
            experiment.average_task_performance = 0

        # Print final scores
        print(f"{Colors.HEADER}Final Scores for {program_config['name']}:{Colors.ENDC}")
        print(f"{Colors.BOLD}JSON Success Rate: {Colors.GREEN}{experiment.num_successes}/{experiment.num_attempts} ({experiment.success_rate:.2%}){Colors.ENDC}")
        print(f"{Colors.BOLD}Average Task Performance: {Colors.GREEN}{experiment.average_task_performance:.2f}{Colors.ENDC}")

        # Save results to JSON file
        os.makedirs("../results/" + SAVE_DIR, exist_ok=True)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        result_file = os.path.join("../results/" + SAVE_DIR, f"{TEST_TYPE}-{MODEL_NAME}-{program_config['name']}-{current_date}.json")

        with open(result_file, "w") as f:
            json.dump(experiment.dict(), f, indent=2)

        print(f"\nResults saved in {result_file}")

        # Append results to experiment log
        with open("experiment-log.md", "a") as f:
            f.write(f"| {MODEL_NAME} | {experiment.success_rate:.2%} | {TEST_TYPE} | {current_date} |\n")

        total_inference_count += inference_count

    # Print total number of inferences run
    print(f"{Colors.BOLD}Total number of inferences run: {total_inference_count}{Colors.ENDC}")

if __name__ == "__main__":
    run_test()