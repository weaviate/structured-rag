import json
import os
import argparse
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

def run_test(args):
    filename = "../../../data/WikiQuestions.json"
    json_data = load_json_from_file(filename)

    print(f"{Colors.BOLD}Number of samples in the dataset: {len(json_data)}{Colors.ENDC}")

    if args.test not in test_params:
        raise ValueError(f"Unsupported test: {args.test}")

    test_to_run = test_params[args.test]
    output_model = test_to_output_model[args.test]

    # Define program configurations
    program_configs = [
        # DSPy Programs
        {
            'name': 'dspy_NO_OPRO_JSON',
            'type': 'dspy',
            'params': {
                'use_OPRO_JSON': False,
                'test_params': test_to_run,
                'model_name': args.model_name,
                'model_provider': args.model_provider,
                'api_key': args.api_key
            }
        },
        {
            'name': 'dspy_WITH_OPRO_JSON',
            'type': 'dspy',
            'params': {
                'use_OPRO_JSON': True,
                'test_params': test_to_run,
                'model_name': args.model_name,
                'model_provider': args.model_provider,
                'api_key': args.api_key
            }
        },
        # f-string Programs
        {
            'name': 'fstring_without_structured_outputs',
            'type': 'fstring',
            'params': {
                'structured_outputs': False,
                'test_params': test_to_run,
                'model_name': args.model_name,
                'model_provider': args.model_provider,
                'api_key': args.api_key
            }
        },
        {
            'name': 'fstring_with_structured_outputs',
            'type': 'fstring',
            'params': {
                'structured_outputs': True,
                'test_params': test_to_run,
                'model_name': args.model_name,
                'model_provider': args.model_provider,
                'api_key': args.api_key
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
            test_name=args.test,
            model_name=args.model_name,
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
                test_type=args.test,
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
        os.makedirs("../results/" + args.save_dir, exist_ok=True)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        result_file = os.path.join("../results/" + args.save_dir, f"{args.test}-{args.model_name}-{program_config['name']}-{current_date}.json")

        with open(result_file, "w") as f:
            json.dump(experiment.dict(), f, indent=2)

        print(f"\nResults saved in {result_file}")

        total_inference_count += inference_count

    # Print total number of inferences run
    print(f"{Colors.BOLD}Total number of inferences run: {total_inference_count}{Colors.ENDC}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM testing with different models.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--model_provider", type=str, required=True, choices=["ollama", "google", "openai", "anthropic"], help="Provider of the model")
    parser.add_argument("--api_key", type=str, required=False, help="API key for the model provider (if required)")
    parser.add_argument("--test", type=str, required=True, choices=[
        "GenerateAnswer","RateContext","AssessAnswerability",
        "ParaphraseQuestions","RAGAS","RateMultipleAspects",
        "GenerateAnswerWithConfidence","GenerateAnswersWithConfidence"
    ], help="Type of test to run")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the results")

    args = parser.parse_args()
    run_test(args)