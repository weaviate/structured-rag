import ast
import json
import os
import requests
import time
import argparse
from pydantic import BaseModel

from structured_rag.run_test.utils_and_metrics.helpers import Colors, load_json_from_file
from structured_rag.run_test.utils_and_metrics.metrics import is_valid_json_output, assess_answerability_metric

from typing import List
from pydantic import BaseModel

from structured_rag.mock_gfl.fstring_prompts import get_prompt
from structured_rag.models import GenerateAnswer, RateContext, AssessAnswerability, ParaphraseQuestions, RAGAS, GenerateAnswerWithConfidence, GenerateAnswersWithConfidence
from structured_rag.models import test_params

from structured_rag.models import Experiment, PromptWithResponse, PromptingMethod

url = "YOUR_MODAL_URL"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_MODAL_API_KEY", # replace with your Modal API Key
}

def prepare_prompts_for_llama3(prompts: List[str]) -> List[str]:
    prompt_preface = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""

    prompt_ending = """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    # Preface each prompt and append the ending
    return [prompt_preface + prompt + prompt_ending for prompt in prompts]

# currently doing nearly everything in this single function
def run_batch_test(dataset_filepath, test_type, save_dir, with_outlines):
    dataset = load_json_from_file(dataset_filepath)

    # ToD, update to ablate `with_outlines`
    payload = {
        "with_outlines": True
    }

    # Get Pydantic Model to send to vLLM / Outlines
    if with_outlines:
        if test_type == "GenerateAnswer":
            payload["output_model"] = GenerateAnswer.schema()
        elif test_type == "RateContext":
            payload["output_model"] = RateContext.schema()
        elif test_type == "AssessAnswerability":
            payload["output_model"] = AssessAnswerability.schema()
        elif test_type == "ParaphraseQuestions":
            payload["output_model"] = ParaphraseQuestions.schema()
        elif test_type == "RAGAS":
            payload["output_model"] = RAGAS.schema()
        elif test_type == "GenerateAnswerWithConfidence":
            payload["output_model"] = GenerateAnswerWithConfidence.schema()
        elif test_type == "GenerateAnswersWithConfidence":
            payload["output_model"] = GenerateAnswersWithConfidence.schema()

    # ToDo, ablate interfacing the response_format instructions with structured decoding?

    prompts = []
    for item in dataset:
        # just format all references for all potential tasks
        references = {"context": item["context"], 
                      "question": item["question"],
                      "answer": item["answer"]}
        formatted_prompt = get_prompt(test_type, references, test_params[test_type])
        prompts.append(formatted_prompt)

    prompts_for_llama3 = prepare_prompts_for_llama3(prompts)

    payload["prompts"] = prompts_for_llama3

    start_time = time.time()
    # Run all inferences
    response = requests.post(url, headers=headers, json=payload)
    total_time = int(time.time() - start_time)
    print(f"Total time taken: {total_time} seconds")
    print(f"Average time per task: {(total_time) / len(prompts):.2f} seconds")

    batch_experiment = Experiment(
        test_name=args.test,
        model_name="llama3-8b-instruct-Modal",
        prompting_method=PromptingMethod.fstring,
        num_successes=0,
        total_task_performance=0,
        num_attempts=0,
        success_rate=0,
        average_task_performance=0,
        total_time=total_time,
        all_responses=[],
        failed_responses=[]
    )

    if response.status_code == 200:
        response_list = ast.literal_eval(response.text)
        results_dict = {int(result["id"]): result["answer"] for result in response_list}
        sorted_results = dict(sorted(results_dict.items()))
        for id, output in sorted_results.items():
            if is_valid_json_output(output, test_type):
                print(f"{Colors.GREEN}Valid output:\n{output}{Colors.ENDC}")
                batch_experiment.num_successes += 1
                if test_type == "AssessAnswerability":
                    assess_answerability_response = json.loads(output)["answerable_question"]
                    print(f"{Colors.BOLD}Assess Answerability Response: {assess_answerability_response}{Colors.ENDC}")
                    task_metric = assess_answerability_metric(assess_answerability_response, dataset[id]["answerable"])
                    print(f"{Colors.BOLD}Task Metric: {task_metric}{Colors.ENDC}")
                    batch_experiment.total_task_performance += task_metric
            else:
                print(f"{Colors.RED}Invalid output:\n{output}{Colors.ENDC}")
                batch_experiment.failed_responses.append(PromptWithResponse(
                    prompt="placeholder",
                    response=output
                ))
            batch_experiment.num_attempts += 1
            batch_experiment.all_responses.append(PromptWithResponse(
                prompt="placeholder",
                response=output
            ))

        batch_experiment.success_rate = batch_experiment.num_successes / batch_experiment.num_attempts
        batch_experiment.average_task_performance = batch_experiment.total_task_performance / batch_experiment.num_attempts
        print(f"{Colors.GREEN}JSON Success rate: {batch_experiment.success_rate:.2f}{Colors.ENDC}")
        print(f"{Colors.GREEN}Average task performance: {batch_experiment.average_task_performance:.2f}{Colors.ENDC}")
        print(f"{Colors.GREEN}Time to run experiment: {total_time:.2f} seconds{Colors.ENDC}")
        
        # serialize experiment to JSON
        os.makedirs(args.save_dir, exist_ok=True)
        # ToDo, ablate `args.model_name`

        # Fix this save path
        batch_result_file = os.path.join(args.save_dir, f"{args.test}-BATCH-llama3-8b-instruct-Modal.json")

        with open(batch_result_file, "w") as f:
            json.dump(batch_experiment.dict(), f, indent=2)
        
        print(f"\nResults saved in {batch_result_file}.")

    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test with or without Outlines")
    # ToDo, update to ablate `with_outlines`
    #parser.add_argument("--with-outlines", action="store_true", help="Run test with Outlines")
    parser.add_argument("--test", type=str, default="AssessAnswerability", help="Test to run")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    dataset_filepath = "../../../data/WikiQuestions.json"
    run_batch_test(dataset_filepath, args.test, args.save_dir, with_outlines=True)