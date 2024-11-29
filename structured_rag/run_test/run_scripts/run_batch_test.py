import ast
import json
import os
import requests
import time
from pydantic import BaseModel

from structured_rag.run_test.utils_and_metrics.helpers import Colors, load_json_from_file
from structured_rag.run_test.utils_and_metrics.metrics import is_valid_json_output, assess_answerability_metric, classification_metric
from structured_rag.run_test.utils_and_metrics.metrics import GenerateAnswerTaskMetric

from typing import List
from pydantic import BaseModel

from structured_rag.mock_gfl.fstring_prompts import get_prompt
from structured_rag.models import GenerateAnswer, RateContext, AssessAnswerability, ParaphraseQuestions, RAGAS, GenerateAnswerWithConfidence, GenerateAnswersWithConfidence, ClassifyDocument
from structured_rag.models import test_params
from structured_rag.models import create_enum, _ClassifyDocument, _ClassifyDocumentWithRationale

from structured_rag.models import Experiment, PromptWithResponse, PromptingMethod

# Configuration variables
url = "YOUR_MODAL_URL"
openai_api_key = "sk-foobar"
test_type = "AssessAnswerability"
save_dir = "results"
dataset_filepath = "SuperBEIR"

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
    # fix this with a CLI argument `dataset`
    # Leaving the hardcoded filepath
    if dataset_filepath == "../../../data/WikiQuestions.json":
        dataset = load_json_from_file(dataset_filepath)
    else:
        #dataset = load_superbeir()
        dataset = load_json_from_file("../../../data/SuperBEIR/SuperBEIR-small-balanced.json")[:340]

        # Load SuperBEIR categories and their descriptions
        with open('../../../data/SuperBEIR/SuperBEIR-categories-with-rationales.json', 'r') as file:
            data = json.load(file)

        # Create a list of dictionaries with category name and description
        categories = [{category: info['category_description']} for category, info in data.items()]

        formatted_categories = ""
        for category_dict in categories:
            for category_name, category_description in category_dict.items():
                formatted_categories += f"{category_name}: {category_description}\n"

        # Remove the trailing newline
        formatted_categories = formatted_categories.rstrip()
        categories = list(data.keys())


    # ToD, update to ablate `with_outlines`
    payload = {
        "with_outlines": True
    }

    # Get Pydantic Model to send to vLLM / Outlines
    if with_outlines:
        if test_type == "GenerateAnswer":
            payload["output_model"] = GenerateAnswer.schema()
            generate_answer_task_metric = GenerateAnswerTaskMetric(api_key=openai_api_key)
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
        elif test_type == "ClassifyDocument":
            ClassifyDocumentModel = _ClassifyDocument(categories)
            payload["output_model"] = ClassifyDocumentModel.schema()
        elif test_type == "ClassifyDocumentWithRationale":
            ClassifyDocumentWithRationale = _ClassifyDocumentWithRationale(categories)
            payload["output_model"] = ClassifyDocumentWithRationale.schema()

    # ToDo, ablate interfacing the response_format instructions with structured decoding?

    prompts = []
    for item in dataset:
        # ToDo, fix this
        if test_type == "ClassifyDocument" or test_type == "ClassifyDocumentWithRationale":
            references = {"document": item["document"],
                          "label": item["label"],
                          "classes_with_descriptions": formatted_categories}
        else:
            references = {"context": item["context"], 
                          "question": item["question"],
                          "answer": item["answer"]}
        formatted_prompt = get_prompt(test_type, references, test_params[test_type])
        prompts.append(formatted_prompt)

    prompts_for_llama3 = prepare_prompts_for_llama3(prompts)

    payload["prompts"] = prompts_for_llama3

    start_time = time.time()
    # Run all inferences
    response = requests.post(url, headers=headers, json=payload, timeout=3000)  # Increased timeout to 5 minutes
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time} seconds")
    print(f"Average time per task: {(total_time) / len(prompts):.2f} seconds")

    # check the `int` valued total_time, I don't think that's right
    batch_experiment = Experiment(
        test_name=test_type,
        model_name="llama3.2-3B-Instruct-Modal",
        prompting_method=PromptingMethod.fstring,
        num_successes=0,
        total_task_performance=0,
        num_attempts=0,
        success_rate=0,
        average_task_performance=0,
        total_time=int(total_time),
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
                if test_type == "GenerateAnswer":
                    answer_response = json.loads(output)["answer"]
                    print(f"{Colors.BOLD}Answer Response: {answer_response}{Colors.ENDC}")
                    print(f"{Colors.RED}Ground Truth: {dataset[id]['answer']}{Colors.ENDC}")
                    task_metric, rationale = generate_answer_task_metric.assess_answer_metric(context=dataset[id]["context"], 
                                                                                   question=dataset[id]["question"], 
                                                                                   system_answer=answer_response, 
                                                                                   ground_truth=dataset[id]["answer"])
                    print(f"{Colors.BOLD}Task Metric: {task_metric}{Colors.ENDC}\n")
                    print(f"{Colors.CYAN}Rationale: {rationale}{Colors.ENDC}")
                    batch_experiment.total_task_performance += task_metric
                if test_type == "ClassifyDocument":
                    classification_response = json.loads(output)["category"] # extend to return classification and rationale
                    print(f"{Colors.BOLD}Classification Response: {classification_response}{Colors.ENDC}")
                    ground_truth = dataset[id]["label"]
                    print(f"{Colors.CYAN}Ground Truth: {ground_truth}{Colors.ENDC}")
                    task_metric = classification_metric(classification_response, ground_truth)
                    print(f"{Colors.BOLD}Task Metric: {task_metric}{Colors.ENDC}")
                    batch_experiment.total_task_performance += task_metric
                if test_type == "ClassifyDocumentWithRationale":
                    # ToDo, extend to do something with the rationale as well
                    classification_response = json.loads(output)["category"] # extend to return classification and rationale
                    print(f"{Colors.BOLD}Classification Response: {classification_response}{Colors.ENDC}")
                    ground_truth = dataset[id]["label"]
                    print(f"{Colors.CYAN}Ground Truth: {ground_truth}{Colors.ENDC}")
                    task_metric = classification_metric(classification_response, ground_truth)
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
        print(f"{Colors.GREEN}Time to run experiment: {total_time} seconds{Colors.ENDC}")
        
        # serialize experiment to JSON
        os.makedirs(save_dir, exist_ok=True)

        # Fix this save path
        batch_result_file = os.path.join(save_dir, f"{test_type}-Modal-vLLM.json")

        with open(batch_result_file, "w") as f:
            json.dump(batch_experiment.dict(), f, indent=2)
        
        print(f"\nResults saved in {batch_result_file}.")

    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    run_batch_test(dataset_filepath, test_type, save_dir, with_outlines=True)