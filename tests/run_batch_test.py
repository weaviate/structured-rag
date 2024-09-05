import ast
import json
import os
import requests
import time
import argparse
from pydantic import BaseModel

from helpers import Colors, load_json_from_file
from metrics import is_valid_json_output

from typing import List
from pydantic import BaseModel

from structured_rag.fstring_prompts import get_prompt
from structured_rag.models.models import GenerateAnswer, RateContext, AssessAnswerability, ParaphraseQuestions, GenerateAnswerWithConfidence, GenerateAnswersWithConfidence
from structured_rag.models.models import test_parameters

url = "YOUR_MODAL_URL"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_MODAL_API_KEY", # replace with your Modal API Key
}

def prepare_prompts_for_llama3(prompts: List[str]) -> List[str]:
    prompt_preface = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    Cutting Knowledge Date: December 2023
    Today Date: 23 Jul 2024

    You are a helpful assistant that follows the provided instructions to complete the task in the desired JSON format.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
"""

    prompt_ending = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    # Preface each prompt and append the ending
    return [prompt_preface + prompt + prompt_ending for prompt in prompts]

def run_test(dataset, test_type, with_outlines):
    payload = {
        "with_outlines": with_outlines
    }
    test_params = None

    if test_type == "GenerateAnswer":
        test_params = test_parameters["GenerateAnswer"]
    elif test_type == "RateContext":
        test_params = test_parameters["RateContext"]
    elif test_type == "AssessAnswerability":
        test_params = test_parameters["AssessAnswerability"]
    elif test_type == "ParaphraseQuestions":
        test_params = test_parameters["ParaphraseQuestions"]
    elif test_type == "GenerateAnswerWithConfidence":
        test_params = test_parameters["GenerateAnswerWithConfidence"]
    elif test_type == "GenerateAnswersWithConfidence":
        test_params = test_parameters["GenerateAnswersWithConfidence"]
    else:
        raise ValueError(f"Unsupported test type: {test_type}")

    if with_outlines:
        if test_type == "GenerateAnswer":
            payload["output_model"] = GenerateAnswer.schema()
        elif test_type == "RateContext":
            payload["output_model"] = RateContext.schema()
        elif test_type == "AssessAnswerability":
            payload["output_model"] = AssessAnswerability.schema()
        elif test_type == "ParaphraseQuestions":
            payload["output_model"] = ParaphraseQuestions.schema()
        elif test_type == "GenerateAnswerWithConfidence":
            payload["output_model"] = GenerateAnswerWithConfidence.schema()
        elif test_type == "GenerateAnswersWithConfidence":
            payload["output_model"] = GenerateAnswersWithConfidence.schema()


    prompts = []
    for item in dataset:
        references = {"context": item["abstract"], "question": item["answerable_question"]}
        formatted_prompt = get_prompt(test_type, references, test_params)
        prompts.append(formatted_prompt)

    prompts_for_llama3 = prepare_prompts_for_llama3(prompts)

    payload["prompts"] = prompts_for_llama3

    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload)
    end_time = time.time()
    print(f"Total time taken: {end_time-start_time:.2f} seconds")
    print(f"Average time per task: {(end_time-start_time) / len(prompts):.2f} seconds")

    if response.status_code == 200:
        success_count, total_count = 0, 0
        for idx, output in enumerate(response):
            print(f"Response for prompt: {prompts_for_llama3[idx]}\n")
            print(f"{output}\n\n")
            if is_valid_json_output(output, test_type):
                print(f"{Colors.GREEN}Valid output:\n{output}{Colors.ENDC}")
                success_count += 1
            else:
                print(f"{Colors.RED}Invalid output:\n{output}{Colors.ENDC}")
            total_count += 1

        print(f"Success rate: {success_count / total_count:.2f}")

    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test with or without Outlines")
    parser.add_argument("--with-outlines", action="store_true", help="Run test with Outlines")
    args = parser.parse_args()

    filename = "./data/wiki-answerable-questions.json"
    json_data = load_json_from_file(filename)

    run_test(json_data, "GenerateAnswer", with_outlines=args.with_outlines)
