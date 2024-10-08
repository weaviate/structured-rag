import ast
import json
import requests
import time
import argparse
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence_rating: float

url = "YOUR_MODAL_URL"

headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer YOUR_MODAL_API_KEY", # replace with your Modal API Key
}

prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?",
    "What is the capital of the United Kingdom?",
    "What is the capital of Ireland?",
    "What is the capital of Sweden?",
    "What is the capital of Norway?",
    "What is the capital of Finland?",
    "What is the capital of Denmark?",
    "What is the capital of Poland?",
    "What is the capital of Austria?",
    "What is the capital of Switzerland?",
    "What is the capital of Greece?",
    "What is the capital of Turkey?",
    "What is the capital of Russia?",
    "What is the capital of Ukraine?",
    "What is the capital of Romania?",
    "What is the capital of Bulgaria?"
]

prompt_preface = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
"""

prompt_ending = """<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

# Loops through `prompts` and prefaces each with the prompt_preface
prefaced_prompts = [prompt_preface + prompt for prompt in prompts]
# Loops through `prefaced_prompts` and appends the prompt_ending to each
prefaced_prompts_with_ending = [prompt + prompt_ending for prompt in prefaced_prompts]

'''
prompts_with_ids = []

for idx, prompt in enumerate(prefaced_prompts_with_ending):
    prompts_with_ids.append(PromptWithID(prompt=prompt, id=idx))
'''
    
def run_test(with_outlines):
    payload = {
        "prompts": prefaced_prompts_with_ending,
        "with_outlines": with_outlines,
    }
    if with_outlines:
        payload["output_model"] = Answer.schema()

    start_time = time.time()
    response = requests.post(url, headers=headers, json=payload)
    

    end_time = time.time()

    if response.status_code == 200:
        response_list = ast.literal_eval(response.text)
        print(f"\nResults {'with' if with_outlines else 'without'} Outlines:")
        results_dict = {int(result["id"]): result["answer"] for result in response_list}
        sorted_results = dict(sorted(results_dict.items()))
        
        for id, answer in sorted_results.items():
            print(f"Prompt {id + 1}: {prompts[id]}")
            print("=" * 50)
            print(f"ID: {id+1}")
            print(f"Answer: {answer}")
            print("=" * 50)
        
        total_time = end_time - start_time
        num_tasks = len(response_list)
        print(f"Number of answers: {num_tasks}")
        print(f"Total time taken: {total_time:.2f} seconds")
        print(f"Average time per task: {total_time / num_tasks:.2f} seconds")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run test with or without Outlines")
    parser.add_argument("--with-outlines", action="store_true", help="Run test with Outlines")
    args = parser.parse_args()

    run_test(with_outlines=args.with_outlines)

# To run this script:
# Without Outlines: python query_test.py
# With Outlines: python query_test.py --with-outlines
