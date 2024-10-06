import json
import os
import argparse
import datetime
import time

from typing import Optional
from pydantic import BaseModel

from structured_rag.mock_gfl.dspy_program import dspy_Program
from structured_rag.mock_gfl.fstring_program import fstring_Program
import dspy

from structured_rag.run_test.utils_and_metrics.helpers import Colors, load_json_from_file
from structured_rag.run_test.utils_and_metrics.metrics import is_valid_json_output, assess_answerability_metric

from structured_rag.models import Experiment, PromptWithResponse, PromptingMethod, SingleTestResult
from structured_rag.models import test_params, test_to_output_model

# Need to clean up how `task_specific_ground_truth` is interfaced
def run_single_test(output_model: Optional[BaseModel],
                    program, test_type, title, context, question, answer, task_specific_ground_truth) -> SingleTestResult:
    try:
        if test_type == "ParaphraseQuestions":
            # will need to fix this in the `dspy_Program` code
            output = program.forward(output_model, test_type, question=question, output_model=output_model) # output_model if present
        elif test_type == "RAGAS":
            output = program.forward(output_model, test_type, context, question, answer, output_model=output_model) # output_model if present
        else:
            output = program.forward(output_model, test_type, context, question, output_model=output_model) # output_model if present
        
        print(f"{Colors.CYAN}{program.__class__.__name__} Output: {output}{Colors.ENDC}\n")
        
        is_valid = False
        task_metric = 0
        
        if is_valid_json_output(output, test_type):
            print(f"{Colors.GREEN}Valid output for {test_type}{Colors.ENDC}")
            is_valid = True
            if test_type == "AssessAnswerability":
                assess_answerability_response = json.loads(output)["answerable_question"]
                print(f"{Colors.BOLD}Assess Answerability Response: {assess_answerability_response}{Colors.ENDC}")
                task_metric = assess_answerability_metric(assess_answerability_response, task_specific_ground_truth)
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
    
    if args.test not in test_params:
        raise ValueError(f"Unsupported test: {args.test}")
    
    test_to_run = test_params[args.test]
    output_model = test_to_output_model[args.test]

    dspy_program = dspy_Program(test_params=test_to_run, 
                                model_name=args.model_name, model_provider=args.model_provider, api_key=args.api_key)
    fstring_program = fstring_Program(test_params=test_to_run, 
                                     model_name=args.model_name, model_provider=args.model_provider, api_key=args.api_key)

    dspy_experiment = Experiment(
        test_name=args.test,
        model_name=args.model_name,
        prompting_method=PromptingMethod.dspy,
        num_successes=0,
        total_task_performance=0,
        num_attempts=0,
        success_rate=0,
        average_task_performance=0,
        total_time=0,
        all_responses=[],
        failed_responses=[]
    )

    fstring_experiment = Experiment(
        test_name=args.test,
        model_name=args.model_name,
        prompting_method=PromptingMethod.fstring,
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

    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    # ToDo, move to config or cli argument
    PARALLEL_EXECUTION = False

    if PARALLEL_EXECUTION:
        lock = threading.Lock()

        def process_entry(entry, dspy_program, fstring_program, args):
            title = entry.get('title', '')
            context = entry.get('context', '')
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            answerable = entry.get('answerable', '')
            
            print(f"{Colors.UNDERLINE}Title: {title}{Colors.ENDC}")
            print(f"{Colors.UNDERLINE}Question: {question}{Colors.ENDC}\n")
            
            dspy_single_test_result = run_single_test(dspy_program, args.test, title, context, 
                                                      question, answerable)
            fstring_single_test_result = run_single_test(fstring_program, args.test, title, context, 
                                                         question, answerable)
            
            return dspy_single_test_result, fstring_single_test_result

        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_entry = {executor.submit(process_entry, entry, dspy_program, fstring_program, args): entry for entry in json_data}
            
            for future in as_completed(future_to_entry):
                dspy_single_test_result, fstring_single_test_result = future.result()
                
                with lock:
                    if dspy_single_test_result:
                        dspy_experiment.all_responses.append(dspy_single_test_result.prompt_with_response)
                        dspy_experiment.num_attempts += 1
                        if dspy_single_test_result.is_valid:
                            dspy_experiment.num_successes += 1
                            dspy_experiment.total_task_performance += dspy_single_test_result.task_metric
                        else:
                            dspy_experiment.failed_responses.append(dspy_single_test_result.prompt_with_response)
                    
                    if fstring_single_test_result:
                        fstring_experiment.all_responses.append(fstring_single_test_result.prompt_with_response)
                        fstring_experiment.num_attempts += 1
                        if fstring_single_test_result.is_valid:
                            fstring_experiment.num_successes += 1
                            fstring_experiment.total_task_performance += fstring_single_test_result.task_metric
                        else:
                            fstring_experiment.failed_responses.append(fstring_single_test_result.prompt_with_response)
                
                print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
    else:
        for entry in json_data:
            title = entry.get('title', '')
            context = entry.get('context', '')
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            answerable = entry.get('answerable', '')
            
            print(f"{Colors.UNDERLINE}Title: {title}{Colors.ENDC}")
            print(f"{Colors.UNDERLINE}Question: {question}{Colors.ENDC}\n")
            
            if args.test == "AssessAnswerability":
                dspy_single_test_result = run_single_test(dspy_program, args.test, title, context, 
                                                        question, answerable, answerable)
                # add the `output_model` argument here
                # ah, so the problem is that `run_single_test` assumes a program with the same arguments
                # now the f-string program requires different arguments. `
                fstring_single_test_result = run_single_test(fstring_program, args.test, title, context, 
                                                            question, answerable, answerable)
            else:
                print(f"{Colors.RED}NOT IMPLEMENTED YET!!\n{Colors.ENDC}")
                print(f"{Colors.CYAN}Need to add the `task_specific_ground_truth` for each StructuredRAG test.{Colors.ENDC}")
                print(f"{Colors.GREEN}The tests implemented now are `AssessAnswerability`.{Colors.ENDC}")
            
            if dspy_single_test_result:
                dspy_experiment.all_responses.append(dspy_single_test_result.prompt_with_response)
                dspy_experiment.num_attempts += 1
                if dspy_single_test_result.is_valid:
                    dspy_experiment.num_successes += 1
                    dspy_experiment.total_task_performance += dspy_single_test_result.task_metric
                else:
                    dspy_experiment.failed_responses.append(dspy_single_test_result.prompt_with_response)
            
            if fstring_single_test_result:
                fstring_experiment.all_responses.append(fstring_single_test_result.prompt_with_response)
                fstring_experiment.num_attempts += 1
                if fstring_single_test_result.is_valid:
                    fstring_experiment.num_successes += 1
                    fstring_experiment.total_task_performance += fstring_single_test_result.task_metric
                else:
                    fstring_experiment.failed_responses.append(fstring_single_test_result.prompt_with_response)
            
            print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
        
        print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
    
    total_time = time.time() - total_start_time
    dspy_experiment.total_time = int(total_time)
    fstring_experiment.total_time = int(total_time)

    print(f"{Colors.HEADER}Time to run test {args.test} with model {args.model_name} = {total_time} seconds.")

    # Print final scores
    print(f"{Colors.HEADER}Final Scores:{Colors.ENDC}")
    print(f"{Colors.BOLD}JSON Success Rates{Colors.ENDC}")
    print(f"{Colors.BOLD}DSPy: {Colors.GREEN}{dspy_experiment.num_successes}/{dspy_experiment.num_attempts} ({dspy_experiment.num_successes/dspy_experiment.num_attempts:.2%}){Colors.ENDC}")
    print(f"{Colors.BOLD}f-string: {Colors.GREEN}{fstring_experiment.num_successes}/{fstring_experiment.num_attempts} ({fstring_experiment.num_successes/fstring_experiment.num_attempts:.2%}){Colors.ENDC}")

    print(f"{Colors.BOLD}Average Task Performance{Colors.ENDC}")
    print(f"{Colors.BOLD}DSPy: {dspy_experiment.average_task_performance:.2f}{Colors.ENDC}")
    print(f"{Colors.BOLD}f-string: {fstring_experiment.average_task_performance:.2f}{Colors.ENDC}")

    # Save results to JSON file in the specified save directory
    os.makedirs("../results/" + args.save_dir, exist_ok=True)
    dspy_result_file = os.path.join("../results/" + args.save_dir, f"{args.test}-{args.model_name}-dspy.json")
    fstring_result_file = os.path.join("../results/" + args.save_dir, f"{args.test}-{args.model_name}-fstring.json")
    
    # calculate success rate
    dspy_experiment.success_rate = dspy_experiment.num_successes / dspy_experiment.num_attempts
    fstring_experiment.success_rate = fstring_experiment.num_successes / fstring_experiment.num_attempts

    dspy_experiment.average_task_performance = dspy_experiment.total_task_performance / dspy_experiment.num_attempts
    fstring_experiment.average_task_performance = fstring_experiment.total_task_performance / fstring_experiment.num_attempts

    with open(dspy_result_file, "w") as f:
        json.dump(dspy_experiment.dict(), f, indent=2)
    
    with open(fstring_result_file, "w") as f:
        json.dump(fstring_experiment.dict(), f, indent=2)

    print(f"\nResults saved in {dspy_result_file} and {fstring_result_file}")

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
    # Add --decode, "prompt" "constrained-gen" 
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the results")

    args = parser.parse_args()
    run_test(args)