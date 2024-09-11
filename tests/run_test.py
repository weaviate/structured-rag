import json
import os
import argparse
import datetime
import time
from structured_rag.dspy_program import dspy_Program
from structured_rag.fstring_program import fstring_Program
import dspy

from helpers import Colors, load_json_from_file
from metrics import is_valid_json_output

from models import Experiment, PromptWithResponse, PromptingMethod

def run_single_test(program, test_type, context, question):
    try:
        # probably a better way to do this
        if test_type == "ParaphraseQuestions":
            output = program.forward(test_type, question=question)
        else:
            output = program.forward(test_type, context, question)
        
        print(f"{Colors.CYAN}{program.__class__.__name__} Output: {output}{Colors.ENDC}\n")
        
        if is_valid_json_output(output, test_type):
            print(f"{Colors.GREEN}Valid output for {test_type}{Colors.ENDC}")
            return PromptWithResponse(prompt=f"{context}\n{question}", response=output), True
        else:
            print(f"{Colors.RED}Invalid output for {test_type}{Colors.ENDC}")
            return PromptWithResponse(prompt=f"{context}\n{question}", response=output), False
    
    except Exception as e:
        print(f"{Colors.YELLOW}Error occurred: {str(e)}{Colors.ENDC}")
        print(f"{Colors.RED}Skipping this test due to error.{Colors.ENDC}")
        return None, False

# ToDo, add calculation of success rate to the Experiment object

def run_test(args):
    filename = "../data/wiki-answerable-questions.json"
    json_data = load_json_from_file(filename)
    # would also like to refactor this into a config file
    test_params = {
        "GenerateAnswer": {
            "task_instructions": "Assess the context and answer the question. If the context does not contain sufficient information to answer the question, respond with \"NOT ENOUGH CONTEXT\".",
            "response_format": '{"answer": "string"}'
        },
        "RateContext": {
            "task_instructions": "Assess how well the context helps answer the question.",
            "response_format": '{"context_score": "int (0-5)"}'
        },
        "AssessAnswerability": {
            "task_instructions": "Determine if the question is answerable based on the context.",
            "response_format": '{"answerable_question": "bool"}'
        },
        "ParaphraseQuestions": {
            "task_instructions": "Generate 3 paraphrased versions of the given question.",
            "response_format": '{"paraphrased_questions": ["string", "string", "string"]}'
        },
        "GenerateAnswerWithConfidence": {
            "task_instructions": "Generate an answer with a confidence score.",
            "response_format": '{"Answer": "string", "Confidence": "int (0-5)"}'
        },
        "GenerateAnswersWithConfidence": {
            "task_instructions": "Generate multiple answers with confidence scores.",
            "response_format": '[{"Answer": "string", "Confidence": "int (0-5)"}, ...]'
        }
    }

    if args.test not in test_params:
        raise ValueError(f"Unsupported test: {args.test}")
    
    test_to_run = test_params[args.test]

    dspy_program = dspy_Program(test_params=test_to_run, 
                                model_name=args.model_name, model_provider=args.model_provider, api_key=args.api_key)
    fstring_program = fstring_Program(test_params=test_to_run, 
                                     model_name=args.model_name, model_provider=args.model_provider, api_key=args.api_key)

    dspy_experiment = Experiment(
        test_name=args.test,
        model_name=args.model_name,
        prompting_method=PromptingMethod.dspy,
        num_successes=0,
        num_attempts=0,
        success_rate=0,
        total_time=0,
        all_responses=[],
        failed_responses=[]
    )

    fstring_experiment = Experiment(
        test_name=args.test,
        model_name=args.model_name,
        prompting_method=PromptingMethod.fstring,
        num_successes=0,
        num_attempts=0,
        success_rate=0,
        total_time=0,
        all_responses=[],
        failed_responses=[]
    )

    total_start_time = time.time()

    for entry in json_data:
        context = entry.get('abstract', '')
        answerable_question = entry.get('answerable_question', '')
        unanswerable_question = entry.get('unanswerable_question', '')
        
        # Test with answerable question
        print(f"{Colors.UNDERLINE}Answerable Question: {answerable_question}{Colors.ENDC}\n")
        dspy_response, dspy_success = run_single_test(dspy_program, args.test, context, answerable_question)
        fstring_response, fstring_success = run_single_test(fstring_program, args.test, context, answerable_question)
        
        if dspy_response:
            dspy_experiment.all_responses.append(dspy_response)
            dspy_experiment.num_attempts += 1
            if dspy_success:
                dspy_experiment.num_successes += 1
            else:
                dspy_experiment.failed_responses.append(dspy_response)
        
        if fstring_response:
            fstring_experiment.all_responses.append(fstring_response)
            fstring_experiment.num_attempts += 1
            if fstring_success:
                fstring_experiment.num_successes += 1
            else:
                fstring_experiment.failed_responses.append(fstring_response)
        
        print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
        
        # Test with unanswerable question
        print(f"{Colors.UNDERLINE}Unanswerable Question: {unanswerable_question}{Colors.ENDC}\n")
        dspy_response, dspy_success = run_single_test(dspy_program, args.test, context, unanswerable_question)
        fstring_response, fstring_success = run_single_test(fstring_program, args.test, context, unanswerable_question)
        
        if dspy_response:
            dspy_experiment.all_responses.append(dspy_response)
            dspy_experiment.num_attempts += 1
            if dspy_success:
                dspy_experiment.num_successes += 1
            else:
                dspy_experiment.failed_responses.append(dspy_response)
        
        if fstring_response:
            fstring_experiment.all_responses.append(fstring_response)
            fstring_experiment.num_attempts += 1
            if fstring_success:
                fstring_experiment.num_successes += 1
            else:
                fstring_experiment.failed_responses.append(fstring_response)
        
        print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
    
    total_time = time.time() - total_start_time
    dspy_experiment.total_time = int(total_time)
    fstring_experiment.total_time = int(total_time)

    print(f"{Colors.HEADER}Time to run test {args.test} with model {args.model_name} = {total_time} seconds.")

    # Print final scores
    print(f"{Colors.HEADER}Final Scores:{Colors.ENDC}")
    print(f"{Colors.BOLD}DSPy: {Colors.GREEN}{dspy_experiment.num_successes}/{dspy_experiment.num_attempts} ({dspy_experiment.num_successes/dspy_experiment.num_attempts:.2%}){Colors.ENDC}")
    print(f"{Colors.BOLD}f-string: {Colors.GREEN}{fstring_experiment.num_successes}/{fstring_experiment.num_attempts} ({fstring_experiment.num_successes/fstring_experiment.num_attempts:.2%}){Colors.ENDC}")

    # Save results to JSON file in the specified save directory
    os.makedirs(args.save_dir, exist_ok=True)
    dspy_result_file = os.path.join(args.save_dir, f"{args.test}-{args.model_name}-dspy.json")
    fstring_result_file = os.path.join(args.save_dir, f"{args.test}-{args.model_name}-fstring.json")
    
    # calculate success rate
    dspy_experiment.success_rate = dspy_experiment.num_successes / dspy_experiment.num_attempts
    fstring_experiment.success_rate = fstring_experiment.num_successes / fstring_experiment.num_attempts

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
        "GenerateAnswer", "RateContext", "AssessAnswerability", "ParaphraseQuestions",
        "RateMultipleAspects", "GenerateAnswerWithConfidence", "GenerateAnswersWithConfidence"
    ], help="Type of test to run")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the results")

    args = parser.parse_args()
    run_test(args)