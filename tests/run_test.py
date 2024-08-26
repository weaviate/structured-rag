import json
import os
import argparse
import time
from src.dspy_program import dspy_Program
from src.fstring_program import fstring_Program
import dspy

from tests.helpers import Colors, load_json_from_file
from tests.metrics import is_valid_json_output

def run_test(program, test_type, context, question):
    try:
        # probably a better way to do this
        if test_type == "ParaphraseQuestions":
            output = program.forward(test_type, question=question)
        else:
            output = program.forward(test_type, context, question)
        
        print(f"{Colors.CYAN}{program.__class__.__name__} Output: {output}{Colors.ENDC}\n")
        
        if is_valid_json_output(output, test_type):
            print(f"{Colors.GREEN}Valid output for {test_type}{Colors.ENDC}")
            return 1, True  # Success, count this attempt
        else:
            print(f"{Colors.RED}Invalid output for {test_type}{Colors.ENDC}")
            return 0, True  # Failure, but still count this attempt
    
    except Exception as e:
        print(f"{Colors.YELLOW}Error occurred: {str(e)}{Colors.ENDC}")
        print(f"{Colors.RED}Skipping this test due to error.{Colors.ENDC}")
        return 0, False  # Failure due to error, don't count this attempt

def get_latest_trial_number():
    base_dir = "experimental-results"
    if not os.path.exists(base_dir):
        return 0
    trial_dirs = [d for d in os.listdir(base_dir) if d.startswith("trial-")]
    if not trial_dirs:
        return 0
    return max(int(d.split("-")[-1]) for d in trial_dirs)

def get_or_create_trial_directory(test_type):
    base_dir = "experimental-results"
    os.makedirs(base_dir, exist_ok=True)
    
    latest_trial = get_latest_trial_number()
    
    if latest_trial > 0:
        latest_trial_dir = os.path.join(base_dir, f"trial-{latest_trial}")
        # Check if the test_type result already exists in the latest trial
        if not any(f.startswith(f"{test_type}-") and f.endswith(".json") for f in os.listdir(latest_trial_dir)):
            return latest_trial_dir
    
    # Create a new trial directory
    new_trial = latest_trial + 1
    new_trial_dir = os.path.join(base_dir, f"trial-{new_trial}")
    os.makedirs(new_trial_dir, exist_ok=True)
    return new_trial_dir

def main(args):
    filename = "wiki-answerable-questions.json"
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

    results = {
        "test_type": args.test,
        "model_name": args.model_name,
        "model_provider": args.model_provider,
        "dspy_score": 0,
        "fstring_score": 0,
        "dspy_total_attempts": 0,
        "fstring_total_attempts": 0
    }

    total_start_time = time.time()

    for entry in json_data:
        context = entry.get('abstract', '')
        answerable_question = entry.get('answerable_question', '')
        unanswerable_question = entry.get('unanswerable_question', '')
        
        # Test with answerable question
        print(f"{Colors.UNDERLINE}Answerable Question: {answerable_question}{Colors.ENDC}\n")
        dspy_score, dspy_counted = run_test(dspy_program, args.test, context, answerable_question)
        fstring_score, fstring_counted = run_test(fstring_program, args.test, context, answerable_question)
        results["dspy_score"] += dspy_score
        results["fstring_score"] += fstring_score
        results["dspy_total_attempts"] += int(dspy_counted)
        results["fstring_total_attempts"] += int(fstring_counted)
        
        print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
        
        # Test with unanswerable question
        print(f"{Colors.UNDERLINE}Unanswerable Question: {unanswerable_question}{Colors.ENDC}\n")
        dspy_score, dspy_counted = run_test(dspy_program, args.test, context, unanswerable_question)
        fstring_score, fstring_counted = run_test(fstring_program, args.test, context, unanswerable_question)
        results["dspy_score"] += dspy_score
        results["fstring_score"] += fstring_score
        results["dspy_total_attempts"] += int(dspy_counted)
        results["fstring_total_attempts"] += int(fstring_counted)
        
        print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")
    
    print(f"{Colors.HEADER}Time to run test {args.test} with model {args.model_name} = {time.time() - total_start_time} seconds.")

    # Print final scores
    print(f"{Colors.HEADER}Final Scores:{Colors.ENDC}")
    print(f"{Colors.BOLD}DSPy: {Colors.GREEN}{results['dspy_score']}/{results['dspy_total_attempts']} ({results['dspy_score']/results['dspy_total_attempts']:.2%}){Colors.ENDC}")
    print(f"{Colors.BOLD}f-string: {Colors.GREEN}{results['fstring_score']}/{results['fstring_total_attempts']} ({results['fstring_score']/results['fstring_total_attempts']:.2%}){Colors.ENDC}")

    # Save results to JSON file in the appropriate trial directory
    trial_dir = get_or_create_trial_directory(args.test)
    result_file = os.path.join(trial_dir, f"{args.test}-{args.model_name}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in {result_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM testing with different models.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use")
    parser.add_argument("--model_provider", type=str, required=True, choices=["ollama", "google"], help="Provider of the model")
    parser.add_argument("--api_key", type=str, required=False, help="API key for the model provider (if required)")
    parser.add_argument("--test", type=str, required=True, choices=[
        "GenerateAnswer", "RateContext", "AssessAnswerability", "ParaphraseQuestions",
        "RateMultipleAspects", "GenerateAnswerWithConfidence", "GenerateAnswersWithConfidence"
    ], help="Type of test to run")

    args = parser.parse_args()
    main(args)