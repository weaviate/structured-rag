import json
import os
import argparse
import time
from src.dspy_program import dspy_Program
from src.fstring_inference import fstring_Program
import dspy

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_json_from_file(filename):
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"{Colors.RED}Error: File '{filename}' not found.{Colors.ENDC}")
        return None
    except json.JSONDecodeError:
        print(f"{Colors.RED}Error: Invalid JSON format in '{filename}'.{Colors.ENDC}")
        return None

def is_valid_json_output(output, test_type):
    try:
        parsed = json.loads(output)
        if test_type == "GenerateAnswer":
            return isinstance(parsed.get("answer"), str)
        elif test_type == "RateContext":
            score = parsed.get("context_score")
            return isinstance(score, int) and 0 <= score <= 5
        elif test_type == "AssessAnswerability":
            return isinstance(parsed.get("answerable_question"), bool)
        elif test_type == "ParaphraseQuestions":
            questions = parsed.get("paraphrased_questions")
            return isinstance(questions, list) and all(isinstance(q, str) for q in questions)
        elif test_type == "GenerateAnswerWithConfidence":
            return isinstance(parsed.get("Answer"), str) and isinstance(parsed.get("Confidence"), int) and 0 <= parsed["Confidence"] <= 5
        elif test_type == "GenerateAnswersWithConfidence":
            answers = parsed
            return isinstance(answers, list) and all(isinstance(a.get("Answer"), str) and isinstance(a.get("Confidence"), int) and 0 <= a["Confidence"] <= 5 for a in answers)
        else:
            return False
    except json.JSONDecodeError:
        return False

def run_test(program, test_type, context, question):
    try:
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
    trial_dirs = [d for d in os.listdir('.') if d.startswith("results-trial-")]
    if not trial_dirs:
        return 0
    return max(int(d.split("-")[-1]) for d in trial_dirs)

def get_or_create_trial_directory(test_type):
    latest_trial = get_latest_trial_number()
    
    if latest_trial > 0:
        latest_trial_dir = f"results-trial-{latest_trial}"
        # Check if the test_type result already exists in the latest trial
        if not any(f.startswith(f"{test_type}-") and f.endswith(".json") for f in os.listdir(latest_trial_dir)):
            return latest_trial_dir
    
    # Create a new trial directory
    new_trial = latest_trial + 1
    new_trial_dir = f"results-trial-{new_trial}"
    os.makedirs(new_trial_dir, exist_ok=True)
    return new_trial_dir

def main(args):
    filename = "wiki-answerable-questions.json"
    json_data = load_json_from_file(filename)

    dspy_program = dspy_Program(model_name=args.model_name, model_provider=args.model_provider, api_key=args.api_key)
    fstring_program = fstring_Program(model_name=args.model_name, model_provider=args.model_provider, api_key=args.api_key)

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
    with open(f"{trial_dir}/{args.test}-{args.model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved in {trial_dir}")

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