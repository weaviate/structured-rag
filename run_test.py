import json
import os
import argparse
from dspy_program import dspy_Program
from fstring_inference import fstring_Program
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
    if test_type == "ParaphraseQuestions":
        output = program.forward(test_type, question=question)
    else:
        output = program.forward(test_type, context, question)
    
    print(f"{Colors.CYAN}{program.__class__.__name__} Output: {output}{Colors.ENDC}\n")
    
    if is_valid_json_output(output, test_type):
        print(f"{Colors.GREEN}Valid output for {test_type}{Colors.ENDC}")
        return 1
    else:
        print(f"{Colors.RED}Invalid output for {test_type}{Colors.ENDC}")
        return 0

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
        "total_questions": 0
    }

    for entry in json_data:
        context = entry.get('abstract', '')
        question = entry.get('answerable_question', '')
        results["total_questions"] += 1
        
        print(f"{Colors.UNDERLINE}Question: {question}{Colors.ENDC}\n")
        
        results["dspy_score"] += run_test(dspy_program, args.test, context, question)
        results["fstring_score"] += run_test(fstring_program, args.test, context, question)
        
        print(f"\n{Colors.BOLD}==============={Colors.ENDC}\n")

    # Print final scores
    print(f"{Colors.HEADER}Final Scores:{Colors.ENDC}")
    print(f"{Colors.BOLD}DSPy: {Colors.GREEN}{results['dspy_score']}/{results['total_questions']} ({results['dspy_score']/results['total_questions']:.2%}){Colors.ENDC}")
    print(f"{Colors.BOLD}f-string: {Colors.GREEN}{results['fstring_score']}/{results['total_questions']} ({results['fstring_score']/results['total_questions']:.2%}){Colors.ENDC}")

    # Save results to JSON file
    os.makedirs("results", exist_ok=True)
    with open(f"results/{args.test}-{args.model_name}.json", "w") as f:
        json.dump(results, f, indent=2)

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