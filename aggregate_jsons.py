import json
import os
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt

def read_json_files(directory: str) -> List[Dict]:
    results = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                results.append(json.load(f))
    return results

def aggregate_results(results: List[Dict]) -> Dict:
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "dspy_total": 0,
        "fstring_total": 0,
        "total_questions": 0,
        "runs": 0
    })))
    
    for result in results:
        test_type = result["test_type"]
        model_name = result["model_name"]
        model_provider = result["model_provider"]
        
        summary[test_type][model_name][model_provider]["dspy_total"] += result["dspy_score"]
        summary[test_type][model_name][model_provider]["fstring_total"] += result["fstring_score"]
        summary[test_type][model_name][model_provider]["total_questions"] += result["total_questions"]
        summary[test_type][model_name][model_provider]["runs"] += 1
    
    for test_type, models in summary.items():
        for model_name, providers in models.items():
            for provider, data in providers.items():
                data["dspy_average"] = data["dspy_total"] / data["runs"]
                data["fstring_average"] = data["fstring_total"] / data["runs"]
                data["average_questions"] = data["total_questions"] / data["runs"]
    
    return dict(summary)

def print_summary(summary: Dict):
    print("Experiment Results Summary:")
    print("===========================")
    for test_type, models in summary.items():
        print(f"\nTest: {test_type}")
        for model_name, providers in models.items():
            for provider, data in providers.items():
                print(f"\nModel: {model_name} (Provider: {provider})")
                print(f"Number of runs: {data['runs']}")
                print(f"Average questions per run: {data['average_questions']:.2f}")
                print(f"DSPy average score: {data['dspy_average']:.2f} ({data['dspy_average']/data['average_questions']:.2%})")
                print(f"f-string average score: {data['fstring_average']:.2f} ({data['fstring_average']/data['average_questions']:.2%})")

def create_bar_chart(summary: Dict):
    tests = list(summary.keys())
    llama3_dspy = []
    llama3_fstring = []
    gemini_dspy = []
    gemini_fstring = []

    for test in tests:
        for model, providers in summary[test].items():
            for provider, data in providers.items():
                if "llama" in model.lower():
                    llama3_dspy.append(data['dspy_average'] / data['average_questions'])
                    llama3_fstring.append(data['fstring_average'] / data['average_questions'])
                elif "gemini" in model.lower():
                    gemini_dspy.append(data['dspy_average'] / data['average_questions'])
                    gemini_fstring.append(data['fstring_average'] / data['average_questions'])

    x = range(len(tests))
    width = 0.2

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar([i - width*1.5 for i in x], llama3_dspy, width, label='Llama3 FF', color='#1f77b4')
    ax.bar([i - width/2 for i in x], llama3_fstring, width, label='Llama3 f-string', color='#2ca02c')
    ax.bar([i + width/2 for i in x], gemini_dspy, width, label='Gemini FF', color='#ff7f0e')
    ax.bar([i + width*1.5 for i in x], gemini_fstring, width, label='Gemini f-string', color='#d62728')

    ax.set_ylabel('Average Score (as percentage)')
    ax.set_title('Model Performance Comparison by Test Type')
    ax.set_xticks(x)
    ax.set_xticklabels(tests, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

    print("Bar chart saved as 'model_comparison.png'")

def main():
    results_dir = "results"
    results = read_json_files(results_dir)
    summary = aggregate_results(results)
    
    print_summary(summary)
    create_bar_chart(summary)
    
    # Save aggregated results
    with open("aggregated_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nAggregated results saved to aggregated_results.json")

if __name__ == "__main__":
    main()