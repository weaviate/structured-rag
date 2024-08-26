import json
import os
from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def read_json_files() -> List[Dict]:
    results = []
    for directory in os.listdir('.'):
        if directory.startswith("results-trial-"):
            for filename in os.listdir(directory):
                if filename.endswith(".json"):
                    with open(os.path.join(directory, filename), "r") as f:
                        data = json.load(f)
                        data['file_path'] = os.path.join(directory, filename)
                        results.append(data)
    return results

def aggregate_results(results: List[Dict]) -> Dict:
    summary = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {
        "dspy_total": 0,
        "fstring_total": 0,
        "total_questions": 0,
        "runs": 0
    }))))
    
    for result in results:
        test_type = result["test_type"]
        model_name = result["model_name"]
        model_provider = result["model_provider"]
        trial = os.path.basename(os.path.dirname(result["file_path"]))
        
        summary[test_type][model_name][model_provider][trial]["dspy_total"] += result["dspy_score"]
        summary[test_type][model_name][model_provider][trial]["fstring_total"] += result["fstring_score"]
        
        # Handle both old and new JSON formats
        if "total_questions" in result:
            total_questions = result["total_questions"]
        else:
            total_questions = max(result.get("dspy_total_attempts", 0), result.get("fstring_total_attempts", 0))
        
        summary[test_type][model_name][model_provider][trial]["total_questions"] += total_questions
        summary[test_type][model_name][model_provider][trial]["runs"] += 1
    
    for test_type, models in summary.items():
        for model_name, providers in models.items():
            for provider, trials in providers.items():
                for trial, data in trials.items():
                    data["dspy_average"] = data["dspy_total"] / data["runs"] if data["runs"] > 0 else 0
                    data["fstring_average"] = data["fstring_total"] / data["runs"] if data["runs"] > 0 else 0
                    data["average_questions"] = data["total_questions"] / data["runs"] if data["runs"] > 0 else 0
    
    return dict(summary)

def print_summary(summary: Dict):
    print("Experiment Results Summary:")
    print("===========================")
    for test_type, models in summary.items():
        print(f"\nTest: {test_type}")
        for model_name, providers in models.items():
            for provider, trials in providers.items():
                print(f"\nModel: {model_name} (Provider: {provider})")
                for trial, data in trials.items():
                    print(f"  Trial: {trial}")
                    print(f"    Number of runs: {data['runs']}")
                    print(f"    Average questions per run: {data['average_questions']:.2f}")
                    if data['average_questions'] > 0:
                        print(f"    DSPy average score: {data['dspy_average']:.2f} ({data['dspy_average']/data['average_questions']:.2%})")
                        print(f"    f-string average score: {data['fstring_average']:.2f} ({data['fstring_average']/data['average_questions']:.2%})")
                    else:
                        print(f"    DSPy average score: {data['dspy_average']:.2f} (N/A)")
                        print(f"    f-string average score: {data['fstring_average']:.2f} (N/A)")

def create_bar_chart(summary: Dict, trial: str = None):
    tests = list(summary.keys())
    models = list(set(model for test in summary.values() for model in test.keys()))
    
    fig, ax = plt.subplots(figsize=(20, 10))  # Increased figure size for better readability
    
    num_models = len(models)
    group_width = 0.8
    bar_width = group_width / (2 * num_models)  # We have 2 bars per model (DSPy/FF and f-string)
    
    # Define the color scheme
    color_scheme = {
        'gemini-1.5-pro f-String': 'forestgreen',
        'gemini-1.5-pro FF': 'dodgerblue',
        'llama3:instruct f-String': 'red',
        'llama3:instruct FF': 'darkorange'
    }
    
    for i, test in enumerate(tests):
        for j, model in enumerate(models):
            if model in summary[test]:
                providers = summary[test][model]
                if trial:
                    provider_data = next((provider[trial] for provider in providers.values() if trial in provider), None)
                    if provider_data and provider_data["average_questions"] > 0:
                        dspy_avg = provider_data["dspy_average"] / provider_data["average_questions"]
                        fstring_avg = provider_data["fstring_average"] / provider_data["average_questions"]
                    else:
                        dspy_avg = fstring_avg = 0
                else:
                    # For overall average, we need to aggregate across all trials
                    all_trials_data = [t for provider in providers.values() for t in provider.values()]
                    if all_trials_data:
                        total_questions = sum(t["average_questions"] for t in all_trials_data)
                        if total_questions > 0:
                            dspy_avg = sum(t["dspy_average"] for t in all_trials_data) / total_questions
                            fstring_avg = sum(t["fstring_average"] for t in all_trials_data) / total_questions
                        else:
                            dspy_avg = fstring_avg = 0
                    else:
                        dspy_avg = fstring_avg = 0
                
                # Calculate positions for the bars
                base_position = i + (j - num_models/2 + 0.5) * group_width / num_models
                dspy_position = base_position - bar_width/2
                fstring_position = base_position + bar_width/2
                
                # Plot the bars with specified coloring
                ax.bar(dspy_position, dspy_avg, bar_width, color=color_scheme[f'{model} FF'], alpha=0.8)
                ax.bar(fstring_position, fstring_avg, bar_width, color=color_scheme[f'{model} f-String'], alpha=0.8)

    ax.set_ylabel('Average Score (as percentage)', fontsize=16)  # Increased font size
    title = 'Model Performance Comparison by Test Type'
    if trial:
        title += f' - {trial}'
    else:
        title += ' - Average Across All Trials'
    ax.set_title(title, fontsize=18)  # Increased font size
    
    ax.set_xticks(range(len(tests)))
    ax.set_xticklabels(tests, rotation=45, ha='right', fontsize=12)  # Increased font size
    
    # Adjust y-axis to start from 0 and end at 1 (100%)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    ax.tick_params(axis='y', labelsize=14)  # Increased y-axis tick label font size
    
    # Add gridlines for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Create custom legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, edgecolor='none', alpha=0.8) 
                       for color in color_scheme.values()]
    legend_labels = list(color_scheme.keys())
    
    ax.legend(legend_elements, legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', 
              borderaxespad=0., fontsize=14)  # Increased legend font size
    
    plt.tight_layout()
    filename = 'model_comparison.png' if not trial else f'model_comparison_{trial}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=300)  # Increased DPI for better quality
    plt.close()

    print(f"Bar chart saved as '{filename}'")

def main():
    results = read_json_files()
    summary = aggregate_results(results)
    
    print_summary(summary)
    
    # Create a plot for each trial
    trials = set()
    for test in summary.values():
        for model in test.values():
            for provider in model.values():
                trials.update(provider.keys())
    
    for trial in sorted(trials):
        create_bar_chart(summary, trial)
    
    # Create a plot for the average across all trials
    create_bar_chart(summary)
    
    # Save aggregated results
    with open("aggregated_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nAggregated results saved to aggregated_results.json")

if __name__ == "__main__":
    main()