import os
import json
import pandas as pd
from models import PromptWithResponse, PromptingMethod, Experiment

def load_experiments(directory: str) -> pd.DataFrame:
    experiments = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), 'r') as f:
                data = json.load(f)
                experiment = Experiment(**data)
                experiments.append({
                    'test_name': experiment.test_name,
                    'model_name': experiment.model_name,
                    'prompting_method': experiment.prompting_method,
                    'num_successes': experiment.num_successes,
                    'num_attempts': experiment.num_attempts,
                    'success_rate': experiment.success_rate,
                    'total_time': experiment.total_time,
                    'avg_response_time': experiment.total_time / experiment.num_attempts
                })
    return pd.DataFrame(experiments)

def calculate_avg_accuracy_per_prompting_method(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['prompting_method'])['success_rate'].mean().reset_index()

def calculate_avg_accuracy_per_model(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['model_name'])['success_rate'].mean().reset_index()

def calculate_avg_accuracy_per_test(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['test_name'])['success_rate'].mean().reset_index()

def calculate_avg_accuracy_per_prompting_method_per_model(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['model_name', 'prompting_method'])['success_rate'].mean().reset_index()

def calculate_avg_accuracy_per_prompting_method_per_test(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['test_name', 'prompting_method'])['success_rate'].mean().reset_index()

def calculate_avg_accuracy_per_model_per_test(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['model_name', 'test_name'])['success_rate'].mean().reset_index()

def calculate_overall_average(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments['success_rate'].mean()

def calculate_avg_response_time_per_model(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['model_name'])['avg_response_time'].mean().reset_index()

def list_all_results(experiments: pd.DataFrame) -> pd.DataFrame:
    for index, row in experiments.iterrows():
        print(row["test_name"], row["model_name"], row["prompting_method"])
        print(f"\033[92mSuccess rate: {row['success_rate']}\033[0m")

if __name__ == "__main__":
    experiments = load_experiments("experimental-results-9-11-24")
    print("\033[92m\nAverage accuracy per prompting method:\n\033[0m")
    print(calculate_avg_accuracy_per_prompting_method(experiments))

    print("\033[92m\nAverage accuracy per model:\n\033[0m")
    print(calculate_avg_accuracy_per_model(experiments))

    print("\033[92m\nAverage accuracy per test:\n\033[0m")
    print(calculate_avg_accuracy_per_test(experiments))

    print("\033[92m\nAverage accuracy per prompting method per model:\n\033[0m")
    print(calculate_avg_accuracy_per_prompting_method_per_model(experiments))

    print("\033[92m\nAverage accuracy per prompting method per test:\n\033[0m")
    print(calculate_avg_accuracy_per_prompting_method_per_test(experiments))

    print("\033[92m\nAverage accuracy per model per test:\n\033[0m")
    print(calculate_avg_accuracy_per_model_per_test(experiments))

    print("\033[92m\nOverall average accuracy:\n\033[0m")
    print(calculate_overall_average(experiments))

    print("\033[92m\nAverage response time per model:\n\033[0m")
    print(calculate_avg_response_time_per_model(experiments))

    print("\033[92m\nList all results:\n\033[0m")
    list_all_results(experiments)
