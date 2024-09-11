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

def calculate_averages(experiments: pd.DataFrame) -> pd.DataFrame:
    return experiments.groupby(['test_name', 'model_name', 'prompting_method']).mean().reset_index()