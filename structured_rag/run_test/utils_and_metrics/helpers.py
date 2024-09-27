import json
import os
import datetime

import pandas as pd

from structured_rag.models import Experiment

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
                    'avg_response_time': experiment.total_time / experiment.num_attempts,
                    'failed_responses': experiment.failed_responses
                })
    return pd.DataFrame(experiments)

def count_objects_in_json_file(filename):
  """Loads JSON data from a file and returns the number of objects in the list."""
  with open(filename, "r") as f:
      data = json.load(f)
  
  if isinstance(data, list):  # Check if data is a list of objects
      return len(data)
  else:
      raise ValueError("The JSON file does not contain a list of objects.")