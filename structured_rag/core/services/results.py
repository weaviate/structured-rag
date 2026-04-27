import json
import os
import datetime

from structured_rag.core.domain.models import Experiment


def save_experiment(experiment: Experiment, save_dir: str, config_name: str) -> str:
    """Save experiment results to a JSON file. Returns the path of the saved file."""
    os.makedirs(save_dir, exist_ok=True)
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    filename = f"{experiment.test_name}-{experiment.model_name}-{config_name}-{current_date}.json"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        json.dump(experiment.dict(), f, indent=2)

    print(f"Results saved to {filepath}")
    return filepath


def append_to_experiment_log(experiment: Experiment, config_type: str, log_path: str = "experiment-log.md"):
    """Append a one-line summary to the experiment log."""
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    with open(log_path, "a") as f:
        f.write(
            f"| {experiment.model_name} | {experiment.success_rate:.2%} "
            f"| {experiment.test_name} | {config_type} | {current_date} |\n"
        )


def load_json_dataset(filepath: str) -> list[dict]:
    """Load a JSON dataset file (e.g. WikiQuestions.json)."""
    with open(filepath, 'r') as f:
        return json.load(f)
