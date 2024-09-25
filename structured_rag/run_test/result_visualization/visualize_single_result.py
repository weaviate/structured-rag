# This code reads one result json file and visualizes it.

import json
import os
import argparse
import matplotlib.pyplot as plt

from models import Experiment

def visualize_single_result(result_file):
    with open(result_file, "r") as f:
        data = json.load(f)
    experiment = Experiment(**data)
    print(experiment)
    return experiment

def pretty_print_experiment(experiment):
    print(f"Test: {experiment.test_name}")
    print(f"Model: {experiment.model_name}")
    print(f"Prompting Method: {experiment.prompting_method}")
    print(f"Number of Successes: {experiment.num_successes}")
    print(f"Number of Attempts: {experiment.num_attempts}")
    print(f"Success Rate: {experiment.num_successes/experiment.num_attempts:.2%}")
    print(f"Total Time: {experiment.total_time}")

# The data should be in the format of a list of dictionaries, where each dictionary represents a single experiment.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a single result JSON file")
    parser.add_argument("--result-file", type=str, required=True, help="Path to the result JSON file")
    args = parser.parse_args()
    visualize_single_result(args.result_file)