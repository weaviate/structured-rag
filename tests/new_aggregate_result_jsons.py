# This code loops from a folder `save-dir` (Cli argument) and aggregates the results from each trial into a single json file.

import json
import os
import argparse

def aggregate_results(save_dir):
    # Loops through all the files in the `save_dir` directory and aggregates the results into a single JSON file.
    results = []
    for filename in os.listdir(save_dir):
        if filename.endswith(".json"):
            with open(os.path.join(save_dir, filename), "r") as f:
                data = json.load(f)
                results.append(data)
                print(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate results from a save directory")
    parser.add_argument("--save-dir", type=str, required=True, help="Directory to save the results")
    args = parser.parse_args()
    aggregate_results(args.save_dir)