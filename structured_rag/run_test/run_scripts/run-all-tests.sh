#!/bin/bash

# Set your API key here
GOOGLE_API_KEY="AI..."

# Set the save directory
SAVE_DIR="9-8-24"

# List of all test types
TESTS=("GenerateAnswer" "RateContext" "AssessAnswerability" "ParaphraseQuestions" "RAGAS" "GenerateAnswerWithConfidence" "GenerateAnswersWithConfidence")

# Function to run tests for a specific model
run_tests_for_model() {
    local model_name=$1
    local model_provider=$2
    local api_key=$3

    for test in "${TESTS[@]}"
    do
        echo "Running $test for $model_name"
        if [ -z "$api_key" ]; then
            python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --test "$test" --save-dir "$SAVE_DIR"
        else
            python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --api_key "$api_key" --test "$test" --save-dir "$SAVE_DIR"
        fi
        echo "-----------------------"
    done
}

# Run tests for Llama3
# Need to clean this up, this is actually the only "model directory" in the repo.
run_tests_for_model "llama3:instruct" "ollama" ""

# claude-3-5-sonnet-20240620 gpt-4o

# Run tests for Gemini
run_tests_for_model "gemini-1.5-pro" "google" "$GOOGLE_API_KEY"

python3 aggregate_jsons.py

echo "All tests completed!"
