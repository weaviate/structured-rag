# Set the save directory
SAVE_DIR="batch-9-13-24"

# List of all test types
TESTS=("GenerateAnswer" "RateContext" "AssessAnswerability" "ParaphraseQuestions" "RAGAS" "GenerateAnswerWithConfidence" "GenerateAnswersWithConfidence")

# Function to run tests for a specific model
run_tests_for_model() {
    for test in "${TESTS[@]}"
    do
        echo "Running $test"
        if [ -z "$api_key" ]; then
            python3 run_batch_test.py --test "$test" --save-dir "$SAVE_DIR"
        else
            python3 run_batch_test.py --test "$test" --save-dir "$SAVE_DIR"
        fi
        echo "-----------------------"
    done
}

run_tests_for_model