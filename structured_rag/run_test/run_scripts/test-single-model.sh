# Runs all tests for a single model
model_name="gemini-1.5-pro" # claude-3-5-sonnet-20240620
model_provider="google"
api_key="sk-foobar"
save_dir="9-10-24"

# List of all test types
TESTS=("GenerateAnswer" "RateContext" "AssessAnswerability" "ParaphraseQuestions" "RAGAS" "GenerateAnswerWithConfidence" "GenerateAnswersWithConfidence")

# Run all tests for the given model
for test in "${TESTS[@]}"
do
    echo "Running $test for $model_name"
    if [ -z "$api_key" ]; then
        python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --test "$test" --save-dir "$save_dir"
    else
        python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --api_key "$api_key" --test "$test" --save-dir "$save_dir"
    fi
    echo "-----------------------"
done

echo "All tests completed!"
