model_name="gpt-4o"
model_provider="openai"
api_key="sk-foobar"
test="GenerateAnswersWithConfidence"
save_dir="9-8-24"

python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --api_key "$api_key" --test "$test" --save-dir "$save_dir"