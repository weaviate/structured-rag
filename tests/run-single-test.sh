model_name="gemini-1.5-pro" # claude-3-5-sonnet-20240620
model_provider="google"
api_key="AI-foobar"
test="AssessAnswerability"
save_dir="rerun-gemini-tests"

# ToDo:
# # Load environment variables from a .env file
# if [ -f .env ]; then
#   export $(cat .env | xargs)
# fi

python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --api_key "$api_key" --test "$test" --save-dir "$save_dir"