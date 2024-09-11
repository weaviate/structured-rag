model_name="claude-3-5-sonnet-20240620"
model_provider="anthropic"
api_key="sk-foobar"
test="RAGAS"
save_dir="9-9-24"

# ToDo:
# # Load environment variables from a .env file
# if [ -f .env ]; then
#   export $(cat .env | xargs)
# fi

python3 run_test.py --model_name "$model_name" --model_provider "$model_provider" --api_key "$api_key" --test "$test" --save-dir "$save_dir"