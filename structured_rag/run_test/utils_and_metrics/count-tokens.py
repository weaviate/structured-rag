# Count input and output tokens per task from the `all_responses` key
import json
import tiktoken

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

# Read results file
with open("../results/results/AssessAnswerability-gpt-4o-fstring_with_structured_outputs-2024-11-29.json", "r") as f:
    results = json.load(f)

total_input_tokens = 0
total_output_tokens = 0

# Process each response
num_responses = len(results["all_responses"])
for response in results["all_responses"]:
    # Count input tokens from prompt
    input_tokens = len(encoding.encode(response["prompt"]))
    total_input_tokens += input_tokens
    
    # Count output tokens from response
    output_tokens = len(encoding.encode(response["response"]))
    total_output_tokens += output_tokens

# Calculate averages
avg_input_tokens = total_input_tokens / num_responses
avg_output_tokens = total_output_tokens / num_responses

print(f"Total input tokens: {total_input_tokens}")
print(f"Total output tokens: {total_output_tokens}")
print(f"Average input tokens per response: {avg_input_tokens:.1f}")
print(f"Average output tokens per response: {avg_output_tokens:.1f}")
