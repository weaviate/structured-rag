# Count input and output tokens per task from the `all_responses` key
import json
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4o-mini")

print(len(encoding.encode("tiktoken is great!")))