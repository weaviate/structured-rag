# Temporary Solution for RAGASmetrics test
import json

with open("./WikiQuestions-2.0.json", 'r') as json_file:
    data = json.load(json_file)

for item in data:
    item["answer"] = item["llama_3_1_8b_instruct_answer"]

with open("./WikiQuestions-2.1.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)
