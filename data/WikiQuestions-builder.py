import dspy
import json

api_key = "sk-foobar"

claude = dspy.Claude(model="claude-3-5-sonnet-20240620", api_key=api_key)
dspy.settings.configure(lm=claude)

from pydantic import BaseModel, validator

class Answer(BaseModel):
    answer: str

    @validator("answer")
    def validate_answer(cls, v):
        if v is None or v == "":
            raise ValueError("Answer cannot be empty")
        if v.strip().lower().startswith("answer:") or v.strip().lower().startswith("context"):
            raise ValueError("Answer should not start with 'Answer:' or 'Context'")
        return v

class GenerateAnswer(dspy.Signature):
    """Assess the context and answer the question."""

    context: str = dspy.InputField(description="The context to use for answering the question.")
    question: str = dspy.InputField(description="The question to answer.")
    answer: Answer = dspy.OutputField(description="The answer to the question. ONLY OUTPUT THE ANSWER AND NOTHING ELSE!!")

generate_answer = dspy.TypedPredictor(GenerateAnswer)

#rag(context="foo", question="bar").answer

with open("./WikiQuestions.json", 'r') as json_file:
    data = json.load(json_file)

print(data[0])

# Rename the "answer" column to "llama_3_1_8b_instruct_answer"
for item in data:
    item["llama_3_1_8b_instruct_answer"] = item.pop("answer")

print(data[0])

for item in data:
    context = item["context"]
    question = item["question"]
    answerable = item["answerable"]
    claude_sonnet_answer_obj = generate_answer(context=context, question=question).answer
    claude_sonnet_answer = claude_sonnet_answer_obj.answer
    print(f"\033[94m{question}\n\033[0m")
    print(f"\033[93m{answerable}\n\033[0m")
    print(f"\033[92m{claude_sonnet_answer}\n\033[0m")
    item["claude_sonnet_answer"] = claude_sonnet_answer

with open("./WikiQuestions-2.0.json", 'w') as json_file:
    json.dump(data, json_file, indent=4)
