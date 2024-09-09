from pydantic import BaseModel
from enum import Enum

class PromptWithResponse(BaseModel):
    prompt: str
    response: str

class PromptingMethod(str, Enum):
    dspy = "dspy"
    fstring = "fstring"

# Need to add `success_rate` to the Experiment class

class Experiment(BaseModel):
    test_name: str
    model_name: str
    prompting_method: PromptingMethod
    num_successes: int
    num_attempts: int
    success_rate: float
    total_time: int
    all_responses: list[PromptWithResponse]
    failed_responses: list[PromptWithResponse]