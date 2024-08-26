import dspy
from typing import List

class GenerateResponse(dspy.Signature):
    """Follow the task_instructions (Input Field) and generate the response (Output Field) according to the output format given by response_format (Input Field). You will be given references from (Task-Specific Input Field)."""

    task_instructions = dspy.InputField(desc="(Input Field)")
    response_format = dspy.InputField(desc="(Input Field)")
    references = dspy.InputField(desc="Task-Specific Input Field")
    response = dspy.OutputField(desc="(Output Field)")