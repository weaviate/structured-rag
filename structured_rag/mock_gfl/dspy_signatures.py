import dspy
from typing import List

class GenerateResponse(dspy.Signature):
    """Follow the task_instructions (Input Field) and generate the response (Output Field) according to the output format given by response_format (Input Field). You will be given references from (Task-Specific Input Field)."""

    task_instructions = dspy.InputField(desc="(Input Field)")
    response_format = dspy.InputField(desc="(Input Field)")
    references = dspy.InputField(desc="Task-Specific Input Field")
    response = dspy.OutputField(desc="(Output Field)")

# ToDo, OPRO_JSON is derived from a compiled version of GenerateResponse
# -- would load the optimized program from disk in `dspy_program.py`

class OPRO_JSON(dspy.Signature):
    """Carefully interpret the task_instructions provided in the Input Field, synthesizing the necessary information from the Task-Specific Input Field to construct a response. Your response should be formatted exclusively in JSON and must conform precisely to the structure dictated by the response_format Input Field. Ensure that your JSON-formatted response is devoid of extraneous characters or elements, such as markdown code block ticks (```), and includes only the keys specified by the response_format. Your attention to detail in following these instructions is paramount for the accuracy and relevance of your output."""

    task_instructions = dspy.InputField(desc="(Input Field)")
    response_format = dspy.InputField(desc="(Input Field)")
    references = dspy.InputField(desc="Task-Specific Input Field")
    response = dspy.OutputField(desc="(Output Field)")