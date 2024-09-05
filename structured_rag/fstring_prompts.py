from typing import Dict

def get_prompt(test: str, references: Dict[str, str], test_params: Dict[str, str]) -> str:
    references_str = ' | '.join(f"{k}: {v}" for k, v in references.items())

    return f"""Instructions: {test_params['task_instructions']}
    References: {references_str}
    Output the result as a JSON string with the following format: {test_params['response_format']}
    IMPORTANT!! Do not start the JSON with ```json or end it with ```."""