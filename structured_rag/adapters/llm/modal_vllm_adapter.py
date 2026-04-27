import json
from typing import Optional

import requests
from pydantic import BaseModel

from structured_rag.core.ports.llm import LLMPort


LLAMA3_PROMPT_TEMPLATE = """<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

You are a helpful assistant<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""


class ModalVLLMAdapter(LLMPort):
    """Adapter for a Modal-deployed vLLM + Outlines endpoint.

    Deployment files live in structured_rag/adapters/llm/modal_deployment/.
    Deploy with: modal deploy vllm_outlines_setup.py

    The endpoint accepts POST requests with:
      {"prompts": [...], "with_outlines": bool, "output_model": <json_schema>}
    """

    def __init__(self, endpoint_url: str, api_key: str,
                 use_outlines: bool = True, use_llama3_template: bool = True,
                 timeout: int = 300):
        self.endpoint_url = endpoint_url
        self.use_outlines = use_outlines
        self.use_llama3_template = use_llama3_template
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _wrap_prompt(self, prompt: str) -> str:
        if self.use_llama3_template:
            return LLAMA3_PROMPT_TEMPLATE.format(prompt=prompt)
        return prompt

    def generate(self, prompt: str, output_model: Optional[type[BaseModel]] = None) -> str:
        wrapped = self._wrap_prompt(prompt)

        payload = {
            "prompts": [wrapped],
            "with_outlines": self.use_outlines and output_model is not None,
        }

        if self.use_outlines and output_model is not None:
            payload["output_model"] = output_model.schema()

        response = requests.post(
            self.endpoint_url, headers=self.headers,
            json=payload, timeout=self.timeout,
        )
        response.raise_for_status()

        results = response.json()

        if isinstance(results, list):
            # Outlines mode returns [{"id": "0", "answer": "..."}]
            return results[0]["answer"]
        elif isinstance(results, str):
            # Generator mode returns raw text
            return results
        else:
            return json.dumps(results)

    def generate_batch(self, prompts: list[str],
                       output_model: Optional[type[BaseModel]] = None) -> list[str]:
        """Send a batch of prompts in a single request. Returns responses in order."""
        wrapped = [self._wrap_prompt(p) for p in prompts]

        payload = {
            "prompts": wrapped,
            "with_outlines": self.use_outlines and output_model is not None,
        }

        if self.use_outlines and output_model is not None:
            payload["output_model"] = output_model.schema()

        response = requests.post(
            self.endpoint_url, headers=self.headers,
            json=payload, timeout=self.timeout,
        )
        response.raise_for_status()

        results = response.json()

        if isinstance(results, list) and results and isinstance(results[0], dict):
            # Outlines mode: sort by ID, extract answers
            sorted_results = sorted(results, key=lambda r: int(r["id"]))
            return [r["answer"] for r in sorted_results]
        elif isinstance(results, list):
            return results
        else:
            return [json.dumps(results)]

    def test_connection(self) -> str:
        return self.generate("say hello")
