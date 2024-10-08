import modal

vllm_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "vllm==0.5.3post1", "outlines==0.0.46"
)

MODELS_DIR = "/llamas"
# Model IDs
Llama_3_1_8B_Instruct_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
Llama_3_2_1B_Instruct_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
Llama_3_2_3B_Instruct_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# Model Revisions
Llama_3_1_8B_Instruct_MODEL_REVISION = "8c22764a7e3675c50d4c7c9a4edb474456022b16"  # pin model revisions to prevent unexpected changes!
Llama_3_2_1B_Instruct_MODEL_REVISION = "e9f8effbab1cbdc515c11ee6e098e3d5a9f51e14"
Llama_3_2_3B_Instruct_MODEL_REVISION = "392a143b624368100f77a3eafaa4a2468ba50a72"

try:
    volume = modal.Volume.lookup("llamas", create_if_missing=False)
except modal.exception.NotFoundError:
    raise Exception("Download models first with modal run download_llama.py")

# Test this
N_GPUS=1
GPU_CONFIG = modal.gpu.A100(count=N_GPUS)
MINUTES = 60
DTYPE = "float16"
MAX_INPUT_LEN = 2048
MAX_OUTPUT_LEN = 512

app = modal.App("example-vllm-outlines", image=vllm_image)

from pydantic import BaseModel
class Answer(BaseModel):
    answer: str
    confidence_rating: float

@app.cls(
    gpu=GPU_CONFIG, container_idle_timeout=1 * MINUTES, volumes={MODELS_DIR: volume}
)
class Model:
    @modal.enter()
    def load(self):
        """Loads the VLLM engine and configures our tokenizer."""

        from vllm import EngineArgs, LLMEngine, SamplingParams
        from outlines.integrations.vllm import JSONLogitsProcessor
        import vllm

        volume.reload()

        engine_args = EngineArgs(
            model=MODELS_DIR,
            tensor_parallel_size=N_GPUS,
            gpu_memory_utilization=0.9,
            max_model_len=8096,
            enforce_eager=False,
            dtype=DTYPE,
        )

        self.engine = LLMEngine.from_engine_args(engine_args)

    @modal.method(is_generator=True)
    def generate(self, prompts: list[str], settings=None):
        """Generate responses to a batch of prompts, optionally with custom inference settings."""
        from vllm import SamplingParams

        request_id = 0

        # Add all prompts to the engine
        for prompt in prompts:
            sampling_params = SamplingParams(
                max_tokens=MAX_OUTPUT_LEN,
                temperature=0
            )
            self.engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        # Process requests and yield results
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    yield request_output.outputs[0].text

    # TODO: Add the generator back in
    @modal.method()
    def generate_with_outlines(self, prompts: list[str], output_model: BaseModel, settings=None):
        """Generate responses to a batch of prompts using Outlines structured outputs according to the provided Pydantic model."""

        from vllm import SamplingParams
        from outlines.integrations.vllm import JSONLogitsProcessor

        request_id = 0
        results = []

        logits_processor = JSONLogitsProcessor(schema=output_model, llm=self.engine)

        # Add all prompts to the engine
        for prompt in prompts:
            sampling_params = SamplingParams(
                max_tokens=MAX_OUTPUT_LEN,
                temperature=0,
                logits_processors=[logits_processor]
            )
            self.engine.add_request(str(request_id), prompt, sampling_params)
            request_id += 1

        # Process requests and collect results
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            for request_output in request_outputs:
                if request_output.finished:
                    # fix this, `answer` is a terribly confusing key -- `response` is better
                    results.append({
                        "id": request_output.request_id,
                        "answer": request_output.outputs[0].text
                    })

        return results