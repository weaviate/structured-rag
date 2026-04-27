"""
StructuredRAG Benchmark Runner
==============================

Usage:
  python -m structured_rag.scripts.run_benchmark openai
  python -m structured_rag.scripts.run_benchmark anthropic
  python -m structured_rag.scripts.run_benchmark full_sweep
"""

import sys
import os

import yaml


CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")

from structured_rag.core.domain.models import PromptingMethod
from structured_rag.core.domain.tasks import test_params, test_to_output_model, ALL_TASKS
from structured_rag.adapters.llm.registry import get_llm_adapter
from structured_rag.adapters.prompting.fstring_strategy import FStringStrategy
from structured_rag.adapters.prompting.dspy_strategy import DSpyStrategy
from structured_rag.core.services.runner import ExperimentRunner
from structured_rag.core.services.results import save_experiment, append_to_experiment_log, load_json_dataset


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "WikiQuestions.json")


def _resolve_api_key(cfg: dict) -> str:
    """Resolve API key: use api_key field directly, or read from api_key_env env var."""
    if cfg.get("api_key"):
        return cfg["api_key"]
    env_var = cfg.get("api_key_env", "STRUCTURED_RAG_API_KEY")
    key = os.environ.get(env_var, "")
    if not key:
        print(f"Warning: no API key found (checked config field 'api_key' and env var '{env_var}')")
    return key


_TOP_LEVEL_KEYS = {"provider", "model", "api_key", "api_key_env", "strategy", "tasks", "save_dir", "data_path"}


def _adapter_kwargs(cfg: dict) -> dict:
    """Extract provider-specific kwargs (anything not a top-level config key)."""
    return {k: v for k, v in cfg.items() if k not in _TOP_LEVEL_KEYS}


def _build_strategies(cfg: dict) -> list[tuple[str, object, PromptingMethod]]:
    """Return list of (config_name, strategy, prompting_method) tuples to run."""
    provider = cfg["provider"]
    model = cfg["model"]
    api_key = _resolve_api_key(cfg)
    strategy = cfg.get("strategy", "fstring")
    extra = _adapter_kwargs(cfg)
    configs = []

    if strategy in ("fstring", "all"):
        llm = get_llm_adapter(provider, model, api_key, structured_outputs=False, **extra)
        print("Running LLM connection test...")
        print(llm.test_connection())
        configs.append(("fstring", FStringStrategy(llm), PromptingMethod.fstring))

    if strategy in ("fstring_structured", "all"):
        llm_structured = get_llm_adapter(provider, model, api_key, structured_outputs=True, **extra)
        if strategy != "all":
            print("Running LLM connection test...")
            print(llm_structured.test_connection())
        configs.append(("fstring_structured", FStringStrategy(llm_structured), PromptingMethod.fstring))

    if strategy in ("dspy", "all"):
        configs.append((
            "dspy",
            DSpyStrategy(model, provider, api_key, use_opro=False),
            PromptingMethod.dspy,
        ))

    if strategy in ("dspy_opro", "all"):
        configs.append((
            "dspy_opro",
            DSpyStrategy(model, provider, api_key, use_opro=True),
            PromptingMethod.dspy,
        ))

    if not configs:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Choose from: fstring, fstring_structured, dspy, dspy_opro, all"
        )

    return configs


def _resolve_config_path(arg: str) -> str:
    """Resolve a config name or path to an absolute file path.

    Accepts:
      - A bare name like 'openai' → looks up structured_rag/configs/openai.yaml
      - A full/relative path like 'my_configs/custom.yaml' → used as-is
    """
    # If it looks like a direct path (has a slash or ends in .yaml/.yml), use it directly
    if os.sep in arg or arg.endswith((".yaml", ".yml")):
        return arg

    # Otherwise treat it as a config name and look in the package configs dir
    candidate = os.path.join(CONFIGS_DIR, f"{arg}.yaml")
    if os.path.exists(candidate):
        return candidate

    # Try .yml as fallback
    candidate_yml = os.path.join(CONFIGS_DIR, f"{arg}.yml")
    if os.path.exists(candidate_yml):
        return candidate_yml

    # List available configs to help the user
    available = [f.removesuffix(".yaml").removesuffix(".yml")
                 for f in os.listdir(CONFIGS_DIR) if f.endswith((".yaml", ".yml"))]
    raise FileNotFoundError(
        f"Config '{arg}' not found. Available configs: {', '.join(sorted(available))}"
    )


def main():
    if len(sys.argv) < 2:
        available = [f.removesuffix(".yaml").removesuffix(".yml")
                     for f in os.listdir(CONFIGS_DIR) if f.endswith((".yaml", ".yml"))]
        print("Usage: python -m structured_rag.scripts.run_benchmark <config>")
        print(f"\nAvailable configs: {', '.join(sorted(available))}")
        sys.exit(1)

    config_path = _resolve_config_path(sys.argv[1])
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"Loaded config from {config_path}")

    data_path = cfg.get("data_path", DATA_PATH)
    dataset = load_json_dataset(data_path)
    print(f"Loaded {len(dataset)} samples from {data_path}")

    tasks_cfg = cfg.get("tasks", ["AssessAnswerability"])
    if tasks_cfg == "all":
        tasks_to_run = ALL_TASKS
    elif isinstance(tasks_cfg, list):
        tasks_to_run = tasks_cfg
    else:
        tasks_to_run = [tasks_cfg]

    configs = _build_strategies(cfg)
    save_dir = cfg.get("save_dir", "results")

    for task_name in tasks_to_run:
        if task_name not in test_params:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task_p = test_params[task_name]
        output_model = test_to_output_model[task_name]

        for config_name, strategy, prompting_method in configs:
            print(f"\n{'='*60}")
            print(f"Task: {task_name} | Strategy: {config_name} | Model: {cfg['model']}")
            print(f"{'='*60}")

            runner = ExperimentRunner(
                strategy=strategy,
                dataset=dataset,
                task_name=task_name,
                task_params=task_p,
                output_model=output_model,
                model_name=cfg["model"],
                prompting_method=prompting_method,
            )

            experiment = runner.run()

            save_experiment(experiment, save_dir, config_name)
            append_to_experiment_log(experiment, config_name)


if __name__ == "__main__":
    main()
