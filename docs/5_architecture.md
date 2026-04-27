# Architecture

StructuredRAG uses hexagonal (ports & adapters) architecture to cleanly separate domain logic from infrastructure concerns.

## Directory Structure

```
structured_rag/
  core/                          # No external API dependencies
    domain/
      models.py                  # Pydantic output models, Experiment, SingleTestResult
      tasks.py                   # Task definitions (test_params, output model mappings)
      metrics.py                 # JSON validation, scoring functions
    ports/
      llm.py                     # LLMPort abstract interface
      prompting.py               # PromptingStrategy abstract interface
    services/
      runner.py                  # ExperimentRunner -- loops dataset, calls strategy, aggregates
      results.py                 # Save/load experiments, dataset loading
  adapters/                      # External API integrations
    llm/
      openai_adapter.py          # OpenAI implementation of LLMPort
      anthropic_adapter.py       # Anthropic implementation
      google_adapter.py          # Google Gemini implementation
      ollama_adapter.py          # Ollama local + cloud implementations
      modal_vllm_adapter.py      # Modal/vLLM endpoint implementation
      registry.py                # Factory: get_llm_adapter(provider, ...) -> LLMPort
      modal_deployment/          # Modal infrastructure files (deploy separately)
    prompting/
      fstring_strategy.py        # f-string prompting via LLMPort
      dspy_strategy.py           # DSPy prompting (manages its own LLM connection)
  configs/
    benchmark.yaml               # Single config file for all settings
  scripts/
    run_benchmark.py             # CLI entry point
```

## Layers

### Core Domain (`core/domain/`)

Pure Python with zero external API dependencies. Contains:

- **Output models** -- Pydantic `BaseModel` subclasses defining the expected JSON schema for each task (e.g. `GenerateAnswer`, `RateContext`)
- **Task definitions** -- maps task names to their instructions, expected format strings, and output models
- **Metrics** -- `is_valid_json_output()` validates LLM responses against the expected schema; `assess_answerability_metric()` scores correctness

### Ports (`core/ports/`)

Abstract interfaces that define the contracts between the core and the outside world:

- **`LLMPort`** -- `generate(prompt, output_model?) -> str` and `test_connection() -> str`
- **`PromptingStrategy`** -- `run(task_name, task_params, output_model?, context, question, answer) -> str`

### Services (`core/services/`)

Orchestration logic that depends only on ports, not on any specific adapter:

- **`ExperimentRunner`** -- takes a `PromptingStrategy`, loops over the dataset, validates outputs, and aggregates results into an `Experiment`
- **`results.py`** -- saves experiment JSON files and appends to the experiment log

### Adapters (`adapters/`)

Concrete implementations of the ports:

- **LLM adapters** -- one per provider, each implementing `LLMPort` with provider-specific API calls
- **Prompting adapters** -- `FStringStrategy` composes with any `LLMPort`; `DSpyStrategy` manages its own LLM connection internally (DSPy has its own LM configuration system)
- **Registry** -- `get_llm_adapter()` factory that creates the right adapter from a provider string

## Data Flow

```
benchmark.yaml
      |
      v
run_benchmark.py       # reads config, wires everything together
      |
      +---> get_llm_adapter(provider, model, api_key)  --> LLMPort
      |
      +---> FStringStrategy(llm) or DSpyStrategy(...)  --> PromptingStrategy
      |
      +---> ExperimentRunner(strategy, dataset, task)
                  |
                  +---> strategy.run(task, params, ...)    # calls LLM
                  +---> is_valid_json_output(response)     # validates
                  +---> aggregate results --> Experiment
                  |
                  v
            save_experiment() --> results/{task}-{model}-{strategy}-{date}.json
```

## Adding a New Provider

See [Providers > Adding a New Provider](./3_providers.md#adding-a-new-provider).

## Design Decisions

**Why does DSpyStrategy not use LLMPort?** DSPy manages its own LLM connections through `dspy.settings.configure(lm=...)`. Wrapping this behind `LLMPort` would mean fighting DSPy's design. Instead, `DSpyStrategy` implements `PromptingStrategy` directly and configures DSPy internally.

**Why one config file instead of per-provider configs?** The provider and model are just two fields in the config. Separate files per provider created unnecessary friction when the only difference was 2 lines.

**Why YAML over CLI args?** Config files are version-controllable, shareable, and self-documenting. They also handle provider-specific options (like `use_outlines` for Modal/vLLM) cleanly as extra fields.
