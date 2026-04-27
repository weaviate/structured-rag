# Getting Started

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- An API key for at least one supported provider (OpenAI, Anthropic, Google, Ollama Cloud, or Modal)

## Installation

```bash
git clone https://github.com/weaviate/structured-rag.git
cd structured-rag
uv sync
```

## Configuration

All benchmark settings live in a single YAML file: `structured_rag/configs/benchmark.yaml`.

```yaml
provider: openai
model: gpt-5.4-nano
api_key_env: OPENAI_API_KEY

strategy: fstring
tasks:
  - AssessAnswerability

save_dir: results
```

### Config fields

| Field | Description | Required |
|---|---|---|
| `provider` | LLM provider to use (see [Providers](./3_providers.md)) | yes |
| `model` | Model name string passed to the provider API | yes |
| `api_key_env` | Name of the environment variable holding your API key | yes (except `ollama`) |
| `api_key` | Hardcoded API key (alternative to `api_key_env`) | no |
| `strategy` | Prompting strategy (see [Strategies](./4_strategies.md)) | no, defaults to `fstring` |
| `tasks` | List of task names, or `"all"` (see [Tasks](./2_tasks.md)) | no, defaults to `AssessAnswerability` |
| `save_dir` | Directory for result JSON files | no, defaults to `results` |
| `data_path` | Path to dataset JSON file | no, defaults to `data/WikiQuestions.json` |

Any additional fields in the YAML are passed through as provider-specific options (e.g. `use_outlines: true` for Modal/vLLM).

## Running the Benchmark

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

Run with the default config:

```bash
uv run python -m structured_rag.scripts.run_benchmark
```

Or point to a custom config file:

```bash
uv run python -m structured_rag.scripts.run_benchmark path/to/custom.yaml
```

## Understanding the Output

The benchmark reports two metrics per experiment:

- **JSON Format Success Rate** -- what percentage of LLM responses were valid, parseable JSON matching the expected schema
- **Task Accuracy** -- for tasks with ground truth labels (e.g. AssessAnswerability), what percentage of answers were correct

Results are saved as JSON files in the `save_dir` directory, one file per task/strategy combination.
