# Prompting Strategies

StructuredRAG compares different approaches for getting LLMs to produce valid JSON. Each strategy is implemented behind the `PromptingStrategy` interface.

## Available Strategies

### `fstring` -- f-String Prompting

The simplest approach. Builds a prompt with inline instructions telling the model to output JSON in a specific format:

```
Instructions: {task_instructions}
References: {references}
Output the result as a JSON string with the following format: {response_format}
IMPORTANT!! Do not start the JSON with ```json or end it with ```.
```

This is the baseline strategy. Works with any provider.

### `fstring_structured` -- f-String with Provider Structured Outputs

Same prompt as `fstring`, but additionally uses the provider's native structured output API:

- **OpenAI**: `response_format` parameter with Pydantic schema
- **Google**: `response_mime_type="application/json"` with `response_schema`

Not all providers support this. Ollama and Anthropic will behave the same as plain `fstring`.

### `dspy` -- DSPy Follow-the-Format

Uses DSPy's `ChainOfThought` predictor with the `GenerateResponse` signature. DSPy adds a structured prompt template that separates task instructions, response format, and references into labeled fields, plus a reasoning step before the final answer.

This strategy manages its own LLM connection through DSPy's configuration system.

### `dspy_opro` -- DSPy with OPRO-Optimized Signature

Uses DSPy's `Predict` predictor (no chain-of-thought) with an `OPRO_JSON` signature. This signature was discovered through automated prompt optimization (OPRO) and contains detailed instructions about JSON compliance:

> "Ensure that your JSON-formatted response is devoid of extraneous characters or elements, such as markdown code block ticks, and includes only the keys specified by the response_format..."

In the original paper, this achieved **100% JSON success rate** on Llama 3 8B-instruct for the hardest task (GenerateAnswersWithConfidence), where the base DSPy prompt only got 25%.

**Key difference from `dspy`:** trades chain-of-thought reasoning for a heavily optimized system prompt focused on format compliance. Best for maximizing JSON validity on smaller models.

## Choosing a Strategy

| Goal | Recommended strategy |
|---|---|
| Quick baseline test | `fstring` |
| Maximize JSON validity (OpenAI/Google) | `fstring_structured` |
| Test DSPy prompting | `dspy` |
| Maximize JSON validity (any provider) | `dspy_opro` |
| Full comparison | `all` |

## Running Multiple Strategies

Set `strategy: all` in your config to run all 4 strategies sequentially and compare results:

```yaml
strategy: all
```

Each strategy's results are saved as separate JSON files in the `save_dir`.
