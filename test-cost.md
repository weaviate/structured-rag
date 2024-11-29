# Cost for running StructuredRAG tests

StructuredRAG contains 112 inputs that slightly vary with the output format per task:

| Task | Input Tokens | Output Tokens |
|------|--------------|---------------|
| `AssessAnswerability` | 14,170 | 784 |

# Test Costs
| Task | Model | Input Cost | Output Cost | Total Cost |
|------|--------|------------|-------------|------------|
| AssessAnswerability | gpt-4o | $0.35 | $0.08 | $0.43 |

## Model costs

### Per 1M Tokens

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
|-------|---------------------------|----------------------------|
| gpt-4o | $2.50 | $10.00 |

### Per 1K Tokens

| Model | Input Cost (per 1K tokens) | Output Cost (per 1K tokens) |
|-------|---------------------------|----------------------------|
| gpt-4o | $0.0025 | $0.01 |

