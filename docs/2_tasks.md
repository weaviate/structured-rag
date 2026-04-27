# Tasks

StructuredRAG tests LLMs across 7 structured output tasks of increasing JSON complexity. Each task requires the model to return a specific JSON schema given a question and context from the WikiQuestions dataset.

## Task Overview

| Task | Output Type | Complexity |
|---|---|---|
| GenerateAnswer | `string` | Simple |
| RateContext | `integer` | Simple |
| AssessAnswerability | `boolean` | Simple |
| ParaphraseQuestions | `List[string]` | List |
| GenerateAnswerWithConfidence | `composite` | Composite object |
| GenerateAnswersWithConfidence | `List[composite]` | List of composite objects |
| RAGAS | `multi-float` | Composite object |

## Task Details

### GenerateAnswer

Answer a question given context. Return a single string.

```json
{"answer": "The National Gallery of Art, Washington D.C., and the Pinacoteca di Brera, Milan, Italy."}
```

### RateContext

Rate how well the context helps answer the question on a 0-5 integer scale.

```json
{"context_score": 5}
```

### AssessAnswerability

Determine if the question is answerable from the context. Returns a boolean. This is the only task with ground truth labels, so it reports both format success rate and task accuracy.

```json
{"answerable_question": true}
```

### ParaphraseQuestions

Generate 3 paraphrased versions of the question. Returns a list of strings.

```json
{"paraphrased_questions": ["Where can some of Vincenzo Civerchio's works be found?", "Where are some pieces by Vincenzo Civerchio displayed?", "Where can I find some of Vincenzo Civerchio's art?"]}
```

### GenerateAnswerWithConfidence

Answer the question and provide an integer confidence score (0-5). A composite object with two fields.

```json
{"answer": "The National Gallery of Art, Washington D.C.", "confidence": 5}
```

### GenerateAnswersWithConfidence

Generate multiple answers, each with a confidence score. A list of composite objects -- the most complex output type.

```json
[{"answer": "National Gallery of Art, Washington D.C.", "confidence": 5}, {"answer": "Pinacoteca di Brera, Milan, Italy", "confidence": 4}]
```

### RAGAS

Evaluate faithfulness, answer relevance, and context relevance as float scores (0-5). A composite object with three float fields.

```json
{"faithfulness_score": 2.5, "answer_relevance_score": 1.0, "context_relevance_score": 3.5}
```

## Running Specific Tasks

In `benchmark.yaml`, set `tasks` to a list of task names:

```yaml
tasks:
  - GenerateAnswer
  - AssessAnswerability
```

Or run all tasks:

```yaml
tasks: all
```

## Pydantic Models

All task output schemas are defined as Pydantic models in `structured_rag/core/domain/models.py`. The validation logic that checks LLM outputs against these schemas lives in `structured_rag/core/domain/metrics.py`.
