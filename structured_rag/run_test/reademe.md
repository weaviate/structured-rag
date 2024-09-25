# StructuredRAG Tests

StructuredRAG contains 6 tests for JSON Structured Output testing with LLMs.

- `GenerateAnswer` -> str
- `RateContext` -> int
- `AssessAnswerability` -> bool
- `ParaphraseQuestions` -> List[str]
- `AnswerWithConfidence` -> AnswerWithConfidence
- `AnswersWithConfidences` -> List[AnswerWithConfidence]

```python
from pydantic import BaseModel

class AnswerWithConfidence(BaseModel):
    answer: str
    confidence: float
```