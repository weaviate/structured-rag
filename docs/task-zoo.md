# StructuredRAG Task Zoo

### GenerateAnswer

```python
class GenerateAnswer(BaseModel):
  answer: str
```

### RateContext

```python
class RateContext(BaseModel):
  context_score: int
```

### ParaphraseQuestions

```python
class ParaphraseQuestions(BaseModel):
  quesetions: list[str]
```

### RAGAS

```python
class RAGASmetrics(BaseModel):
  faithfulness_score: float
  answer_relevance_score: float
  context_relevance_score: float
```

### AnswerWithConfidence

```python
class AnswerWithConfidence(BaseModel):
  answer: str
  confidence: float
```

### AnswersWithConfidences

```python
class AnswersWithConfidences(BaseModel):
  answers_with_confidences: list[AnswerWithConfidence]
```
