# StructuredRAG Task Zoo

### GenerateAnswer

```python
class GenerateAnswer(BaseModel):
  answer: str
```

### ParaphraseQuestions

```python
class ParaphraseQuestions(BaseModel):
  questions: list[str]
```

### RateContext

```python
class RateContext(BaseModel):
  context_score: int
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

### ResponseOrToolCall

```python
class ResponseOrToolCalls(BaseModel):
  final_response: bool
  response: str
  tool_calls: list[ToolCall]
```
