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
class ToolArgument(BaseModel):
    argument_name: str
    argument_value: str

class ToolCall(BaseModel):
    function_name: str
    arguments: list[ToolArgument]

class ResponseOrToolCalls(BaseModel):
    reflection_about_tool_use: str = Field(
        default=None,
        description="A rationale regarding whether the tool calls are needed to answer the question."
    )
    use_tools: bool = Field()
    response: str = Field(
        default=None,
        description="A direct response from the LLM without calling any tools."
    )
    tool_calls: List[ToolCall] = Field(
        default=None,
        description="A list of tool calls requested by the LLM."
    )
```
