# Run StructuredRAG Test

To run the tests, create the python environment using `poetry install`.

You can then run the script using `poetry run python tests/run_test.py`.

`run_test.py` accepts the following command-line arguments:

- `--model_name`: The name of the model to use.
- `--model_provider`: The provider of the model.
- `--api_key`: The API key for the model provider (not needed for Ollama).
- `--test`: The type of test to run.

StructuredRAG currently supports the following tests:

- `GenerateAnswer` (string)
- `RateContext` (integer)
- `AssessAnswerability` (boolean)
- `ParaphraseQuestions` (list of strings)
- `GenerateAnswerWithConfidence` (AnswerWithConfidence)
- `GenerateAnswersWithConfidence` (list of AnswerWithConfidence)
