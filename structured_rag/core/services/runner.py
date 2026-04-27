import time
from typing import Optional

from pydantic import BaseModel

from structured_rag.core.domain.models import (
    Experiment, PromptWithResponse, PromptingMethod, SingleTestResult,
)
from structured_rag.core.domain.metrics import is_valid_json_output, assess_answerability_metric
from structured_rag.core.ports.prompting import PromptingStrategy


def _run_single_test(
    strategy: PromptingStrategy,
    task_name: str,
    task_params: dict,
    output_model: Optional[type[BaseModel]],
    title: str,
    context: str,
    question: str,
    answer: str,
    ground_truth,
) -> SingleTestResult:
    """Run a single test instance and return the result."""
    prompt_desc = f"Title: {title}\nContext: {context}\nQuestion: {question}"

    try:
        if task_name == "ParaphraseQuestions":
            output = strategy.run(task_name, task_params, output_model, question=question)
        elif task_name == "RAGAS":
            output = strategy.run(task_name, task_params, output_model, context, question, answer)
        else:
            output = strategy.run(task_name, task_params, output_model, context, question)

        parsed_output, is_valid = is_valid_json_output(output, task_name)

        task_metric = 0
        if is_valid and task_name == "AssessAnswerability":
            task_metric = assess_answerability_metric(parsed_output, ground_truth)

        return SingleTestResult(
            prompt_with_response=PromptWithResponse(prompt=prompt_desc, response=output),
            is_valid=is_valid,
            task_metric=task_metric,
        )

    except Exception as e:
        print(f"Error: {e}")
        return SingleTestResult(
            prompt_with_response=PromptWithResponse(prompt=prompt_desc, response="Error"),
            is_valid=False,
            task_metric=0,
        )


class ExperimentRunner:
    """Runs a full experiment: loops over dataset, calls strategy, aggregates results."""

    def __init__(
        self,
        strategy: PromptingStrategy,
        dataset: list[dict],
        task_name: str,
        task_params: dict,
        output_model: Optional[type[BaseModel]],
        model_name: str,
        prompting_method: PromptingMethod,
    ):
        self.strategy = strategy
        self.dataset = dataset
        self.task_name = task_name
        self.task_params = task_params
        self.output_model = output_model
        self.model_name = model_name
        self.prompting_method = prompting_method

    def run(self) -> Experiment:
        experiment = Experiment(
            test_name=self.task_name,
            model_name=self.model_name,
            prompting_method=self.prompting_method,
            num_successes=0,
            total_task_performance=0,
            num_attempts=0,
            success_rate=0,
            average_task_performance=0,
            total_time=0,
            all_responses=[],
            failed_responses=[],
        )

        start_time = time.time()

        for entry in self.dataset:
            title = entry.get('title', '')
            context = entry.get('context', '')
            question = entry.get('question', '')
            answer = entry.get('answer', '')
            answerable = entry.get('answerable', '')

            print(f"--- {title}: {question}")

            result = _run_single_test(
                strategy=self.strategy,
                task_name=self.task_name,
                task_params=self.task_params,
                output_model=self.output_model,
                title=title,
                context=context,
                question=question,
                answer=answer,
                ground_truth=answerable,
            )

            experiment.all_responses.append(result.prompt_with_response)
            experiment.num_attempts += 1
            if result.is_valid:
                experiment.num_successes += 1
                experiment.total_task_performance += result.task_metric
            else:
                experiment.failed_responses.append(result.prompt_with_response)

        experiment.total_time = int(time.time() - start_time)

        if experiment.num_attempts > 0:
            experiment.success_rate = experiment.num_successes / experiment.num_attempts
            experiment.average_task_performance = experiment.total_task_performance / experiment.num_attempts

        print(f"\nJSON Format Success Rate: {experiment.num_successes}/{experiment.num_attempts} ({experiment.success_rate:.2%})")
        print(f"Task Accuracy (correctness): {experiment.average_task_performance:.2%}")

        return experiment
