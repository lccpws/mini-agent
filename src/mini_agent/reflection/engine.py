from typing import Any
from mini_agent.planner.models import Task
from mini_agent.reflection.models import EvaluationResult, ReflectionResult, ReflectionRecord
from mini_agent.reflection.evaluator import Evaluator
from mini_agent.reflection.reflection import Reflection
from mini_agent.reflection.memory import ReflectionMemory
from mini_agent.reflection.corrector import Corrector


class ReflectionEngine:
    def __init__(
        self,
        llm=None,
        threshold: float = 60.0,
        persist_dir: str = "memory_data",
        min_improvement: float = 3.0,
    ):
        self.evaluator = Evaluator(llm=llm, threshold=threshold)
        self.reflection = Reflection(llm=llm)
        self.memory = ReflectionMemory(persist_dir=persist_dir)
        self.corrector = Corrector()
        self.min_improvement = min_improvement
        self.score_history: dict[str, list[float]] = {}

    def evaluate_and_reflect(
        self,
        task: Task,
        result: Any,
        context: str = ""
    ) -> tuple[EvaluationResult, ReflectionResult]:
        evaluation = self.evaluator.evaluate(task, result, context)

        if task.id not in self.score_history:
            self.score_history[task.id] = []
        self.score_history[task.id].append(evaluation.score)

        reflection_result = ReflectionResult()

        if not evaluation.passed and task.should_reflect():
            reflection_result = self.reflection.reflect(task, result, evaluation)

            if reflection_result.reflected:
                record = ReflectionRecord(
                    id="",
                    capability=task.capability or "",
                    failure_type="UNKNOWN",
                    error_message=str(result) if not isinstance(result, dict) else result.get("error", ""),
                    root_cause=reflection_result.root_cause,
                    suggestions=evaluation.suggestions,
                    alternative_capability=reflection_result.suggested_capability,
                    alternative_input=reflection_result.improved_input,
                )
                self.memory.save(record)

        return evaluation, reflection_result

    def apply_historical_experience(self, task: Task) -> Task:
        if not task.capability:
            return task

        records = self.memory.search_by_capability(task.capability)
        if not records:
            return task

        most_successful = self.memory.get_most_successful(task.capability, limit=1)
        if most_successful:
            task = self.memory.apply_experience(task, most_successful[0])

        return task

    def should_retry(
        self,
        task: Task,
        evaluation: EvaluationResult,
        reflection_result: ReflectionResult
    ) -> bool:
        if evaluation.passed:
            return False

        if not (reflection_result.reflected and reflection_result.should_retry):
            return False

        if task.retry_count >= task.max_retry:
            return False

        scores = self.score_history.get(task.id, [])
        if len(scores) >= 2:
            improvement = scores[-1] - scores[-2]
            if improvement < self.min_improvement:
                return False

        return True

    def apply_reflection(
        self,
        task: Task,
        reflection_result: ReflectionResult
    ) -> Task:
        return self.corrector.correct(task, reflection_result)
