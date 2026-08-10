import time
from typing import Any, Callable
from mini_agent.planner.models import Task, FailureType
from mini_agent.executor.failure import FailureClassifier


class RetryManager:
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def should_retry(self, task: Task, failure_type: str) -> bool:
        if failure_type == FailureType.PERMANENT:
            return False

        if task.retry_count >= task.max_retry:
            return False

        return True

    def get_delay(self, retry_count: int) -> float:
        return self.base_delay * (2 ** retry_count)

    def execute_with_retry(
        self,
        func: Callable,
        task: Task,
        failure_type: str = None
    ) -> tuple[bool, Any, str]:
        for attempt in range(task.max_retry):
            try:
                result = func()
                return True, result, None

            except Exception as e:
                failure_type = FailureClassifier.classify(e)
                task.retry_count += 1
                task.error = str(e)

                if not self.should_retry(task, failure_type):
                    return False, str(e), failure_type

                if attempt < task.max_retry - 1:
                    delay = self.get_delay(attempt)
                    time.sleep(delay)

        return False, "Max retries exceeded", FailureType.TRANSIENT
