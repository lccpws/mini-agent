from mini_agent.planner.models import Task
from mini_agent.reflection.models import ReflectionResult


class Corrector:
    def correct(self, task: Task, reflection_result: ReflectionResult) -> Task:
        if reflection_result.improved_input:
            task.input = reflection_result.improved_input

        if reflection_result.suggested_capability:
            task.capability = reflection_result.suggested_capability

        return task
