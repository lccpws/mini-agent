from mini_agent.planner.models import Task


class TaskGraph:
    def __init__(self, tasks=None):
        self.tasks: dict[str, Task] = {}
        if tasks:
            for task in tasks:
                self.add_task(task)

    def add_task(self, task: Task):
        if task.id in self.tasks:
            raise ValueError(f"Duplicate task id: {task.id}")
        self.tasks[task.id] = task

    def get_task(self, task_id: str):
        return self.tasks.get(task_id)

    def get_dependencies(self, task_id: str):
        task = self.get_task(task_id)
        if not task:
            raise ValueError(f"Task not found: {task_id}")
        return [
            self.tasks[dependency_id]
            for dependency_id in task.dependencies
        ]

    def get_dependents(self, task_id: str):
        return [
            task
            for task in self.tasks.values()
            if task_id in task.dependencies
        ]

    def get_ready_tasks(self):
        ready = []
        for task in self.tasks.values():
            if task.status != "PENDING":
                continue
            dependencies = self.get_dependencies(task.id)
            if all(
                dependency.status == "COMPLETED"
                for dependency in dependencies
            ):
                ready.append(task)
        return ready

    def all_completed(self):
        return all(task.status == "COMPLETED" for task in self.tasks.values())

    def has_failed(self):
        return any(task.status == "FAILED" for task in self.tasks.values())

    def get_failed_tasks(self):
        return [task for task in self.tasks.values() if task.status == "FAILED"]
