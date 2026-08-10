from mini_agent.planner.models import Task, TaskStatus


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

    def update_task_status(self):
        for task in self.tasks.values():
            if task.status in [TaskStatus.PENDING, TaskStatus.READY]:
                dependencies = self.get_dependencies(task.id)

                if any(dep.status == TaskStatus.FAILED for dep in dependencies):
                    task.status = TaskStatus.BLOCKED
                    continue

                if all(dep.status == TaskStatus.SUCCESS for dep in dependencies):
                    task.status = TaskStatus.READY

    def get_ready_tasks(self):
        self.update_task_status()
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.READY
        ]

    def all_completed(self):
        return all(
            task.status == TaskStatus.SUCCESS
            for task in self.tasks.values()
        )

    def has_failed(self):
        return any(
            task.status == TaskStatus.FAILED
            for task in self.tasks.values()
        )

    def get_failed_tasks(self):
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.FAILED
        ]

    def get_retryable_tasks(self):
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.FAILED
            and task.retry_count < task.max_retry
        ]

    def get_blocked_tasks(self):
        return [
            task for task in self.tasks.values()
            if task.status == TaskStatus.BLOCKED
        ]

    def has_blocked(self):
        return any(
            task.status == TaskStatus.BLOCKED
            for task in self.tasks.values()
        )
