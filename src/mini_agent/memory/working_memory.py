from typing import Any


class WorkingMemory:
    """当前任务的工作记忆，单次任务结束后清空"""

    def __init__(self):
        self.data = {}

    def add(self, key: str, value: Any):
        self.data[key] = value

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def get_all(self) -> dict:
        return self.data.copy()

    def clear(self):
        self.data = {}

    def __str__(self):
        return str(self.data)
