from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any



@dataclass
class ToolInvocation:
    role: str | None = None
    tool_name: str | None = None
    args: Any = None
    state: str = "pending"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any = None
    error: str | None = None

    def start(self):
        self.state = "running"
        self.started_at = datetime.now(timezone.utc)

    def succeed(self, result):
        self.state = "success"
        self.result = result
        self.completed_at = datetime.now(timezone.utc)

    def fail(self, error):
        self.state = "failed"
        self.error = str(error)
        self.completed_at = datetime.now(timezone.utc)

    def to_dict(self):
        return {
            "role": self.role,
            "tool_name": self.tool_name,
            "args": self.args,
            "state": self.state,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
        }

