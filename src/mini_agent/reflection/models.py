from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Action(str, Enum):
    RETRY = "retry"
    REGENERATE = "regenerate"
    REPLAN = "replan"
    NONE = "none"


@dataclass
class EvaluationResult:
    score: float
    passed: bool
    reason: str
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ReflectionResult:
    reflected: bool = False
    feedback: str = ""
    improved_input: dict = field(default_factory=dict)
    should_retry: bool = False
    action: str = Action.NONE
    root_cause: str = ""
    suggested_capability: str | None = None


@dataclass
class ReflectionRecord:
    id: str
    capability: str
    failure_type: str
    error_message: str
    root_cause: str
    suggestions: list[str] = field(default_factory=list)
    alternative_capability: str | None = None
    alternative_input: dict = field(default_factory=dict)
    success_count: int = 0
    fail_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used: datetime | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "capability": self.capability,
            "failure_type": self.failure_type,
            "error_message": self.error_message,
            "root_cause": self.root_cause,
            "suggestions": self.suggestions,
            "alternative_capability": self.alternative_capability,
            "alternative_input": self.alternative_input,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ReflectionRecord":
        return cls(
            id=data.get("id", ""),
            capability=data.get("capability", ""),
            failure_type=data.get("failure_type", ""),
            error_message=data.get("error_message", ""),
            root_cause=data.get("root_cause", ""),
            suggestions=data.get("suggestions", []),
            alternative_capability=data.get("alternative_capability"),
            alternative_input=data.get("alternative_input", {}),
            success_count=data.get("success_count", 0),
            fail_count=data.get("fail_count", 0),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            last_used=datetime.fromisoformat(data["last_used"]) if data.get("last_used") else None,
        )
