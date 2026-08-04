from dataclasses import dataclass, field


class PlanStatus:
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEED_REPLAN = "NEED_REPLAN"


class StepStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class PlanStep:
    id: str
    task: str
    dependencies: list[str] = field(default_factory=list)
    status: str = StepStatus.PENDING
    capability: str | None = None
    result: str | None = None

@dataclass
class Plan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    status: str = PlanStatus.CREATED
    reasoning: str = ""
