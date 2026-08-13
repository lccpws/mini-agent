from dataclasses import dataclass, field
from typing import Any
from jsonschema import validate, ValidationError


class PlanStatus:
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    NEED_REPLAN = "NEED_REPLAN"


class TaskStatus:
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    RETRYING = "RETRYING"
    CANCELLED = "CANCELLED"


class FailureType:
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    UNKNOWN = "UNKNOWN"
    NEED_REPLAN = "NEED_REPLAN"


class TaskSchemas:
    WEATHER = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature": {"type": "number"},
            "weather": {"type": "string"},
            "humidity": {"type": "integer"}
        },
        "required": ["city", "temperature", "weather"]
    }

    SEARCH = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "content": {"type": "string"},
            "source": {"type": "string"}
        },
        "required": ["title", "content"]
    }

    ANALYSIS = {
        "type": "object",
        "properties": {
            "analysis": {"type": "string"},
            "conclusion": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["analysis", "conclusion"]
    }

    COMPANY = {
        "type": "object",
        "properties": {
            "company_name": {"type": "string"},
            "founded_year": {"type": "integer"},
            "business": {"type": "string"},
            "source_urls": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["company_name"]
    }

    RESEARCH_REPORT = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "key_findings": {"type": "array", "items": {"type": "string"}},
            "recommendations": {"type": "array", "items": {"type": "string"}},
            "sources": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "summary"]
    }

    COMPARISON = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "object"}},
            "best_choice": {"type": "string"},
            "reason": {"type": "string"}
        },
        "required": ["best_choice", "reason"]
    }

    _schemas = {
        "weather": WEATHER,
        "search": SEARCH,
        "analysis": ANALYSIS,
        "company": COMPANY,
        "research_report": RESEARCH_REPORT,
        "comparison": COMPARISON,
    }

    @classmethod
    def get(cls, name: str) -> dict[str, Any] | None:
        return cls._schemas.get(name)

    @classmethod
    def list_all(cls) -> list[str]:
        return list(cls._schemas.keys())

    @classmethod
    def register(cls, name: str, schema: dict[str, Any]):
        cls._schemas[name] = schema


SIMPLE_CAPABILITIES = {"weather", "calculator"}


@dataclass
class Task:
    id: str
    description: str
    objective: str
    capability: str | None = None
    dependencies: list[str] = field(default_factory=list)
    input: dict[str, Any] = field(default_factory=dict)
    expected_output: str = ""
    output_schema: dict[str, Any] | str | None = None
    status: str = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    max_retry: int = 3
    enable_reflection: bool | None = None

    def should_reflect(self) -> bool:
        if self.enable_reflection is not None:
            return self.enable_reflection
        return self.capability not in SIMPLE_CAPABILITIES

    def _resolve_schema(self) -> dict[str, Any] | None:
        if self.output_schema is None:
            return None
        if isinstance(self.output_schema, str):
            return TaskSchemas.get(self.output_schema)
        return self.output_schema

    def validate_result(self, result: Any) -> tuple[bool, str]:
        schema = self._resolve_schema()
        if not schema:
            return True, ""

        try:
            validate(instance=result, schema=schema)
            return True, ""
        except ValidationError as e:
            return False, f"Schema validation failed: {e.message}"
        except ImportError:
            return True, ""

    def get_result_field(self, field_name: str) -> Any:
        if isinstance(self.result, dict):
            return self.result.get(field_name)
        return None


@dataclass
class Plan:
    goal: str
    tasks: list[Task] = field(default_factory=list)
    status: str = PlanStatus.CREATED
    reasoning: str = ""
    version: int = 1
    quality_score: int = 100


class PlanQuality:
    def __init__(self, capabilities: list[str] = None):
        self.capabilities = capabilities or []

    def evaluate(self, plan: Plan) -> int:
        score = 100

        if len(plan.tasks) > 15:
            score -= 20

        if self.has_unnecessary_dependencies(plan):
            score -= 10

        if self.has_unknown_capability(plan):
            score -= 30

        if self.has_cycle(plan):
            score -= 50

        return max(score, 0)

    def has_unnecessary_dependencies(self, plan: Plan) -> bool:
        for task in plan.tasks:
            for dep_id in task.dependencies:
                dep_task = next((t for t in plan.tasks if t.id == dep_id), None)
                if dep_task is None:
                    continue
                if task.capability == dep_task.capability:
                    return True
        return False

    def has_unknown_capability(self, plan: Plan) -> bool:
        if not self.capabilities:
            return False
        for task in plan.tasks:
            if task.capability and task.capability not in self.capabilities:
                return True
        return False

    def has_cycle(self, plan: Plan) -> bool:
        graph = {task.id: task.dependencies for task in plan.tasks}
        visited = set()
        visiting = set()

        def dfs(node):
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for neighbor in graph.get(node, []):
                if dfs(neighbor):
                    return True
            visited.add(node)
            visiting.remove(node)
            return False

        for node in graph:
            if dfs(node):
                return True
        return False

    def get_score_breakdown(self, plan: Plan) -> dict:
        breakdown = {
            "total": 100,
            "task_count_penalty": 0,
            "unnecessary_dep_penalty": 0,
            "unknown_capability_penalty": 0,
            "cycle_penalty": 0,
        }

        if len(plan.tasks) > 15:
            breakdown["task_count_penalty"] = 20

        if self.has_unnecessary_dependencies(plan):
            breakdown["unnecessary_dep_penalty"] = 10

        if self.has_unknown_capability(plan):
            breakdown["unknown_capability_penalty"] = 30

        if self.has_cycle(plan):
            breakdown["cycle_penalty"] = 50

        breakdown["final_score"] = self.evaluate(plan)
        return breakdown
