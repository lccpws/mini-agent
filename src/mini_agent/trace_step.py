import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime


@dataclass
class TraceStep:
    question: str
    step: int
    thought: str
    action: str | None = None
    args: dict | None = None
    observation: str | None = None
    answer: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "step": self.step,
            "thought": self.thought,
            "action": self.action,
            "args": self.args,
            "observation": str(self.observation)[:500] if self.observation else None,
            "answer": self.answer,
        }


@dataclass
class TaskExecutionRecord:
    task_id: str
    task_description: str
    capability: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration: float = 0.0
    success: bool = False
    result: Any = None
    error: str | None = None
    failure_type: str | None = None
    retry_count: int = 0
    status: str = "PENDING"

    def start(self):
        self.start_time = datetime.now()
        self.status = "RUNNING"

    def complete(self, success: bool, result: Any = None, error: str = None, failure_type: str = None):
        self.end_time = datetime.now()
        self.success = success
        self.result = result
        self.error = error
        self.failure_type = failure_type
        self.status = "SUCCESS" if success else "FAILED"
        if self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "capability": self.capability,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "success": self.success,
            "result": str(self.result)[:500] if self.result else None,
            "error": self.error,
            "failure_type": self.failure_type,
            "retry_count": self.retry_count,
            "status": self.status,
        }


@dataclass
class ExecutionTrace:
    question: str = ""
    plan_goal: str = ""
    plan_version: int = 1
    records: list[TaskExecutionRecord] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None
    total_duration: float = 0.0
    success_count: int = 0
    fail_count: int = 0
    persist_dir: str = "traces"

    def start(self):
        self.start_time = datetime.now()

    def finish(self):
        self.end_time = datetime.now()
        if self.start_time:
            self.total_duration = (self.end_time - self.start_time).total_seconds()
        self.success_count = sum(1 for r in self.records if r.success)
        self.fail_count = sum(1 for r in self.records if not r.success)

    def add_record(self, record: TaskExecutionRecord):
        self.records.append(record)

    def get_record(self, task_id: str) -> TaskExecutionRecord | None:
        for record in self.records:
            if record.task_id == task_id:
                return record
        return None

    def get_summary(self) -> dict:
        return {
            "question": self.question,
            "plan_goal": self.plan_goal,
            "plan_version": self.plan_version,
            "total_tasks": len(self.records),
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "total_duration": self.total_duration,
            "records": [r.to_dict() for r in self.records],
        }

    def print_summary(self):
        print("=" * 60)
        print("Execution Trace Summary")
        print("=" * 60)
        print(f"Question: {self.question}")
        print(f"Plan Goal: {self.plan_goal}")
        print(f"Total Duration: {self.total_duration:.2f}s")
        print(f"Tasks: {self.success_count} success, {self.fail_count} failed")
        print()
        for record in self.records:
            status = "✓" if record.success else "✗"
            print(f"  {status} [{record.task_id}] {record.task_description}")
            print(f"    Duration: {record.duration:.2f}s, Retries: {record.retry_count}")
            if record.error:
                print(f"    Error: {record.error}")
        print("=" * 60)

    def save(self, filename: str = None):
        self.finish()
        persist_dir = Path(self.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trace_{timestamp}.json"

        filepath = persist_dir / filename
        data = self.get_summary()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"执行记录已保存: {filepath}")
        return filepath


class TraceLogger:
    def __init__(self, persist_dir: str = "traces"):
        self.logs: list[TraceStep] = []
        self.persist_dir = persist_dir

    def log(self, step: TraceStep):
        self.logs.append(step)

    def dump(self):
        for tracestep in self.logs:
            print(f"Step {tracestep.step}:")
            print(tracestep)

    def save(self, filename: str = None):
        persist_dir = Path(self.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"react_trace_{timestamp}.json"

        filepath = persist_dir / filename
        data = {
            "total_steps": len(self.logs),
            "logs": [step.to_dict() for step in self.logs],
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"ReAct 记录已保存: {filepath}")
        return filepath
