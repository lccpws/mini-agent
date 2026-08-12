import json
import uuid
from pathlib import Path
from datetime import datetime
from mini_agent.reflection.models import ReflectionRecord
from mini_agent.planner.models import Task


class ReflectionMemory:
    def __init__(self, persist_dir: str = "memory_data"):
        self.records: list[ReflectionRecord] = []
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_file = self.persist_dir / "reflection_memory.json"
        self._load()

    def _load(self):
        if self.persist_file.exists():
            try:
                with open(self.persist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = [ReflectionRecord.from_dict(r) for r in data]
            except Exception:
                self.records = []

    def _persist(self):
        try:
            with open(self.persist_file, "w", encoding="utf-8") as f:
                json.dump([r.to_dict() for r in self.records], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存反思记忆失败: {e}")

    def save(self, record: ReflectionRecord):
        existing = self.find_by_capability_and_error(record.capability, record.error_message)
        if existing:
            existing.fail_count += 1
            existing.last_used = datetime.now()
        else:
            if not record.id:
                record.id = str(uuid.uuid4())
            self.records.append(record)
        self._persist()

    def find_by_capability_and_error(self, capability: str, error_message: str) -> ReflectionRecord | None:
        for record in self.records:
            if record.capability == capability and self._is_similar_error(error_message, record.error_message):
                return record
        return None

    def search(self, capability: str, error_message: str = "") -> list[ReflectionRecord]:
        results = []
        for record in self.records:
            if record.capability == capability:
                if not error_message or self._is_similar_error(error_message, record.error_message):
                    results.append(record)
        return sorted(results, key=lambda r: (r.success_count, -r.fail_count), reverse=True)

    def search_by_capability(self, capability: str) -> list[ReflectionRecord]:
        return [r for r in self.records if r.capability == capability]

    def apply_experience(self, task: Task, record: ReflectionRecord) -> Task:
        record.success_count += 1
        record.last_used = datetime.now()
        self._persist()

        if record.alternative_capability:
            task.capability = record.alternative_capability
        if record.alternative_input:
            task.input.update(record.alternative_input)
        return task

    def get_most_successful(self, capability: str, limit: int = 3) -> list[ReflectionRecord]:
        records = self.search_by_capability(capability)
        return sorted(records, key=lambda r: r.success_count, reverse=True)[:limit]

    def _is_similar_error(self, error1: str, error2: str) -> bool:
        error1_lower = error1.lower()
        error2_lower = error2.lower()

        if error1_lower == error2_lower:
            return True

        keywords1 = set(error1_lower.split())
        keywords2 = set(error2_lower.split())
        intersection = keywords1 & keywords2

        if len(intersection) >= 2:
            return True

        return False

    def get_stats(self) -> dict:
        return {
            "total_records": len(self.records),
            "capabilities": list(set(r.capability for r in self.records)),
            "total_success": sum(r.success_count for r in self.records),
            "total_fail": sum(r.fail_count for r in self.records),
        }

    def clear(self):
        self.records = []
        self._persist()
