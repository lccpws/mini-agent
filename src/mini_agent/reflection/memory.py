import json
import uuid
from pathlib import Path
from datetime import datetime
from mini_agent.reflection.models import ReflectionRecord
from mini_agent.planner.models import Task


class ReflectionMemory:
    def __init__(self, persist_dir: str = "memory_data", embedder=None):
        self.records: list[ReflectionRecord] = []
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_file = self.persist_dir / "reflection_memory.json"

        self.vector_index = None
        if embedder is not None:
            from mini_agent.reflection.vector_index import ReflectionVectorIndex
            self.vector_index = ReflectionVectorIndex(embedder, persist_dir)

        self._load()

    def _load(self):
        if self.persist_file.exists():
            try:
                with open(self.persist_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.records = [ReflectionRecord.from_dict(r) for r in data]
            except Exception:
                self.records = []

        if self.vector_index and self.vector_index.count() == 0 and self.records:
            for record in self.records:
                self._index_record(record)
            self.vector_index.save()

    def _index_record(self, record: ReflectionRecord):
        """将记录添加到向量索引"""
        if self.vector_index is None:
            return

        text = self._record_to_text(record)
        metadata = {
            "capability": record.capability,
            "error_message": record.error_message,
            "root_cause": record.root_cause,
            "success_count": record.success_count,
            "fail_count": record.fail_count,
        }
        self.vector_index.add(record.id, text, metadata)

    def _record_to_text(self, record: ReflectionRecord) -> str:
        parts = [
            f"能力: {record.capability}",
            f"错误: {record.error_message}",
            f"原因: {record.root_cause}",
        ]
        if record.alternative_capability:
            parts.append(f"替代能力: {record.alternative_capability}")
        if record.alternative_input:
            parts.append(f"建议输入: {json.dumps(record.alternative_input, ensure_ascii=False)}")
        return " ".join(parts)

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

            if record.score > existing.score:
                existing.root_cause = record.root_cause
                existing.alternative_capability = record.alternative_capability
                existing.alternative_input = record.alternative_input
                existing.suggestions = record.suggestions
                existing.score = record.score

                if self.vector_index and self.vector_index.contains(existing.id):
                    self.vector_index.remove(existing.id)
                    self._index_record(existing)
                    self.vector_index.save()
            else:
                if self.vector_index and self.vector_index.contains(existing.id):
                    idx = self.vector_index.id_to_idx[existing.id]
                    self.vector_index.records[idx].metadata["fail_count"] = existing.fail_count
                    self.vector_index.save()
        else:
            if not record.id:
                record.id = str(uuid.uuid4())
            self.records.append(record)
            self._index_record(record)
            if self.vector_index:
                self.vector_index.save()

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

    def search_semantic(self, query: str, top_k: int = 5) -> list[tuple[ReflectionRecord, float]]:
        """语义检索，返回 (record, similarity_score)"""
        if not self.vector_index:
            return [(r, 0.0) for r in self.records[:top_k]]

        results = self.vector_index.search(query, top_k)
        records_with_score = []
        for record_id, score, metadata in results:
            record = next((r for r in self.records if r.id == record_id), None)
            if record:
                records_with_score.append((record, score))
        return records_with_score

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
        if self.vector_index:
            results = self.vector_index.search(error1, top_k=1)
            if results and results[0][1] > 0.8:
                return True

        return self._keyword_similar(error1, error2)

    def _keyword_similar(self, error1: str, error2: str) -> bool:
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
            "vector_index_enabled": self.vector_index is not None,
            "vector_index_count": self.vector_index.count() if self.vector_index else 0,
        }

    def get_replan_context(self, capabilities: list[str]) -> str:
        parts = []
        for cap in capabilities:
            records = self.search_by_capability(cap)
            if not records:
                continue
            successful = [r for r in records if r.success_count > 0]
            failed = [r for r in records if r.fail_count > 0]
            if not successful and not failed:
                continue
            parts.append(f"\n能力 '{cap}' 的历史经验:")
            for r in successful:
                parts.append(f"  - 成功方案: 替代能力={r.alternative_capability}, 建议输入={r.alternative_input} (成功{r.success_count}次)")
            for r in failed[:2]:
                parts.append(f"  - 失败原因: {r.root_cause} (失败{r.fail_count}次)")
                if r.alternative_capability:
                    parts.append(f"    建议替代: {r.alternative_capability}")
        return "\n".join(parts) if parts else ""

    def clear(self):
        self.records = []
        self._persist()
        if self.vector_index:
            if self.vector_index.index_path.exists():
                self.vector_index.index_path.unlink()
            if self.vector_index.records_path.exists():
                self.vector_index.records_path.unlink()
            self.vector_index = None
