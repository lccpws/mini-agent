from datetime import datetime
from mini_agent.memory.models import Memory, Importance, MemoryStatus
from mini_agent.memory.embedder import Embedder


class MemoryDeduplicator:
    """记忆去重器，基于语义相似度合并重复记忆"""

    def __init__(self, similarity_threshold: float = 0.9):
        self.threshold = similarity_threshold
        self.embedder = Embedder()

    def find_duplicate(
        self,
        new_memory: Memory,
        memories: list[Memory],
        vectors: list[list[float]]
    ) -> tuple[int, float] | None:
        """查找重复记忆，返回 (索引, 相似度) 或 None"""
        new_vector = self.embedder.embed(new_memory.content)

        for i, (memory, vector) in enumerate(zip(memories, vectors)):
            if not memory.is_active():
                continue

            score = self._cosine_similarity(new_vector, vector)

            if score >= self.threshold:
                return (i, score)

        return None

    def merge(self, existing: Memory, new_memory: Memory) -> Memory:
        """合并两条记忆"""
        existing.content = self._merge_content(existing.content, new_memory.content)
        existing.access_count += new_memory.access_count
        existing.last_access = max(existing.last_access, new_memory.last_access)
        existing.importance = self._merge_importance(existing.importance, new_memory.importance)
        existing.metadata = self._merge_metadata(existing.metadata, new_memory.metadata)

        return existing

    def _merge_content(self, old_content: str, new_content: str) -> str:
        """合并内容，保留较长的"""
        if len(new_content) > len(old_content):
            return new_content
        return old_content

    def _merge_importance(self, old: Importance, new: Importance) -> Importance:
        """合并重要性，取较高值"""
        return max(old, new, key=lambda x: x.value)

    def _merge_metadata(self, old_meta: dict, new_meta: dict) -> dict:
        """合并元数据"""
        merged = old_meta.copy()
        merged.update(new_meta)
        return merged

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度计算"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
