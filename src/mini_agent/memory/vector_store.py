import json
from pathlib import Path
from mini_agent.memory.models import Memory, MemoryType, Importance, MemoryStatus
from mini_agent.memory.embedder import Embedder
from mini_agent.memory.scorer import MemoryScorer
from mini_agent.memory.deduplicator import MemoryDeduplicator


class VectorStore:
    """向量存储，支持语义检索、去重和双文件分离持久化"""

    def __init__(
        self,
        embedder: Embedder,
        persist_dir: str = "memory_data",
        similarity_weight: float = 0.7,
        memory_weight: float = 0.3,
        dedup_threshold: float = 0.9
    ):
        self.embedder = embedder
        self.memories: list[Memory] = []
        self.vectors: list[list[float]] = []
        self.persist_dir = Path(persist_dir)
        self.active_path = self.persist_dir / "active.json"
        self.archived_path = self.persist_dir / "archived.json"
        self.sim_weight = similarity_weight
        self.mem_weight = memory_weight
        self.scorer = MemoryScorer()
        self.deduplicator = MemoryDeduplicator(similarity_threshold=dedup_threshold)

        self._load()

    def add(self, memory: Memory, skip_dedup: bool = False):
        """添加记忆并计算向量（支持去重）"""
        if not skip_dedup and self.memories:
            result = self.deduplicator.find_duplicate(
                memory, self.memories, self.vectors
            )

            if result:
                idx, score = result
                existing = self.memories[idx]
                self.deduplicator.merge(existing, memory)
                print(f"合并记忆 (相似度: {score:.2f}): {memory.content[:30]}...")
                self._save()
                return

        vector = self.embedder.embed(memory.content)
        self.memories.append(memory)
        self.vectors.append(vector)
        self._save()

    def search(
        self,
        query: str,
        top_k: int = 3,
        use_memory_score: bool = True,
        include_archived: bool = False
    ) -> list[Memory]:
        """语义检索（支持混合得分）"""
        if not self.memories:
            return []

        query_vector = self.embedder.embed(query)

        similarity_scores = []
        for i, vector in enumerate(self.vectors):
            memory = self.memories[i]
            if not include_archived and not memory.is_active():
                continue
            score = self._cosine_similarity(query_vector, vector)
            similarity_scores.append((i, score))

        if use_memory_score:
            final_scores = self._blend_scores(similarity_scores)
        else:
            final_scores = similarity_scores

        final_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in final_scores[:top_k]:
            memory = self.memories[idx]
            memory.touch()
            results.append(memory)

        return results

    def _blend_scores(self, similarity_scores: list[tuple[int, float]]) -> list[tuple[int, float]]:
        """融合相似度得分和记忆得分"""
        blended = []

        for idx, sim_score in similarity_scores:
            memory = self.memories[idx]
            mem_score = self.scorer.score(memory)

            final_score = (
                self.sim_weight * sim_score +
                self.mem_weight * mem_score
            )

            blended.append((idx, final_score))

        return blended

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度计算"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _serialize(self, memory: Memory, vector: list[float]) -> dict:
        """序列化记忆"""
        return {
            "id": memory.id,
            "content": memory.content,
            "memory_type": memory.memory_type.value,
            "importance": memory.importance.value,
            "status": memory.status.value,
            "source": memory.source,
            "metadata": memory.metadata,
            "created_at": memory.created_at.isoformat(),
            "last_access": memory.last_access.isoformat(),
            "access_count": memory.access_count,
            "vector": vector
        }

    def _deserialize(self, item: dict) -> tuple[Memory, list[float]]:
        """反序列化记忆"""
        memory = Memory(
            id=item.get("id", ""),
            content=item["content"],
            memory_type=MemoryType(item["memory_type"]),
            importance=Importance(item["importance"]),
            status=MemoryStatus(item.get("status", "active")),
            source=item.get("source", "conversation"),
            metadata=item.get("metadata", {}),
            access_count=item.get("access_count", 0)
        )
        return memory, item["vector"]

    def _save(self):
        """按状态分别保存到不同文件"""
        active_data = []
        archived_data = []

        for memory, vector in zip(self.memories, self.vectors):
            item = self._serialize(memory, vector)
            if memory.is_active():
                active_data.append(item)
            else:
                archived_data.append(item)

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.active_path.write_text(json.dumps(active_data, ensure_ascii=False))
        self.archived_path.write_text(json.dumps(archived_data, ensure_ascii=False))

    def _load(self):
        """加载数据（活跃记忆 + 归档记忆）"""
        if self.active_path.exists():
            active_data = json.loads(self.active_path.read_text())
            for item in active_data:
                memory, vector = self._deserialize(item)
                self.memories.append(memory)
                self.vectors.append(vector)

        if self.archived_path.exists():
            archived_data = json.loads(self.archived_path.read_text())
            for item in archived_data:
                memory, vector = self._deserialize(item)
                self.memories.append(memory)
                self.vectors.append(vector)

    def list_memory(self, include_archived: bool = False) -> list[Memory]:
        """列出所有记忆"""
        if include_archived:
            return self.memories
        return [m for m in self.memories if m.is_active()]

    def list_archived(self) -> list[Memory]:
        """列出所有归档记忆"""
        return [m for m in self.memories if m.is_archived()]

    def consolidate(self, consolidator) -> dict:
        """整合记忆"""
        active_memories = [m for m in self.memories if m.is_active()]

        if len(active_memories) < 2:
            return {"original": len(active_memories), "consolidated": len(active_memories)}

        consolidated = consolidator.consolidate(active_memories)

        self.memories = [m for m in self.memories if not m.is_active()]
        self.vectors = [v for m, v in zip(self.memories, self.vectors) if not m.is_active()]

        for memory in consolidated:
            vector = self.embedder.embed(memory.content)
            self.memories.append(memory)
            self.vectors.append(vector)

        self._save()

        return {
            "original": len(active_memories),
            "consolidated": len(consolidated)
        }

    def count(self, include_archived: bool = False) -> int:
        """返回记忆数量"""
        if include_archived:
            return len(self.memories)
        return len([m for m in self.memories if m.is_active()])
