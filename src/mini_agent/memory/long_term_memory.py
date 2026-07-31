from mini_agent.memory.models import Memory
from mini_agent.memory.vector_store import VectorStore
from mini_agent.memory.embedder import Embedder
from mini_agent.memory.cleaner import MemoryCleaner


class LongTermMemory:
    """长期记忆，基于向量的语义检索"""

    def __init__(self, persist_dir: str = "memory_data"):
        self.embedder = Embedder()
        self.vector_store = VectorStore(self.embedder, persist_dir)
        self.cleaner = MemoryCleaner()

    def add(self, memory: Memory):
        self.vector_store.add(memory)

    def search(self, query: str, top_k: int = 3) -> list[Memory]:
        return self.vector_store.search(query, top_k)

    def clean(self, max_age_days: int = 50) -> dict:
        """执行清理"""
        self.cleaner.max_age_days = max_age_days
        return self.cleaner.clean(self.vector_store.memories, self.vector_store)

    def consolidate(self, consolidator) -> dict:
        """整合记忆"""
        return self.vector_store.consolidate(consolidator)

    def count(self) -> int:
        return self.vector_store.count()

    def __str__(self):
        return f"LongTermMemory(count={self.vector_store.count()})"
