from mini_agent.memory.vector_store import VectorStore
from mini_agent.memory.models import Memory


class MemoryRetriever:
    """记忆检索器"""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 3) -> list[Memory]:
        """语义检索记忆"""
        return self.vector_store.search(query, top_k)

    def add_and_search(self, memory: Memory, query: str, top_k: int = 3) -> list[Memory]:
        """添加记忆并检索相关记忆"""
        self.vector_store.add(memory)
        return self.vector_store.search(query, top_k)
