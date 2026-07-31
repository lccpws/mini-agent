from mini_agent.memory.models import Memory


class ShortTermMemory:
    """短期记忆，使用环形缓冲区，保留最近 N 条"""

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.buffer = []

    def add(self, memory: Memory):
        self.buffer.append(memory)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def search(self, query: str, top_k: int = 3) -> list[Memory]:
        """关键词匹配 + 相关度排序"""
        query_words = set(query)
        scored = []

        for memory in self.buffer:
            score = sum(1 for char in memory.content if char in query_words)
            if score > 0:
                scored.append((memory, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    def get_recent(self, n: int = 10) -> list[Memory]:
        return self.buffer[-n:]

    def clear(self):
        self.buffer = []

    def count(self) -> int:
        return len(self.buffer)

    def __str__(self):
        return f"ShortTermMemory(count={len(self.buffer)})"
