from mini_agent.memory.working_memory import WorkingMemory
from mini_agent.memory.short_term_memory import ShortTermMemory
from mini_agent.memory.long_term_memory import LongTermMemory
from mini_agent.memory.models import Memory, MemoryType
from mini_agent.memory.consolidator import MemoryConsolidator


class MemoryManager:
    """记忆管理器，统一管理 working、short-term、long-term 记忆"""

    def __init__(self, short_term_max_size: int = 50, persist_dir: str = "memory_data", llm=None):
        self.working = WorkingMemory()
        self.short_term = ShortTermMemory(max_size=short_term_max_size)
        self.long_term = LongTermMemory(persist_dir=persist_dir)
        self.consolidator = MemoryConsolidator(llm) if llm else None

    def add(self, content: str, memory_type: MemoryType = MemoryType.FACT, scope: str = "working"):
        """添加记忆"""
        memory = Memory(content=content, memory_type=memory_type)

        if scope == "working":
            self.working.add(content, memory)
        elif scope == "short_term":
            self.short_term.add(memory)
        elif scope == "long_term":
            self.long_term.add(memory)
        else:
            self.short_term.add(memory)

    def add_memory(self, memory: Memory, scope: str = "short_term"):
        """添加 Memory 对象"""
        if scope == "working":
            self.working.add(memory.content, memory)
        elif scope == "short_term":
            self.short_term.add(memory)
        elif scope == "long_term":
            self.long_term.add(memory)
        else:
            self.short_term.add(memory)

    def search(self, query: str, top_k: int = 5) -> list[Memory]:
        """检索记忆（从 short_term + long_term）"""
        results = []

        short_term_results = self.short_term.search(query, top_k=top_k)
        results.extend(short_term_results)

        long_term_results = self.long_term.search(query, top_k=top_k)
        for m in long_term_results:
            if m.content not in [r.content for r in results]:
                results.append(m)

        return results[:top_k]

    def get_context(self) -> str:
        """获取当前上下文（用于 LLM prompt）"""
        parts = []

        working_data = self.working.get_all()
        if working_data:
            parts.append(f"当前任务: {working_data}")

        recent = self.short_term.get_recent(5)
        if recent:
            memories_str = "\n".join([f"- {m.content}" for m in recent])
            parts.append(f"最近记忆:\n{memories_str}")

        return "\n\n".join(parts)

    def promote_to_long_term(self, memory: Memory):
        """将记忆提升为长期记忆"""
        self.long_term.add(memory)

    def clean(self, max_age_days: int = 50) -> dict:
        """清理过期记忆"""
        return self.long_term.clean(max_age_days)

    def consolidate(self) -> dict:
        """整合琐碎记忆"""
        if not self.consolidator:
            return {"error": "未配置 LLM，无法整合"}

        return self.long_term.vector_store.consolidate(self.consolidator)

    def clear_working(self):
        """清空当前任务记忆"""
        self.working.clear()

    def clear_all(self):
        """清空所有记忆"""
        self.working.clear()
        self.short_term.clear()

    def stats(self) -> dict:
        """返回记忆统计"""
        return {
            "working": len(self.working.get_all()),
            "short_term": self.short_term.count(),
            "long_term": self.long_term.count()
        }

    def __str__(self):
        return f"MemoryManager({self.stats()})"
