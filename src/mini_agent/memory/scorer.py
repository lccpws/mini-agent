from datetime import datetime
from mini_agent.memory.models import Memory


class MemoryScorer:
    """记忆得分计算器"""

    def score(self, memory: Memory) -> float:
        """计算记忆得分"""
        days = (datetime.now() - memory.last_access).days

        recency = max(0, 1 - days / 365)

        frequency = min(memory.access_count / 20, 1)

        importance_score = memory.importance.value / 3

        score = (
            0.5 * importance_score +
            0.3 * frequency +
            0.2 * recency
        )

        return round(score, 3)

    def rank(self, memories: list[Memory]) -> list[tuple[Memory, float]]:
        """对记忆列表按得分排序"""
        scored = [(m, self.score(m)) for m in memories]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def filter_by_score(self, memories: list[Memory], min_score: float = 0.3) -> list[Memory]:
        """按最低得分过滤记忆"""
        return [m for m in memories if self.score(m) >= min_score]
