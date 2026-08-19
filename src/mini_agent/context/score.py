import math
from datetime import datetime

from mini_agent.context.models import ContextItem
from mini_agent.context.policy import ContextPolicy


class ContextScorer:
    """上下文评分器，计算 ContextItem 的综合得分"""

    RELIABILITY_SCORES = {
        "system": 1.0,
        "user": 0.9,
        "rag": 0.8,
        "memory": 0.7,
        "tool": 0.6,
        "history": 0.5,
    }

    def __init__(
        self,
        embedder=None,
        priority_weight: float = 0.25,
        relevance_weight: float = 0.25,
        recency_weight: float = 0.15,
        reliability_weight: float = 0.15,
        source_priority_weight: float = 0.2,
        recency_half_life_hours: float = 24.0,
        policy: ContextPolicy = None,
    ):
        self.embedder = embedder
        self.priority_weight = priority_weight
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.reliability_weight = reliability_weight
        self.source_priority_weight = source_priority_weight
        self.recency_half_life_hours = recency_half_life_hours
        self.policy = policy or ContextPolicy()

    def score(self, item: ContextItem, query: str = "") -> float:
        """计算综合得分"""
        relevance = self._calc_relevance(item, query)
        recency = self._calc_recency(item)
        reliability = self._calc_reliability(item)
        source_priority = self._calc_source_priority(item)

        item.relevance = relevance
        item.recency = recency
        item.reliability = reliability

        return (
            self.priority_weight * item.priority +
            self.relevance_weight * relevance +
            self.recency_weight * recency +
            self.reliability_weight * reliability +
            self.source_priority_weight * source_priority
        )

    def utility(self, item: ContextItem, query: str = "") -> float:
        """计算效用：score / token_count"""
        score = self.score(item, query)
        if item.token_count <= 0:
            return score
        return score / item.token_count

    def _calc_relevance(self, item: ContextItem, query: str) -> float:
        """计算相关性：embedding 余弦相似度"""
        if not query or not self.embedder:
            return item.relevance

        try:
            query_vec = self.embedder.embed(query)
            item_vec = self.embedder.embed(item.content)
            similarity = self._cosine_similarity(query_vec, item_vec)
            return max(0.0, min(1.0, similarity))
        except Exception:
            return item.relevance

    def _calc_recency(self, item: ContextItem) -> float:
        """计算时效性：指数衰减"""
        if not item.created_at:
            return 0.5

        now = datetime.now()
        age_hours = (now - item.created_at).total_seconds() / 3600
        decay = math.exp(-0.693 * age_hours / self.recency_half_life_hours)
        return round(decay, 4)

    def _calc_reliability(self, item: ContextItem) -> float:
        """计算可靠性：基于数据源类型"""
        return self.RELIABILITY_SCORES.get(item.source, 0.5)

    def _calc_source_priority(self, item: ContextItem) -> float:
        """计算 source 优先级：数值越小优先级越高，转换为 0-1 分数"""
        policy_priority = self.policy.get_priority(item.source)
        max_priority = 5.0
        return max(0.0, 1.0 - (policy_priority / max_priority))

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def score_batch(self, items: list[ContextItem], query: str = "") -> list[tuple[ContextItem, float]]:
        """批量计算得分"""
        return [(item, self.score(item, query)) for item in items]

    def rank(self, items: list[ContextItem], query: str = "", top_k: int | None = None) -> list[ContextItem]:
        """按得分排序"""
        scored = self.score_batch(items, query)
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [item for item, _ in scored]
        if top_k:
            ranked = ranked[:top_k]
        return ranked
