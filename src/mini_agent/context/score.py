from mini_agent.context.models import ContextItem


class ContextScorer:
    """上下文评分器，计算 ContextItem 的综合得分"""

    def __init__(
        self,
        priority_weight: float = 0.3,
        relevance_weight: float = 0.3,
        recency_weight: float = 0.2,
        reliability_weight: float = 0.2,
    ):
        self.priority_weight = priority_weight
        self.relevance_weight = relevance_weight
        self.recency_weight = recency_weight
        self.reliability_weight = reliability_weight

    def score(self, item: ContextItem) -> float:
        """计算综合得分"""
        return (
            self.priority_weight * item.priority +
            self.relevance_weight * item.relevance +
            self.recency_weight * item.recency +
            self.reliability_weight * item.reliability
        )

    def score_batch(self, items: list[ContextItem]) -> list[tuple[ContextItem, float]]:
        """批量计算得分，返回 (item, score) 列表"""
        return [(item, self.score(item)) for item in items]

    def rank(self, items: list[ContextItem], top_k: int = None) -> list[ContextItem]:
        """按得分排序，返回排序后的列表"""
        scored = self.score_batch(items)
        scored.sort(key=lambda x: x[1], reverse=True)
        ranked = [item for item, _ in scored]
        if top_k:
            ranked = ranked[:top_k]
        return ranked
