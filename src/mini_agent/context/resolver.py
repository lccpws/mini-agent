from mini_agent.context.models import ContextItem
from mini_agent.context.policy import ContextPolicy


class ContextResolver:
    """上下文解析器，按 id 去重并选择最优 item"""

    def __init__(self, policy: ContextPolicy = None):
        self.policy = policy or ContextPolicy()

    def resolve(self, items: list[ContextItem]) -> list[ContextItem]:
        """按 id 分组，每组选择 priority 最高的 item"""
        groups: dict[str, list[ContextItem]] = {}

        for item in items:
            key = item.id
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        result = []
        for candidates in groups.values():
            winner = max(candidates, key=self._priority)
            result.append(winner)

        return result

    def _priority(self, item: ContextItem) -> float:
        """计算 item 的优先级分数"""
        source = item.source.value if hasattr(item.source, 'value') else item.source
        source_priority = self.policy.get_priority(source)
        max_source_priority = 5.0
        normalized_source = max(0.0, 1.0 - (source_priority / max_source_priority))

        return (
            normalized_source * 100
            + item.reliability * 10
            + item.recency * 5
        )
