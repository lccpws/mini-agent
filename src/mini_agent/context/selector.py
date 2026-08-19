from mini_agent.context.budget import DynamicTokenBudget, TokenBudget
from mini_agent.context.models import ContextItem
from mini_agent.context.policy import ContextPolicy
from mini_agent.context.score import ContextScorer


class ContextSelector:

    def __init__(self, policy: ContextPolicy = None):
        self.policy = policy or ContextPolicy()
        self.scorer = ContextScorer(policy=self.policy)

    def select(self, items: list[ContextItem], budget: TokenBudget | DynamicTokenBudget, query: str = ""):
        if isinstance(budget, DynamicTokenBudget):
            return self._select_by_source(items, budget, query)
        return self._select_by_priority(items, budget, query)

    def _select_by_priority(self, items: list[ContextItem], budget: TokenBudget, query: str = ""):
        items = sorted(items, key=lambda item: self.scorer.utility(item, query), reverse=True)

        selected = []
        for item in items:
            if budget.consume(item.token_count):
                selected.append(item)

        return selected

    def _select_by_source(self, items: list[ContextItem], budget: DynamicTokenBudget, query: str = ""):
        grouped = self._group_by_source(items)
        
        selected = []
        for source, source_items in grouped.items():
            source_items.sort(key=lambda x: self.scorer.utility(x, query), reverse=True)
            for item in source_items:
                source_str = source.value if hasattr(source, 'value') else source
                if budget.consume(source_str, item.token_count):
                    selected.append(item)

        return selected

    def _group_by_source(self, items: list[ContextItem]) -> dict[str, list[ContextItem]]:
        grouped: dict[str, list[ContextItem]] = {}
        for item in items:
            if item.source not in grouped:
                grouped[item.source] = []
            grouped[item.source].append(item)
        return grouped
