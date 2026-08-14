from mini_agent.context.budget import DynamicTokenBudget, TokenBudget
from mini_agent.context.models import ContextItem


class ContextSelector:

    def __init__(self):
        from mini_agent.context.score import ContextScorer
        self.scorer = ContextScorer()

    def select(self, items: list[ContextItem], budget: TokenBudget | DynamicTokenBudget):
        if isinstance(budget, DynamicTokenBudget):
            return self._select_by_source(items, budget)
        return self._select_by_priority(items, budget)

    def _select_by_priority(self, items: list[ContextItem], budget: TokenBudget):
        items = sorted(items, key=lambda item: self.scorer.score(item), reverse=True)

        selected = []
        for item in items:
            if budget.consume(item.token_count):
                selected.append(item)

        return selected

    def _select_by_source(self, items: list[ContextItem], budget: DynamicTokenBudget):
        grouped = self._group_by_source(items)
        
        selected = []
        for source, source_items in grouped.items():
            source_items.sort(key=lambda x: self.scorer.score(x), reverse=True)
            for item in source_items:
                if budget.consume(source, item.token_count):
                    selected.append(item)

        return selected

    def _group_by_source(self, items: list[ContextItem]) -> dict[str, list[ContextItem]]:
        grouped: dict[str, list[ContextItem]] = {}
        for item in items:
            if item.source not in grouped:
                grouped[item.source] = []
            grouped[item.source].append(item)
        return grouped
