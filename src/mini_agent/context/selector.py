from mini_agent.context.budget import DynamicTokenBudget, TokenBudget
from mini_agent.context.models import ContextItem, ContextSource


class ContextSelector:

    def select(self, items: list[ContextItem], budget: TokenBudget | DynamicTokenBudget):
        if isinstance(budget, DynamicTokenBudget):
            return self._select_by_source(items, budget)
        return self._select_by_priority(items, budget)

    def _select_by_priority(self, items: list[ContextItem], budget: TokenBudget):
        items = sorted(items, key=lambda item: (item.priority, item.score), reverse=True)

        selected = []
        for item in items:
            if budget.consume(item.token_count):
                selected.append(item)

        return selected

    def _select_by_source(self, items: list[ContextItem], budget: DynamicTokenBudget):
        grouped = self._group_by_source(items)
        
        selected = []
        for source, source_items in grouped.items():
            source_items.sort(key=lambda x: (x.priority, x.score), reverse=True)
            for item in source_items:
                if budget.consume(source, item.token_count):
                    selected.append(item)

        return selected

    def _group_by_source(self, items: list[ContextItem]) -> dict[ContextSource, list[ContextItem]]:
        grouped = {source: [] for source in ContextSource}
        for item in items:
            grouped[item.source].append(item)
        return grouped
