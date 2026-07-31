from mini_agent.context.budget import DynamicTokenBudget, TokenBudget
from mini_agent.context.compressor import ContextCompressor
from mini_agent.context.models import ContextItem, ContextRoute, ContextSource
from mini_agent.context.policy import ContextPolicy
from mini_agent.context.selector import ContextSelector


class ContextManager:
    """上下文管理器，整合 Routing → Selection → Budget → Compression 流程"""

    def __init__(
        self,
        selector: ContextSelector = None,
        compressor: ContextCompressor = None,
        policy: ContextPolicy = None,
        total_tokens: int = 8000,
        output_tokens: int = 2000,
        use_dynamic_budget: bool = True
    ):
        self.selector = selector or ContextSelector()
        self.compressor = compressor or ContextCompressor()
        self.policy = policy or ContextPolicy()
        self.total_tokens = total_tokens
        self.output_tokens = output_tokens
        self.use_dynamic_budget = use_dynamic_budget

    def build_context(
        self,
        items: list[ContextItem],
        route: ContextRoute = None,
        total_tokens: int = None,
        output_tokens: int = None
    ) -> list[ContextItem]:
        total = total_tokens or self.total_tokens
        output = output_tokens or self.output_tokens
        
        if self.use_dynamic_budget:
            budget = DynamicTokenBudget(total, output, self.policy)
        else:
            budget = TokenBudget(total, output)
        
        filtered_items = self._filter_by_route(items, route)
        
        filtered_items = self._apply_policy(filtered_items)
        
        selected = self.selector.select(filtered_items, budget)
        
        target_per_item = budget.input_budget // max(len(selected), 1)
        for item in selected:
            item = self.compressor.compress(item, target_per_item)
        
        selected = sorted(selected, key=lambda item: item.priority, reverse=True)
        
        return selected

    def _filter_by_route(self, items: list[ContextItem], route: ContextRoute = None) -> list[ContextItem]:
        if route is None:
            return items
        
        filtered = []
        for item in items:
            if item.source == ContextSource.SYSTEM and not route.needs_system:
                continue
            if item.source == ContextSource.USER and not route.needs_user:
                continue
            if item.source == ContextSource.MEMORY and not route.needs_memory:
                continue
            if item.source == ContextSource.RAG and not route.needs_rag:
                continue
            if item.source == ContextSource.HISTORY and not route.needs_history:
                continue
            filtered.append(item)
        
        return filtered

    def _apply_policy(self, items: list[ContextItem]) -> list[ContextItem]:
        """根据策略调整优先级并过滤非必须项"""
        filtered = []
        for item in items:
            if self.policy.is_required(item.source):
                item.priority = self.policy.get_priority(item.source)
                filtered.append(item)
            else:
                policy_priority = self.policy.get_priority(item.source)
                if policy_priority > 0:
                    item.priority = policy_priority
                    filtered.append(item)
        return filtered

    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "output_tokens": self.output_tokens,
            "input_budget": self.total_tokens - self.output_tokens
        }
