from mini_agent.context.budget import DynamicTokenBudget, TokenBudget
from mini_agent.context.builder import ContextBuilder
from mini_agent.context.compressor import ContextCompressor, PriorityCompressor
from mini_agent.context.models import ContextItem, ContextRoute
from mini_agent.context.policy import ContextPolicy
from mini_agent.context.resolver import ContextResolver
from mini_agent.context.score import ContextScorer
from mini_agent.context.selector import ContextSelector


class ContextManager:
    """上下文管理器，整合 Routing → Selection → Budget → Compression 流程"""

    def __init__(
        self,
        selector=None,
        compressor: ContextCompressor = None,
        policy: ContextPolicy = None,
        scorer: ContextScorer = None,
        resolver: ContextResolver = None,
        priority_compressor: PriorityCompressor = None,
        total_tokens: int = 8000,
        output_tokens: int = 2000,
        use_dynamic_budget: bool = True
    ):
        self.policy = policy or ContextPolicy()
        if isinstance(selector, ContextSelector):
            self.selector = selector
        elif isinstance(selector, ContextBuilder):
            self.selector = selector.selector
        else:
            self.selector = ContextSelector(policy=self.policy)
        self.compressor = compressor or ContextCompressor()
        self.scorer = scorer or ContextScorer(policy=self.policy)
        self.resolver = resolver or ContextResolver(policy=self.policy)
        self.priority_compressor = priority_compressor or PriorityCompressor(policy=self.policy)
        self.total_tokens = total_tokens
        self.output_tokens = output_tokens
        self.use_dynamic_budget = use_dynamic_budget
        
        self.cached_items: list[ContextItem] = []
        self.cached_route: ContextRoute = None

    def build_context(
        self,
        items: list[ContextItem],
        route: ContextRoute = None,
        query: str = "",
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
        
        filtered_items = self.resolver.resolve(filtered_items)
        
        selected = self.selector.select(filtered_items, budget, query)
        
        selected = self.priority_compressor.compress_all(selected)
        
        selected = sorted(selected, key=lambda item: self.scorer.utility(item, query), reverse=True)
        
        self.cached_items = selected
        self.cached_route = route
        
        return selected

    def _filter_by_route(self, items: list[ContextItem], route: ContextRoute = None) -> list[ContextItem]:
        if route is None:
            return items
        
        filtered = []
        for item in items:
            if item.source == "system" and not route.needs_system:
                continue
            if item.source == "user" and not route.needs_user:
                continue
            if item.source == "memory" and not route.needs_memory:
                continue
            if item.source == "rag" and not route.needs_rag:
                continue
            if item.source == "history" and not route.needs_history:
                continue
            filtered.append(item)
        
        return filtered

    def _apply_policy(self, items: list[ContextItem]) -> list[ContextItem]:
        """根据策略调整优先级并过滤非必须项"""
        filtered = []
        for item in items:
            source = item.source.value if hasattr(item.source, 'value') else item.source
            if self.policy.is_required(source):
                item.priority = self.policy.get_priority(source)
                filtered.append(item)
            else:
                policy_priority = self.policy.get_priority(source)
                if policy_priority > 0:
                    item.priority = policy_priority
                    filtered.append(item)
        return filtered

    def update(self, new_item: ContextItem, query: str = "") -> list[ContextItem]:
        """增量更新：合并新 item 到缓存，重新选择"""
        all_items = self.cached_items + [new_item]
        
        all_items = self.resolver.resolve(all_items)
        
        all_items = self._apply_policy(all_items)
        
        budget = self._create_budget()
        selected = self.selector.select(all_items, budget, query)
        
        selected = self.priority_compressor.compress_all(selected)
        
        selected = sorted(selected, key=lambda item: self.scorer.utility(item, query), reverse=True)
        
        self.cached_items = selected
        return selected
    
    def _create_budget(self):
        """根据配置创建 budget"""
        if self.use_dynamic_budget:
            return DynamicTokenBudget(self.total_tokens, self.output_tokens, self.policy)
        return TokenBudget(self.total_tokens, self.output_tokens)
    
    def get_stats(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "output_tokens": self.output_tokens,
            "input_budget": self.total_tokens - self.output_tokens,
            "cached_items_count": len(self.cached_items)
        }
