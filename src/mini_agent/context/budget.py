from mini_agent.context.models import ContextSource
from mini_agent.context.policy import ContextPolicy


class TokenBudget:
    def __init__(self, total_tokens: int, output_tokens: int):
        self.total_tokens = total_tokens
        self.output_tokens = output_tokens
        self.used_token = 0

    @property
    def input_budget(self):
        return self.total_tokens - self.output_tokens
    
    def can_fit(self, token_counts: int):
        return self.used_token + token_counts <= self.input_budget
    
    def consume(self, token_counts: int):
        if not self.can_fit(token_counts):
            return False
        self.used_token += token_counts
        return True
    
    def estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class DynamicTokenBudget:
    """动态预算，支持按 source 分配独立预算"""

    def __init__(self, total_tokens: int, output_tokens: int, policy: ContextPolicy = None):
        self.total_tokens = total_tokens
        self.output_tokens = output_tokens
        self.input_budget = total_tokens - output_tokens
        self.policy = policy or ContextPolicy()
        self.source_budgets = self._calculate_budgets()
        self.source_used = {s: 0 for s in ContextSource}

    def _calculate_budgets(self) -> dict[ContextSource, int]:
        budgets = {}
        for source in ContextSource:
            ratio = self.policy.get_budget_ratio(source)
            budgets[source] = int(self.input_budget * ratio)
        return budgets

    def can_fit(self, source: ContextSource, tokens: int) -> bool:
        return self.source_used[source] + tokens <= self.source_budgets[source]

    def consume(self, source: ContextSource, tokens: int) -> bool:
        if not self.can_fit(source, tokens):
            return False
        self.source_used[source] += tokens
        return True

    def get_remaining(self, source: ContextSource) -> int:
        return self.source_budgets[source] - self.source_used[source]

    def get_usage(self) -> dict:
        return {
            source.value: {
                "budget": self.source_budgets[source],
                "used": self.source_used[source],
                "remaining": self.get_remaining(source),
            }
            for source in ContextSource
        }
