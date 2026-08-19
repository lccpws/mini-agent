from mini_agent.context.budget import TokenBudget
from mini_agent.context.compressor import ContextCompressor
from mini_agent.context.models import ContextItem
from mini_agent.context.policy import ContextPolicy
from mini_agent.context.selector import ContextSelector


class ContextBuilder:
    def __init__(self, selector: ContextSelector = None, compressor: ContextCompressor = None, policy: ContextPolicy = None):
        self.policy = policy or ContextPolicy()
        self.selector = selector or ContextSelector(policy=self.policy)
        self.compressor = compressor or ContextCompressor()

    def build(self, items: list[ContextItem], budget: TokenBudget):
        selected = self.selector.select(items, budget)
        
        for item in selected:
            item.content = self.compressor.compress(item.content)
        
        selected = sorted(selected, key=lambda item: item.priority, reverse=True)
        return selected