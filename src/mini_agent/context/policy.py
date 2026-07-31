from mini_agent.context.models import ContextSource


class ContextPolicy:
    """上下文策略，定义不同来源的优先级、是否必须和预算比例"""

    def __init__(self):
        self.rules = {
            ContextSource.SYSTEM: {"priority": 1, "required": True, "budget_ratio": 0.3},
            ContextSource.USER: {"priority": 2, "required": True, "budget_ratio": 0.2},
            ContextSource.TOOL: {"priority": 3, "required": False, "budget_ratio": 0.1},
            ContextSource.MEMORY: {"priority": 4, "required": False, "budget_ratio": 0.15},
            ContextSource.HISTORY: {"priority": 5, "required": False, "budget_ratio": 0.15},
            ContextSource.RAG: {"priority": 4, "required": False, "budget_ratio": 0.1},
        }

    def get(self, source: ContextSource) -> dict:
        return self.rules[source]

    def is_required(self, source: ContextSource) -> bool:
        return self.rules[source]["required"]

    def get_priority(self, source: ContextSource) -> int:
        return self.rules[source]["priority"]

    def get_budget_ratio(self, source: ContextSource) -> float:
        return self.rules[source]["budget_ratio"]

    def get_required_sources(self) -> list[ContextSource]:
        return [s for s, r in self.rules.items() if r["required"]]

    def get_optional_sources(self) -> list[ContextSource]:
        return [s for s, r in self.rules.items() if not r["required"]]
