class ContextPolicy:
    """上下文策略，定义不同来源的优先级、是否必须和预算比例"""

    def __init__(self):
        self.rules = {
            "system": {"priority": 1, "required": True, "budget_ratio": 0.3},
            "user": {"priority": 2, "required": True, "budget_ratio": 0.2},
            "tool": {"priority": 3, "required": False, "budget_ratio": 0.1},
            "memory": {"priority": 4, "required": False, "budget_ratio": 0.15},
            "history": {"priority": 5, "required": False, "budget_ratio": 0.15},
            "rag": {"priority": 4, "required": False, "budget_ratio": 0.1},
        }

    def get(self, source: str) -> dict:
        return self.rules.get(source, {"priority": 0, "required": False, "budget_ratio": 0.0})

    def is_required(self, source: str) -> bool:
        return self.rules.get(source, {}).get("required", False)

    def get_priority(self, source: str) -> float:
        return self.rules.get(source, {}).get("priority", 0)

    def get_budget_ratio(self, source: str) -> float:
        return self.rules.get(source, {}).get("budget_ratio", 0.0)

    def get_required_sources(self) -> list[str]:
        return [s for s, r in self.rules.items() if r["required"]]

    def get_optional_sources(self) -> list[str]:
        return [s for s, r in self.rules.items() if not r["required"]]
