from dataclasses import dataclass, field
from mini_agent.memory.models import Memory
from mini_agent.context.models import ContextItem

@dataclass
class AgentState:
    question: str = ""
    memories: list[Memory] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    steps: int = 0
    context_items: list[ContextItem] = field(default_factory=list)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)