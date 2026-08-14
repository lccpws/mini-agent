from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContextSource(Enum):
    """上下文来源"""
    SYSTEM = "system"
    MEMORY = "memory"
    HISTORY = "history"
    TOOL = "tool"
    RAG = "rag"
    USER = "user"


@dataclass
class ContextItem:
    id: str = ""
    content: str = ""
    source: str = "system"
    priority: float = 0.5
    relevance: float = 0.5
    recency: float = 0.5
    reliability: float = 0.5
    token_count: int = 0
    compressible: bool = True
    compressed: bool = False


@dataclass
class ContextRoute:
    """上下文路由结果"""
    needs_system: bool = True
    needs_user: bool = True
    needs_memory: bool = True
    needs_rag: bool = True
    needs_history: bool = True
    reason: str = ""
