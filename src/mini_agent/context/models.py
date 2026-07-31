from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContextSource(Enum):
    """上下文来源"""
    SYSTEM = "system"    # 系统
    MEMORY = "memory"    # 记忆
    HISTORY = "history"  # 历史记录
    TOOL = "tool"        # 执行工具
    RAG = "rag"          # RAG
    USER = "user"        # 用户信息

@dataclass
class ContextItem:

    content: str = ""
    source: ContextSource = ContextSource.SYSTEM
    priority: int = 0
    score: float = 1.0
    token_count: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextRoute:
    """上下文路由结果"""
    needs_system: bool = True
    needs_user: bool = True
    needs_memory: bool = True
    needs_rag: bool = True
    needs_history: bool = True
    reason: str = ""