from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class MemoryType(Enum):
    """记忆类型枚举"""
    PREFERENCE = "preference"    # 用户偏好
    FACT = "fact"                # 事实信息
    SKILL = "skill"              # 技能能力
    PROJECT = "project"          # 项目相关
    QUESTION = "question"        # 用户问题
    ANSWER = "answer"            # 回答结果
    TOOL_RESULT = "tool_result"  # 工具执行结果
    OBSERVATION = "observation"  # 观察结果


class Importance(Enum):
    """重要性枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3


class MemoryStatus(Enum):
    """记忆状态枚举"""
    ACTIVE = "active"      # 活跃
    ARCHIVED = "archived"  # 已归档
    DELETED = "deleted"    # 已删除（软删除）


@dataclass
class Memory:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    memory_type: MemoryType = MemoryType.FACT
    importance: Importance = Importance.MEDIUM
    status: MemoryStatus = MemoryStatus.ACTIVE
    source: str = "conversation"
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_access: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    def touch(self):
        """更新最后访问时间和访问次数"""
        self.last_access = datetime.now()
        self.access_count += 1

    def archive(self):
        """归档记忆"""
        self.status = MemoryStatus.ARCHIVED

    def restore(self):
        """恢复记忆"""
        self.status = MemoryStatus.ACTIVE

    def delete(self):
        """软删除记忆"""
        self.status = MemoryStatus.DELETED

    def is_active(self) -> bool:
        """检查是否为活跃状态"""
        return self.status == MemoryStatus.ACTIVE

    def is_archived(self) -> bool:
        """检查是否为归档状态"""
        return self.status == MemoryStatus.ARCHIVED

    def __str__(self):
        return f"[{self.memory_type.value}] {self.content}"
