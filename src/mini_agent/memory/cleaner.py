from datetime import datetime, timedelta
from mini_agent.memory.models import Memory, MemoryStatus, Importance
from mini_agent.memory.scorer import MemoryScorer


class MemoryCleaner:
    """记忆清理器，负责归档过期记忆"""

    def __init__(self, max_age_days: int = 50):
        self.max_age_days = max_age_days
        self.scorer = MemoryScorer()

    def clean(self, memories: list[Memory], store=None) -> dict:
        """执行清理，返回统计"""
        archived = []
        retained = []

        for memory in memories:
            if not memory.is_active():
                continue

            if self._should_archive(memory):
                memory.archive()
                archived.append(memory)
            else:
                retained.append(memory)

        # 如果传入了 store，触发保存
        if store is not None:
            store._save()

        return {
            "archived": len(archived),
            "retained": len(retained),
            "archived_memories": archived,
            "retained_memories": retained
        }

    def _should_archive(self, memory: Memory) -> bool:
        """判断是否应该归档"""
        days_inactive = (datetime.now() - memory.last_access).days
        is_low_importance = memory.importance == Importance.LOW
        return days_inactive > self.max_age_days and is_low_importance

    def list_archived(self, memories: list[Memory]) -> list[Memory]:
        """列出所有归档记忆"""
        return [m for m in memories if m.is_archived()]

    def restore(self, memories: list[Memory], memory_id: str, store=None) -> bool:
        """恢复归档的记忆"""
        for memory in memories:
            if memory.id == memory_id and memory.is_archived():
                memory.restore()
                if store is not None:
                    store._save()
                return True
        return False

    def restore_all(self, memories: list[Memory], store=None) -> int:
        """恢复所有归档记忆"""
        count = 0
        for memory in memories:
            if memory.is_archived():
                memory.restore()
                count += 1

        if store is not None and count > 0:
            store._save()

        return count

    def delete(self, memories: list[Memory], memory_id: str, store=None) -> bool:
        """软删除记忆"""
        for memory in memories:
            if memory.id == memory_id and memory.is_active():
                memory.delete()
                if store is not None:
                    store._save()
                return True
        return False

    def get_stats(self, memories: list[Memory]) -> dict:
        """获取记忆统计"""
        active = sum(1 for m in memories if m.is_active())
        archived = sum(1 for m in memories if m.is_archived())
        deleted = sum(1 for m in memories if m.status == MemoryStatus.DELETED)

        return {
            "total": len(memories),
            "active": active,
            "archived": archived,
            "deleted": deleted
        }
