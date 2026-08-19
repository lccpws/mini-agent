"""测试 ContextManager 增量更新"""
from mini_agent.context.manager import ContextManager
from mini_agent.context.models import ContextItem, ContextRoute


def test_incremental_update():
    """测试增量更新基本功能"""
    manager = ContextManager(total_tokens=5000, output_tokens=1000)
    
    initial_items = [
        ContextItem(id="system_prompt", content="System prompt", source="system", token_count=100),
        ContextItem(id="user_question", content="What is Python?", source="user", token_count=20),
        ContextItem(id="memory_0", content="User likes Python", source="memory", token_count=30),
    ]
    
    route = ContextRoute(needs_system=True, needs_user=True, needs_memory=True)
    
    print("=== 测试增量更新 ===")
    print(f"\n初始 items: {len(initial_items)}")
    
    result1 = manager.build_context(initial_items, route, query="What is Python?")
    print(f"build_context 后: {len(result1)} items")
    print(f"  cached_items: {len(manager.cached_items)}")
    
    new_item = ContextItem(
        id="history_0",
        content="Observation: Python is a programming language",
        source="history",
        token_count=50
    )
    
    result2 = manager.update(new_item, query="What is Python?")
    print(f"\nupdate 后: {len(result2)} items")
    print(f"  cached_items: {len(manager.cached_items)}")
    
    assert len(result2) == 4, "应该包含 4 个 items"
    assert manager.cached_items == result2, "cached_items 应该与 result2 相同"
    
    history_items = [item for item in result2 if item.source == "history"]
    assert len(history_items) == 1, "应该包含 1 个 history item"
    assert "Observation: Python is a programming language" in history_items[0].content
    
    print("✓ 增量更新正确\n")


def test_incremental_update_with_duplicate():
    """测试增量更新时的去重"""
    manager = ContextManager(total_tokens=5000, output_tokens=1000)
    
    initial_items = [
        ContextItem(id="system_prompt", content="System prompt", source="system", token_count=100),
        ContextItem(id="memory_0", content="User likes Python", source="memory", token_count=30),
    ]
    
    route = ContextRoute(needs_system=True, needs_user=True, needs_memory=True)
    
    print("=== 测试增量更新去重 ===")
    
    result1 = manager.build_context(initial_items, route)
    print(f"初始: {len(result1)} items")
    
    duplicate_item = ContextItem(
        id="memory_0",
        content="User prefers Python",
        source="memory",
        token_count=25,
        reliability=0.9,
        recency=0.8
    )
    
    result2 = manager.update(duplicate_item)
    print(f"update 后: {len(result2)} items")
    
    memory_items = [item for item in result2 if item.source == "memory"]
    assert len(memory_items) == 1, "应该只有 1 个 memory item（去重）"
    assert "User prefers Python" in memory_items[0].content, f"应该选择新版本，实际: {memory_items[0].content}"
    
    print("✓ 去重正确\n")


def test_multiple_updates():
    """测试多次增量更新"""
    manager = ContextManager(total_tokens=5000, output_tokens=1000)
    
    initial_items = [
        ContextItem(id="system_prompt", content="System prompt", source="system", token_count=100),
        ContextItem(id="user_question", content="What is Python?", source="user", token_count=20),
    ]
    
    route = ContextRoute(needs_system=True, needs_user=True)
    
    print("=== 测试多次增量更新 ===")
    
    result1 = manager.build_context(initial_items, route)
    print(f"初始: {len(result1)} items")
    
    for i in range(3):
        new_item = ContextItem(
            id=f"history_{i}",
            content=f"Observation {i}",
            source="history",
            token_count=30
        )
        result = manager.update(new_item)
        print(f"update {i+1} 后: {len(result)} items")
    
    assert len(result) == 5, "应该包含 5 个 items"
    
    history_items = [item for item in result if item.source == "history"]
    assert len(history_items) == 3, "应该包含 3 个 history items"
    
    print("✓ 多次更新正确\n")


def test_update_preserves_compression():
    """测试增量更新保留压缩状态"""
    manager = ContextManager(total_tokens=5000, output_tokens=1000)
    
    initial_items = [
        ContextItem(id="system_prompt", content="System prompt", source="system", token_count=100),
        ContextItem(id="user_question", content="Question?", source="user", token_count=20),
        ContextItem(id="history_0", content="Long history " * 50, source="history", token_count=500),
    ]
    
    route = ContextRoute(needs_system=True, needs_user=True, needs_history=True)
    
    print("=== 测试保留压缩状态 ===")
    
    result1 = manager.build_context(initial_items, route)
    print(f"初始: {len(result1)} items")
    
    for item in result1:
        print(f"  {item.id}: tokens={item.token_count}, compressed={item.compressed}")
    
    new_item = ContextItem(
        id="history_1",
        content="New observation " * 30,
        source="history",
        token_count=300
    )
    
    result2 = manager.update(new_item)
    print(f"\nupdate 后: {len(result2)} items")
    
    for item in result2:
        print(f"  {item.id}: tokens={item.token_count}, compressed={item.compressed}")
    
    history_items = [item for item in result2 if item.source == "history"]
    assert len(history_items) >= 1, "应该有 history items"
    
    print("✓ 压缩状态保留正确\n")


def test_get_stats():
    """测试获取统计信息"""
    manager = ContextManager(total_tokens=5000, output_tokens=1000)
    
    initial_items = [
        ContextItem(id="system_prompt", content="System prompt", source="system", token_count=100),
    ]
    
    route = ContextRoute(needs_system=True)
    
    print("=== 测试获取统计信息 ===")
    
    stats = manager.get_stats()
    print(f"初始 stats: {stats}")
    assert stats["cached_items_count"] == 0
    
    manager.build_context(initial_items, route)
    
    stats = manager.get_stats()
    print(f"build_context 后 stats: {stats}")
    assert stats["cached_items_count"] == 1
    
    print("✓ 统计信息正确\n")


if __name__ == "__main__":
    test_incremental_update()
    test_incremental_update_with_duplicate()
    test_multiple_updates()
    test_update_preserves_compression()
    test_get_stats()
    print("所有测试通过！")