"""测试 ContextResolver 去重/冲突解决功能"""
from mini_agent.context.resolver import ContextResolver
from mini_agent.context.models import ContextItem


def test_resolve_basic():
    """测试基本去重功能"""
    resolver = ContextResolver()
    
    items = [
        ContextItem(id="mem1", content="Version 1", source="memory", reliability=0.5, recency=0.3),
        ContextItem(id="mem1", content="Version 2", source="memory", reliability=0.8, recency=0.6),
        ContextItem(id="mem2", content="Unique item", source="system", reliability=0.9, recency=0.9),
    ]
    
    result = resolver.resolve(items)
    
    print("=== 测试基本去重 ===")
    print(f"输入: {len(items)} items")
    print(f"输出: {len(result)} items")
    
    assert len(result) == 2, "应该去重为 2 个 items"
    
    mem_items = [item for item in result if item.id == "mem1"]
    assert len(mem_items) == 1, "mem1 应该只有一个 item"
    assert mem_items[0].content == "Version 2", "应该选择 reliability 和 recency 更高的版本"
    
    print("✓ 基本去重测试通过\n")


def test_resolve_priority():
    """测试优先级计算"""
    resolver = ContextResolver()
    
    items = [
        ContextItem(id="item1", content="Low priority source", source="history", reliability=0.9, recency=0.9),
        ContextItem(id="item1", content="High priority source", source="system", reliability=0.5, recency=0.5),
    ]
    
    result = resolver.resolve(items)
    
    print("=== 测试优先级计算 ===")
    for item in items:
        priority = resolver._priority(item)
        print(f"  {item.source}: {item.content} -> priority={priority:.1f}")
    
    assert len(result) == 1, "应该去重为 1 个 item"
    assert result[0].source == "system", "应该选择 system source（更高优先级）"
    
    print("✓ 优先级计算测试通过\n")


def test_resolve_no_duplicates():
    """测试无重复的情况"""
    resolver = ContextResolver()
    
    items = [
        ContextItem(id="item1", content="First", source="system"),
        ContextItem(id="item2", content="Second", source="user"),
        ContextItem(id="item3", content="Third", source="memory"),
    ]
    
    result = resolver.resolve(items)
    
    print("=== 测试无重复 ===")
    print(f"输入: {len(items)} items")
    print(f"输出: {len(result)} items")
    
    assert len(result) == 3, "无重复时应该返回所有 items"
    
    print("✓ 无重复测试通过\n")


def test_resolve_empty():
    """测试空列表"""
    resolver = ContextResolver()
    
    result = resolver.resolve([])
    
    print("=== 测试空列表 ===")
    print(f"输出: {len(result)} items")
    
    assert len(result) == 0, "空列表应该返回空结果"
    
    print("✓ 空列表测试通过\n")


def test_resolve_multiple_duplicates():
    """测试多个重复"""
    resolver = ContextResolver()
    
    items = [
        ContextItem(id="item1", content="V1", source="memory", reliability=0.3, recency=0.2),
        ContextItem(id="item1", content="V2", source="memory", reliability=0.7, recency=0.8),
        ContextItem(id="item1", content="V3", source="memory", reliability=0.5, recency=0.5),
        ContextItem(id="item2", content="V1", source="system", reliability=0.9, recency=0.9),
        ContextItem(id="item2", content="V2", source="system", reliability=0.6, recency=0.6),
    ]
    
    result = resolver.resolve(items)
    
    print("=== 测试多个重复 ===")
    print(f"输入: {len(items)} items")
    print(f"输出: {len(result)} items")
    
    assert len(result) == 2, "应该去重为 2 个 items"
    
    item1 = next(item for item in result if item.id == "item1")
    assert item1.content == "V2", "item1 应该选择 V2（最高 reliability + recency）"
    
    item2 = next(item for item in result if item.id == "item2")
    assert item2.content == "V1", "item2 应该选择 V1"
    
    print("✓ 多个重复测试通过\n")


if __name__ == "__main__":
    test_resolve_basic()
    test_resolve_priority()
    test_resolve_no_duplicates()
    test_resolve_empty()
    test_resolve_multiple_duplicates()
    print("所有测试通过！")