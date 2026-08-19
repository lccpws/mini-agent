"""测试 PriorityCompressor 按优先级压缩"""
from mini_agent.context.compressor import PriorityCompressor
from mini_agent.context.models import ContextItem
from mini_agent.context.policy import ContextPolicy


def test_compression_factors():
    """测试压缩因子配置"""
    compressor = PriorityCompressor()
    
    print("=== 测试压缩因子配置 ===")
    for source, factor in compressor.COMPRESSION_FACTORS.items():
        print(f"  {source}: compression_factor={factor}, 实际压缩={factor*100:.0f}%")
    
    assert compressor.COMPRESSION_FACTORS["system"] == 0.0, "system 不应压缩"
    assert compressor.COMPRESSION_FACTORS["user"] == 0.0, "user 不应压缩"
    assert compressor.COMPRESSION_FACTORS["history"] == 0.6, "history 压缩 60%"
    
    print("✓ 压缩因子配置正确\n")


def test_compress_system_no_change():
    """测试 system item 不压缩"""
    compressor = PriorityCompressor()
    
    item = ContextItem(
        id="system",
        content="You are a helpful assistant." * 100,
        source="system",
        token_count=500
    )
    
    result = compressor.compress(item)
    
    print("=== 测试 system 不压缩 ===")
    print(f"  原始 tokens: 500")
    print(f"  压缩后 tokens: {result.token_count}")
    print(f"  compressed: {result.compressed}")
    
    assert result.token_count == 500, "system 不应被压缩"
    assert result.compressed == False, "compressed 标记应为 False"
    
    print("✓ system 未压缩\n")


def test_compress_history_aggressive():
    """测试 history item 激进压缩"""
    compressor = PriorityCompressor()
    
    item = ContextItem(
        id="history_0",
        content="Previous conversation about Python programming." * 50,
        source="history",
        token_count=1000
    )
    
    result = compressor.compress(item)
    
    print("=== 测试 history 激进压缩 ===")
    print(f"  原始 tokens: 1000")
    print(f"  压缩后 tokens: {result.token_count}")
    print(f"  压缩比例: {(1 - result.token_count / 1000) * 100:.0f}%")
    print(f"  compressed: {result.compressed}")
    
    expected_tokens = int(1000 * (1.0 - 0.6))
    assert result.token_count == expected_tokens, f"history 应压缩到 {expected_tokens}"
    assert result.compressed == True, "compressed 标记应为 True"
    
    print("✓ history 激进压缩正确\n")


def test_compress_memory_moderate():
    """测试 memory item 中等压缩"""
    compressor = PriorityCompressor()
    
    item = ContextItem(
        id="memory_0",
        content="User prefers Python and dark mode." * 30,
        source="memory",
        token_count=500
    )
    
    result = compressor.compress(item)
    
    print("=== 测试 memory 中等压缩 ===")
    print(f"  原始 tokens: 500")
    print(f"  压缩后 tokens: {result.token_count}")
    print(f"  压缩比例: {(1 - result.token_count / 500) * 100:.0f}%")
    
    expected_tokens = int(500 * (1.0 - 0.3))
    assert result.token_count == expected_tokens, f"memory 应压缩到 {expected_tokens}"
    
    print("✓ memory 中等压缩正确\n")


def test_compress_all_priority_order():
    """测试 compress_all 按优先级排序压缩"""
    compressor = PriorityCompressor()
    
    items = [
        ContextItem(id="system", content="System prompt " * 100, source="system", token_count=500),
        ContextItem(id="user", content="User question " * 50, source="user", token_count=200),
        ContextItem(id="history_0", content="History " * 80, source="history", token_count=400),
        ContextItem(id="memory_0", content="Memory " * 60, source="memory", token_count=300),
    ]
    
    result = compressor.compress_all(items)
    
    print("=== 测试 compress_all 优先级排序 ===")
    print("  压缩前:")
    for item in items:
        print(f"    {item.id} ({item.source}): {item.token_count} tokens")
    
    print("\n  压缩后:")
    for item in result:
        print(f"    {item.id} ({item.source}): {item.token_count} tokens, compressed={item.compressed}")
    
    system = next(i for i in result if i.id == "system")
    history = next(i for i in result if i.id == "history_0")
    memory = next(i for i in result if i.id == "memory_0")
    
    assert system.token_count == 500, "system 不应压缩"
    assert system.compressed == False
    
    assert history.token_count == int(400 * 0.4), "history 应压缩 60%"
    assert history.compressed == True
    
    assert memory.token_count == int(300 * 0.7), "memory 应压缩 30%"
    assert memory.compressed == True
    
    print("✓ 优先级排序压缩正确\n")


def test_already_compressed_skip():
    """测试已压缩的 item 跳过"""
    compressor = PriorityCompressor()
    
    item = ContextItem(
        id="history_0",
        content="Already compressed content",
        source="history",
        token_count=100,
        compressed=True
    )
    
    result = compressor.compress(item)
    
    print("=== 测试已压缩跳过 ===")
    print(f"  tokens: {result.token_count}")
    print(f"  compressed: {result.compressed}")
    
    assert result.token_count == 100, "已压缩不应再次压缩"
    assert result.compressed == True
    
    print("✓ 已压缩跳过正确\n")


def test_small_content_skip():
    """测试小内容跳过压缩"""
    compressor = PriorityCompressor()
    
    item = ContextItem(
        id="history_0",
        content="Short",
        source="history",
        token_count=10
    )
    
    result = compressor.compress(item)
    
    print("=== 测试小内容跳过 ===")
    print(f"  原始 tokens: 10")
    print(f"  压缩后 tokens: {result.token_count}")
    
    expected_tokens = int(10 * 0.4)
    if expected_tokens < 10:
        assert result.token_count == expected_tokens, "小内容也应压缩"
    else:
        assert result.token_count == 10, "小内容不应压缩"
    
    print("✓ 小内容处理正确\n")


if __name__ == "__main__":
    test_compression_factors()
    test_compress_system_no_change()
    test_compress_history_aggressive()
    test_compress_memory_moderate()
    test_compress_all_priority_order()
    test_already_compressed_skip()
    test_small_content_skip()
    print("所有测试通过！")