from mini_agent.context import ContextManager, ContextBuilder, ContextSelector, ContextItem, ContextSource, ContextCompressor

def test_compressor():
    selector = ContextSelector()
    compressor = ContextCompressor()
    builder = ContextBuilder(selector, compressor)
    manager = ContextManager(builder)
    
    long_text = "A" * 5000
    short_text = "B" * 100
    
    items = [
        ContextItem(
            content=long_text,
            source=ContextSource.SYSTEM,
            priority=100,
            token_count=100
        ),
        ContextItem(
            content=short_text,
            source=ContextSource.USER,
            priority=90,
            token_count=50
        ),
    ]
    
    selected = manager.build_context(items, total_tokens=1000, output_tokens=0)
    
    print("测试压缩功能:")
    for item in selected:
        print(f"  [{item.source.value}] priority={item.priority}")
        print(f"    原始长度: {'A' * 5000 if item.source == ContextSource.SYSTEM else 'B' * 100}")
        print(f"    压缩后长度: {len(item.content)}")
        print(f"    内容预览: {item.content[:50]}...")
        print()

if __name__ == "__main__":
    test_compressor()