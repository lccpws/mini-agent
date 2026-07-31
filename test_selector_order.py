from mini_agent.context import ContextManager, ContextBuilder, ContextSelector, ContextItem, ContextSource

def test_selector_order():
    selector = ContextSelector()
    
    items = [
        ContextItem(content="System", source=ContextSource.SYSTEM, priority=100, token_count=100),
        ContextItem(content="User", source=ContextSource.USER, priority=90, token_count=50),
        ContextItem(content="Memory", source=ContextSource.MEMORY, priority=50, token_count=30),
        ContextItem(content="History", source=ContextSource.HISTORY, priority=30, token_count=40),
    ]
    
    print("原始items:")
    for item in items:
        print(f"  [{item.source.value}] priority={item.priority}, tokens={item.token_count}")
    
    print("\nSelector排序后 (按priority降序):")
    sorted_items = sorted(items, key=lambda item:(item.priority, item.score), reverse=True)
    for item in sorted_items:
        print(f"  [{item.source.value}] priority={item.priority}, tokens={item.token_count}")
    
    print("\n消费顺序（budget=150）:")
    budget_tokens = 150
    print(f"  Budget: {budget_tokens} tokens")
    consumed = 0
    for item in sorted_items:
        if consumed + item.token_count <= budget_tokens:
            consumed += item.token_count
            print(f"  ✓ 选中 [{item.source.value}] priority={item.priority}, 累计={consumed}")
        else:
            print(f"  ✗ 跳过 [{item.source.value}] priority={item.priority}, 需要={item.token_count}, 剩余={budget_tokens-consumed}")

if __name__ == "__main__":
    test_selector_order()