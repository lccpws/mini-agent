from mini_agent.context import ContextManager, ContextBuilder, ContextSelector, ContextItem, ContextSource

def test_budget_filtering():
    selector = ContextSelector()
    builder = ContextBuilder(selector)
    manager = ContextManager(builder)
    
    items = [
        ContextItem(
            content="System prompt with very long content " * 10,
            source=ContextSource.SYSTEM,
            priority=100,
            token_count=200
        ),
        ContextItem(
            content="User question",
            source=ContextSource.USER,
            priority=90,
            token_count=50
        ),
        ContextItem(
            content="Memory 1",
            source=ContextSource.MEMORY,
            priority=50,
            token_count=30
        ),
        ContextItem(
            content="Memory 2",
            source=ContextSource.MEMORY,
            priority=49,
            token_count=30
        ),
        ContextItem(
            content="Memory 3",
            source=ContextSource.MEMORY,
            priority=48,
            token_count=30
        ),
        ContextItem(
            content="History 1",
            source=ContextSource.HISTORY,
            priority=30,
            token_count=40
        ),
        ContextItem(
            content="History 2",
            source=ContextSource.HISTORY,
            priority=29,
            token_count=40
        ),
    ]
    
    total_tokens = sum(item.token_count for item in items)
    print(f"Total tokens needed: {total_tokens}")
    print(f"Budget: 300 tokens")
    print()
    
    selected = manager.build_context(items, total_tokens=300, output_tokens=0)
    
    selected_tokens = sum(item.token_count for item in selected)
    print(f"Selected {len(selected)}/{len(items)} items, using {selected_tokens}/300 tokens")
    print()
    
    print("Selected items:")
    for item in selected:
        print(f"  [{item.source.value}] priority={item.priority}, tokens={item.token_count}")
        print(f"    内容: {item.content[:30]}...")
        print()

if __name__ == "__main__":
    test_budget_filtering()