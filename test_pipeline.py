"""展示 ContextManager.build_context 的完整 pipeline"""
from mini_agent.context.manager import ContextManager
from mini_agent.context.models import ContextItem, ContextRoute


def print_items(title: str, items: list[ContextItem]):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if not items:
        print("  (空)")
        return
    for i, item in enumerate(items):
        source = item.source.value if hasattr(item.source, 'value') else item.source
        compressed = "[compressed]" if item.compressed else ""
        print(f"  [{i}] id={item.id}, source={source}, priority={item.priority:.2f}, tokens={item.token_count} {compressed}")
        print(f"      content: {item.content[:50]}...")


def main():
    from mini_agent.context.policy import ContextPolicy
    
    policy = ContextPolicy()
    manager = ContextManager(policy=policy)
    
    items = [
        ContextItem(
            id="system_prompt",
            content="You are a helpful assistant. " * 50,
            source="system",
            priority=1.0,
            token_count=500
        ),
        ContextItem(
            id="user_question",
            content="问题：What is Python?",
            source="user",
            priority=0.9,
            token_count=20
        ),
        ContextItem(
            id="memory_0",
            content="记忆：User likes Python programming. " * 20,
            source="memory",
            priority=0.5,
            token_count=400
        ),
        ContextItem(
            id="memory_1",
            content="记忆：User prefers dark mode and VS Code. " * 15,
            source="memory",
            priority=0.49,
            token_count=350
        ),
        ContextItem(
            id="memory_0",
            content="记忆：User is a developer",  # 重复 id
            source="memory",
            priority=0.5,
            token_count=25
        ),
        ContextItem(
            id="rag_001",
            content="[知识库:python.md] Python is a programming language. " * 25,
            source="rag",
            priority=0.4,
            token_count=600
        ),
        ContextItem(
            id="history_0",
            content="观察：User asked about weather and Python. " * 30,
            source="history",
            priority=0.3,
            token_count=700
        ),
    ]
    
    route = ContextRoute(
        needs_system=True,
        needs_user=True,
        needs_memory=True,
        needs_rag=True,
        needs_history=True,
    )
    
    total_input_tokens = sum(item.token_count for item in items)
    print_items(f"Step 0: 输入 items (总 tokens={total_input_tokens})", items)
    print(f"\n  Route: memory={route.needs_memory}, rag={route.needs_rag}, history={route.needs_history}")
    
    # Step 1: filter_by_route
    filtered = manager._filter_by_route(items, route)
    print_items("Step 1: filter_by_route (过滤不需要的来源)", filtered)
    
    # Step 2: _apply_policy
    policy_applied = manager._apply_policy(filtered)
    print_items("Step 2: _apply_policy (根据 policy 调整 priority)", policy_applied)
    
    # Step 3: resolver.resolve
    resolved = manager.resolver.resolve(policy_applied)
    print_items("Step 3: resolver.resolve (按 id 去重)", resolved)
    print(f"\n  去重: {len(policy_applied)} items -> {len(resolved)} items")
    
    # Step 4: selector.select
    from mini_agent.context.budget import TokenBudget
    budget = TokenBudget(5000, 500)
    
    print(f"\n  Token budget: input={budget.input_budget}")
    
    selected = manager.selector.select(resolved, budget)
    print_items("Step 4: selector.select (按 utility 选择)", selected)
    
    # Step 5: priority_compressor.compress_all
    compressed = manager.priority_compressor.compress_all(selected)
    print_items("Step 5: priority_compressor.compress_all (按优先级压缩)", compressed)
    
    # 显示压缩统计
    print("\n  压缩统计:")
    for orig, comp in zip(selected, compressed):
        if orig.token_count != comp.token_count:
            reduction = (1 - comp.token_count / orig.token_count) * 100
            print(f"    {comp.id} ({comp.source}): {orig.token_count} -> {comp.token_count} tokens (压缩 {reduction:.0f}%)")
        else:
            print(f"    {comp.id} ({comp.source}): {orig.token_count} tokens (未压缩)")
    
    # Step 6: final sort
    final = sorted(compressed, key=lambda item: manager.scorer.utility(item), reverse=True)
    total_output_tokens = sum(item.token_count for item in final)
    print_items(f"Step 6: 按 utility 排序 (最终输出, 总 tokens={total_output_tokens})", final)
    
    print(f"\n{'='*60}")
    print(f"  Pipeline 完成: {len(items)} items -> {len(final)} items")
    print(f"  Token 节省: {total_input_tokens} -> {total_output_tokens} ({(1-total_output_tokens/total_input_tokens)*100:.0f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()