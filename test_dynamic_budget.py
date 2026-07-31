"""Dynamic Context Budget 测试用例"""
from mini_agent.context import (
    ContextPolicy, ContextSource, ContextManager, ContextItem,
    DynamicTokenBudget, TokenBudget
)


def test_dynamic_budget_calculation():
    """测试 DynamicTokenBudget 预算计算"""
    print("=== Test 1: DynamicTokenBudget 预算计算 ===")
    
    policy = ContextPolicy()
    budget = DynamicTokenBudget(total_tokens=8000, output_tokens=2000, policy=policy)
    
    assert budget.input_budget == 6000, f"Expected 6000, got {budget.input_budget}"
    
    expected = {
        ContextSource.SYSTEM: int(6000 * 0.3),
        ContextSource.USER: int(6000 * 0.2),
        ContextSource.TOOL: int(6000 * 0.1),
        ContextSource.MEMORY: int(6000 * 0.15),
        ContextSource.HISTORY: int(6000 * 0.15),
        ContextSource.RAG: int(6000 * 0.1),
    }
    
    for source, expected_budget in expected.items():
        actual = budget.source_budgets[source]
        assert actual == expected_budget, f"{source}: expected {expected_budget}, got {actual}"
        print(f"  {source.value}: {actual} tokens ✓")
    
    print()


def test_dynamic_budget_consume():
    """测试 DynamicTokenBudget 消耗和限制"""
    print("=== Test 2: DynamicTokenBudget 消耗和限制 ===")
    
    policy = ContextPolicy()
    budget = DynamicTokenBudget(total_tokens=100, output_tokens=0, policy=policy)
    
    system_budget = budget.source_budgets[ContextSource.SYSTEM]
    print(f"  System budget: {system_budget}")
    
    result = budget.consume(ContextSource.SYSTEM, system_budget - 10)
    assert result == True, "Should fit within budget"
    print(f"  Consume {system_budget - 10}: True ✓")
    
    result = budget.consume(ContextSource.SYSTEM, 20)
    assert result == False, "Should exceed budget"
    print(f"  Consume 20 more: False ✓")
    
    remaining = budget.get_remaining(ContextSource.SYSTEM)
    assert remaining == 10, f"Expected 10, got {remaining}"
    print(f"  Remaining: {remaining} ✓")
    
    print()


def test_dynamic_budget_usage():
    """测试 DynamicTokenBudget 使用统计"""
    print("=== Test 3: DynamicTokenBudget 使用统计 ===")
    
    policy = ContextPolicy()
    budget = DynamicTokenBudget(total_tokens=1000, output_tokens=0, policy=policy)
    
    budget.consume(ContextSource.SYSTEM, 100)
    budget.consume(ContextSource.USER, 50)
    budget.consume(ContextSource.MEMORY, 30)
    
    usage = budget.get_usage()
    
    assert usage["system"]["used"] == 100
    assert usage["user"]["used"] == 50
    assert usage["memory"]["used"] == 30
    assert usage["history"]["used"] == 0
    
    print(f"  System: used={usage['system']['used']}, remaining={usage['system']['remaining']} ✓")
    print(f"  User: used={usage['user']['used']}, remaining={usage['user']['remaining']} ✓")
    print(f"  Memory: used={usage['memory']['used']}, remaining={usage['memory']['remaining']} ✓")
    print(f"  History: used={usage['history']['used']}, remaining={usage['history']['remaining']} ✓")
    
    print()


def test_selector_by_source():
    """测试 ContextSelector 按 source 分组选择"""
    print("=== Test 4: ContextSelector 按 source 分组选择 ===")
    
    from mini_agent.context import ContextSelector
    
    selector = ContextSelector()
    policy = ContextPolicy()
    budget = DynamicTokenBudget(total_tokens=1000, output_tokens=0, policy=policy)
    
    items = [
        ContextItem(content="System 1", source=ContextSource.SYSTEM, priority=10, token_count=100),
        ContextItem(content="System 2", source=ContextSource.SYSTEM, priority=5, token_count=100),
        ContextItem(content="User 1", source=ContextSource.USER, priority=8, token_count=50),
        ContextItem(content="Memory 1", source=ContextSource.MEMORY, priority=6, token_count=80),
        ContextItem(content="Memory 2", source=ContextSource.MEMORY, priority=4, token_count=80),
    ]
    
    selected = selector.select(items, budget)
    
    source_counts = {}
    for item in selected:
        src = item.source.value
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print(f"  Selected {len(selected)} items")
    print(f"  Distribution: {source_counts}")
    
    system_items = [i for i in selected if i.source == ContextSource.SYSTEM]
    assert len(system_items) == 2, f"Expected 2 system items, got {len(system_items)}"
    print(f"  System items: {len(system_items)} ✓")
    
    print()


def test_manager_with_dynamic_budget():
    """测试 ContextManager 使用 DynamicTokenBudget"""
    print("=== Test 5: ContextManager 使用 DynamicTokenBudget ===")
    
    policy = ContextPolicy()
    manager = ContextManager(
        policy=policy,
        total_tokens=1000,
        output_tokens=0,
        use_dynamic_budget=True
    )
    
    items = [
        ContextItem(content="System", source=ContextSource.SYSTEM, priority=1, token_count=100),
        ContextItem(content="User", source=ContextSource.USER, priority=2, token_count=50),
        ContextItem(content="Memory 1", source=ContextSource.MEMORY, priority=3, token_count=80),
        ContextItem(content="Memory 2", source=ContextSource.MEMORY, priority=2, token_count=80),
        ContextItem(content="History", source=ContextSource.HISTORY, priority=4, token_count=60),
        ContextItem(content="RAG", source=ContextSource.RAG, priority=5, token_count=70),
    ]
    
    result = manager.build_context(items, total_tokens=1000, output_tokens=0)
    
    source_counts = {}
    for item in result:
        src = item.source.value
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print(f"  Selected {len(result)} items")
    print(f"  Distribution: {source_counts}")
    
    assert ContextSource.SYSTEM.value in source_counts, "System should be selected"
    assert ContextSource.USER.value in source_counts, "User should be selected"
    print(f"  Required sources selected ✓")
    
    print()


def test_custom_policy():
    """测试自定义 Policy"""
    print("=== Test 6: 自定义 Policy ===")
    
    class ResearchPolicy(ContextPolicy):
        def __init__(self):
            super().__init__()
            self.rules[ContextSource.RAG]["budget_ratio"] = 0.5
            self.rules[ContextSource.HISTORY]["budget_ratio"] = 0.3
            self.rules[ContextSource.MEMORY]["budget_ratio"] = 0.1
            self.rules[ContextSource.SYSTEM]["budget_ratio"] = 0.05
            self.rules[ContextSource.USER]["budget_ratio"] = 0.05
    
    policy = ResearchPolicy()
    budget = DynamicTokenBudget(total_tokens=1000, output_tokens=0, policy=policy)
    
    print(f"  RAG budget: {budget.source_budgets[ContextSource.RAG]}")
    print(f"  History budget: {budget.source_budgets[ContextSource.HISTORY]}")
    
    assert budget.source_budgets[ContextSource.RAG] == 500
    assert budget.source_budgets[ContextSource.HISTORY] == 300
    print(f"  Custom budget ratios applied ✓")
    
    manager = ContextManager(policy=policy, total_tokens=1000, output_tokens=0)
    
    items = [
        ContextItem(content="System", source=ContextSource.SYSTEM, priority=1, token_count=50),
        ContextItem(content="User", source=ContextSource.USER, priority=2, token_count=50),
        ContextItem(content="Memory", source=ContextSource.MEMORY, priority=3, token_count=50),
        ContextItem(content="History 1", source=ContextSource.HISTORY, priority=4, token_count=50),
        ContextItem(content="History 2", source=ContextSource.HISTORY, priority=3, token_count=50),
        ContextItem(content="RAG 1", source=ContextSource.RAG, priority=5, token_count=50),
        ContextItem(content="RAG 2", source=ContextSource.RAG, priority=4, token_count=50),
        ContextItem(content="RAG 3", source=ContextSource.RAG, priority=3, token_count=50),
    ]
    
    result = manager.build_context(items, total_tokens=1000, output_tokens=0)
    
    source_counts = {}
    for item in result:
        src = item.source.value
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print(f"  Selected items: {source_counts}")
    print(f"  (RAG should have more items due to higher budget)")
    
    print()


def test_budget_overflow():
    """测试预算溢出处理 - 单个 source 超预算"""
    print("=== Test 7: 预算溢出处理 ===")
    
    policy = ContextPolicy()
    manager = ContextManager(policy=policy, total_tokens=1000, output_tokens=0)
    
    # System budget = 1000 * 0.3 = 300, but items total 500
    items = [
        ContextItem(content="System 1", source=ContextSource.SYSTEM, priority=1, token_count=200),
        ContextItem(content="System 2", source=ContextSource.SYSTEM, priority=2, token_count=200),
        ContextItem(content="System 3", source=ContextSource.SYSTEM, priority=3, token_count=200),
        ContextItem(content="User", source=ContextSource.USER, priority=4, token_count=100),
    ]
    
    result = manager.build_context(items, total_tokens=1000, output_tokens=0)
    
    system_items = [i for i in result if i.source == ContextSource.SYSTEM]
    system_tokens = sum(i.token_count for i in system_items)
    
    print(f"  System budget: {policy.get_budget_ratio(ContextSource.SYSTEM) * 1000}")
    print(f"  System items selected: {len(system_items)}")
    print(f"  System tokens used: {system_tokens}")
    
    assert system_tokens <= 300, f"System tokens {system_tokens} exceeds budget 300"
    print(f"  System within budget ✓")
    
    print()


if __name__ == "__main__":
    test_dynamic_budget_calculation()
    test_dynamic_budget_consume()
    test_dynamic_budget_usage()
    test_selector_by_source()
    test_manager_with_dynamic_budget()
    test_custom_policy()
    test_budget_overflow()
    
    print("=== All tests passed! ===")
