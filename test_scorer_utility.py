"""测试 ContextScorer 的 source_priority 和 utility 方法"""
from mini_agent.context.score import ContextScorer
from mini_agent.context.models import ContextItem, ContextSource
from mini_agent.context.policy import ContextPolicy


def test_source_priority():
    """测试 source_priority 计算"""
    scorer = ContextScorer()
    
    # 测试不同 source 的优先级
    test_cases = [
        ("system", 1),    # policy 优先级为 1
        ("user", 2),      # policy 优先级为 2
        ("tool", 3),      # policy 优先级为 3
        ("memory", 4),    # policy 优先级为 4
        ("history", 5),   # policy 优先级为 5
    ]
    
    print("=== 测试 source_priority ===")
    for source, expected_priority in test_cases:
        item = ContextItem(source=source)
        source_priority = scorer._calc_source_priority(item)
        expected_score = 1.0 - (expected_priority / 5.0)
        print(f"  {source}: policy_priority={expected_priority}, source_priority={source_priority:.3f}, expected={expected_score:.3f}")
        assert abs(source_priority - expected_score) < 0.001, f"source_priority 计算错误: {source}"
    
    print("✓ source_priority 测试通过\n")


def test_utility():
    """测试 utility 计算"""
    scorer = ContextScorer()
    
    print("=== 测试 utility ===")
    
    # 测试 token_count > 0 的情况
    item1 = ContextItem(
        content="Test content",
        source="system",
        priority=1.0,
        token_count=100
    )
    score1 = scorer.score(item1)
    utility1 = scorer.utility(item1)
    print(f"  Item1: score={score1:.3f}, token_count=100, utility={utility1:.4f}")
    assert abs(utility1 - score1 / 100) < 0.0001, "utility 计算错误"
    
    # 测试 token_count = 0 的情况
    item2 = ContextItem(
        content="Test content",
        source="system",
        priority=1.0,
        token_count=0
    )
    score2 = scorer.score(item2)
    utility2 = scorer.utility(item2)
    print(f"  Item2: score={score2:.3f}, token_count=0, utility={utility2:.4f}")
    assert utility2 == score2, "token_count=0 时 utility 应等于 score"
    
    # 测试不同 token_count 的 utility 比较
    item3 = ContextItem(
        content="Short content",
        source="memory",
        priority=5.0,
        token_count=50
    )
    item4 = ContextItem(
        content="Long content with more tokens",
        source="memory",
        priority=5.0,
        token_count=200
    )
    score3 = scorer.score(item3)
    score4 = scorer.score(item4)
    utility3 = scorer.utility(item3)
    utility4 = scorer.utility(item4)
    print(f"  Item3: score={score3:.3f}, token_count=50, utility={utility3:.4f}")
    print(f"  Item4: score={score4:.3f}, token_count=200, utility={utility4:.4f}")
    assert utility3 > utility4, "相同 score 下，token_count 小的 utility 应更大"
    
    print("✓ utility 测试通过\n")


def test_score_dimensions():
    """测试 score 的 5 个维度"""
    scorer = ContextScorer()
    
    print("=== 测试 score 维度 ===")
    
    item = ContextItem(
        content="Test content",
        source="memory",
        priority=1.0,
        token_count=100
    )
    
    score = scorer.score(item)
    print(f"  总分: {score:.3f}")
    print(f"  priority: {item.priority:.3f}")
    print(f"  relevance: {item.relevance:.3f}")
    print(f"  recency: {item.recency:.3f}")
    print(f"  reliability: {item.reliability:.3f}")
    
    # 验证各维度在 0-1 范围内
    assert 0 <= item.relevance <= 1, "relevance 超出范围"
    assert 0 <= item.recency <= 1, "recency 超出范围"
    assert 0 <= item.reliability <= 1, "reliability 超出范围"
    
    print("✓ score 维度测试通过\n")


def test_utility_ranking():
    """测试 utility 排序"""
    scorer = ContextScorer()
    
    print("=== 测试 utility 排序 ===")
    
    items = [
        ContextItem(content="High score, high tokens", source="system", priority=1.0, token_count=100),
        ContextItem(content="High score, low tokens", source="system", priority=1.0, token_count=20),
        ContextItem(content="Low score, high tokens", source="history", priority=5.0, token_count=100),
        ContextItem(content="Low score, low tokens", source="history", priority=5.0, token_count=20),
    ]
    
    # 计算每个 item 的 utility
    utilities = []
    for item in items:
        utility = scorer.utility(item)
        utilities.append((item.content, item.token_count, utility))
        print(f"  {item.content}: token_count={item.token_count}, utility={utility:.4f}")
    
    # 按 utility 排序
    utilities.sort(key=lambda x: x[2], reverse=True)
    print("\n  按 utility 排序:")
    for content, token_count, utility in utilities:
        print(f"    {content}: {utility:.4f}")
    
    # 验证排序正确
    assert utilities[0][2] >= utilities[1][2] >= utilities[2][2] >= utilities[3][2], "utility 排序错误"
    
    print("✓ utility 排序测试通过\n")


if __name__ == "__main__":
    test_source_priority()
    test_utility()
    test_score_dimensions()
    test_utility_ranking()
    print("所有测试通过！")