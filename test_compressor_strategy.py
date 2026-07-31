"""Compressor Strategy 测试用例"""
from mini_agent.context import (
    ContextCompressor, ContextItem, ContextSource,
    TruncateStrategy, LLMSummaryStrategy
)


def test_truncate_strategy():
    """测试 TruncateStrategy 截断策略"""
    print("=== Test 1: TruncateStrategy 截断策略 ===")

    strategy = TruncateStrategy()
    content = "A" * 1000
    target_tokens = 100

    result = strategy.compress(content, target_tokens)

    result_tokens = max(1, len(result) // 4)
    assert result_tokens <= target_tokens + 10, f"Result tokens {result_tokens} exceeds target {target_tokens}"
    assert result.endswith("\n...[truncated]"), "Should end with truncation marker"
    print(f"  Input: 1000 chars, Target: {target_tokens} tokens")
    print(f"  Output: {len(result)} chars, ~{result_tokens} tokens ✓")
    print()


def test_truncate_compressor():
    """测试 ContextCompressor 使用 TruncateStrategy"""
    print("=== Test 2: ContextCompressor with TruncateStrategy ===")

    compressor = ContextCompressor(strategy=TruncateStrategy())
    item = ContextItem(
        content="B" * 800,
        source=ContextSource.SYSTEM,
        priority=1,
        token_count=200
    )

    result = compressor.compress(item, target_tokens=50)

    assert result.token_count == 50
    assert result.content.endswith("\n...[truncated]")
    print(f"  Input: {item.token_count} tokens")
    print(f"  Output: {result.token_count} tokens ✓")
    print()


def test_compressor_skip_small_item():
    """测试压缩器跳过小 item"""
    print("=== Test 3: Compressor skips small items ===")

    compressor = ContextCompressor(strategy=TruncateStrategy())
    item = ContextItem(
        content="Small content",
        source=ContextSource.USER,
        priority=1,
        token_count=10
    )

    result = compressor.compress(item, target_tokens=50)

    assert result.token_count == 10
    assert result.content == "Small content"
    print(f"  Input: {item.token_count} tokens, Target: 50 tokens")
    print(f"  Skipped (no compression needed) ✓")
    print()


def test_llm_summary_strategy_fallback():
    """测试 LLMSummaryStrategy 无 LLM 时回退到截断"""
    print("=== Test 4: LLMSummaryStrategy fallback to Truncate ===")

    strategy = LLMSummaryStrategy(llm=None)
    content = "C" * 1000
    target_tokens = 100

    result = strategy.compress(content, target_tokens)

    result_tokens = max(1, len(result) // 4)
    assert result_tokens <= target_tokens + 10
    assert result.endswith("\n...[truncated]")
    print(f"  No LLM provided, fallback to truncation ✓")
    print()


def test_llm_summary_strategy_with_mock_llm():
    """测试 LLMSummaryStrategy 使用 mock LLM"""
    print("=== Test 5: LLMSummaryStrategy with mock LLM ===")

    class MockLLM:
        class client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kwargs):
                        class Message:
                            content = "这是通过LLM生成的摘要内容"
                        class Choice:
                            message = Message()
                        class Response:
                            choices = [Choice()]
                        return Response()

        model = "mock-model"

    strategy = LLMSummaryStrategy(llm=MockLLM())
    content = "D" * 1000
    target_tokens = 50

    result = strategy.compress(content, target_tokens)

    assert result == "这是通过LLM生成的摘要内容"
    print(f"  LLM summary generated ✓")
    print(f"  Result: {result}")
    print()


def test_custom_strategy():
    """测试自定义压缩策略"""
    print("=== Test 6: Custom compression strategy ===")

    from mini_agent.context.compressor import CompressionStrategy

    class UpperCaseStrategy(CompressionStrategy):
        def compress(self, content: str, target_tokens: int) -> str:
            truncated = content[:target_tokens * 4]
            return truncated.upper()

    strategy = UpperCaseStrategy()
    compressor = ContextCompressor(strategy=strategy)
    item = ContextItem(
        content="hello world",
        source=ContextSource.MEMORY,
        priority=1,
        token_count=12
    )

    result = compressor.compress(item, target_tokens=2)

    assert result.content == "HELLO WO"
    assert result.token_count == 2
    print(f"  Custom strategy applied ✓")
    print(f"  Result: {result.content}")
    print()


if __name__ == "__main__":
    test_truncate_strategy()
    test_truncate_compressor()
    test_compressor_skip_small_item()
    test_llm_summary_strategy_fallback()
    test_llm_summary_strategy_with_mock_llm()
    test_custom_strategy()

    print("=== All tests passed! ===")
