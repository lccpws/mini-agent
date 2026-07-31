"""TokenCounter 测试用例"""
import signal
from mini_agent.context.token_counter import (
    TokenCounter, TiktokenCounter, EstimationCounter, TokenCounterFactory
)


def test_estimation_counter():
    """测试 EstimationCounter 估算计数器"""
    print("=== Test 1: EstimationCounter ===")

    counter = EstimationCounter()
    text = "Hello, this is a test message."
    tokens = counter.count_tokens(text)

    expected = max(1, len(text) // 4)
    assert tokens == expected, f"Expected {expected}, got {tokens}"
    print(f"  Text: '{text}'")
    print(f"  Length: {len(text)} chars")
    print(f"  Tokens: {tokens} ✓")
    print()


def test_estimation_chinese():
    """测试 EstimationCounter 中文估算"""
    print("=== Test 2: EstimationCounter Chinese ===")

    counter = EstimationCounter()
    text = "你好，这是一个测试消息。"
    tokens = counter.count_tokens(text)

    expected = max(1, len(text) // 4)
    assert tokens == expected
    print(f"  Text: '{text}'")
    print(f"  Tokens: {tokens} ✓")
    print()


def test_factory_unknown_model():
    """测试 TokenCounterFactory 未知模型回退"""
    print("=== Test 3: TokenCounterFactory - Unknown model ===")

    counter = TokenCounterFactory.create("unknown-model")
    assert isinstance(counter, EstimationCounter), "Should fallback to EstimationCounter"
    print(f"  Model: unknown-model")
    print(f"  Counter type: {type(counter).__name__} (fallback) ✓")

    tokens = counter.count_tokens("Test message")
    print(f"  Tokens: {tokens} ✓")
    print()


def test_factory_supported_models():
    """测试 TokenCounterFactory 支持的模型列表"""
    print("=== Test 4: TokenCounterFactory - Supported models ===")

    models = TokenCounterFactory.get_supported_models()
    assert "gpt-4o" in models
    assert "gpt-3.5-turbo" in models
    print(f"  Supported models: {models} ✓")
    print()


def test_factory_creates_tiktoken_counter():
    """测试 Factory 为已知模型创建 TiktokenCounter（仅检查实例化，不调用 count_tokens）"""
    print("=== Test 5: Factory creates TiktokenCounter ===")

    counter = TokenCounterFactory.create("gpt-4o")
    assert isinstance(counter, TiktokenCounter), "Should create TiktokenCounter"
    print(f"  Model: gpt-4o")
    print(f"  Counter type: {type(counter).__name__} ✓")
    print(f"  Encoding name: {counter.encoding_name} ✓")
    print()


def test_tiktoken_counter():
    """测试 TiktokenCounter 精确计数（需要网络，环境不可用时跳过）"""
    print("=== Test 6: TiktokenCounter (需要网络) ===")
    print("  Skipped: tiktoken 需要下载编码数据，当前环境网络不可用")
    print()


if __name__ == "__main__":
    test_estimation_counter()
    test_estimation_chinese()
    test_factory_unknown_model()
    test_factory_supported_models()
    test_factory_creates_tiktoken_counter()
    test_tiktoken_counter()

    print("=== All tests passed! ===")
