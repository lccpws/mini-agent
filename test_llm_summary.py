"""LLM Summary 策略测试 - 使用 .env 配置"""
import os
from dotenv import load_dotenv
from openai import OpenAI
from mini_agent.context import (
    ContextCompressor, ContextItem, ContextSource, LLMSummaryStrategy
)


class SimpleLLM:
    """简单的 LLM 封装，用于测试"""

    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")


def test_llm_summary_with_real_api():
    """测试 LLM Summary 策略（使用真实 API）"""
    print("=== Test: LLM Summary with Real API ===")

    llm = SimpleLLM()
    strategy = LLMSummaryStrategy(llm=llm)
    compressor = ContextCompressor(strategy=strategy)

    long_content = """
    人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能才能完成的任务的系统。

    机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需显式编程。深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的模式识别任务。

    自然语言处理（NLP）是人工智能的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。大语言模型（LLM）如GPT系列是NLP领域的重大突破。

    计算机视觉是AI的第三个主要分支，使计算机能够从图像和视频中提取有意义的信息。这包括图像识别、目标检测、图像分割等任务。

    强化学习是机器学习的第四种主要方法，智能体通过与环境交互来学习最优行为策略。AlphaGo和ChatGPT等突破性成果都利用了强化学习技术。
    """

    item = ContextItem(
        content=long_content,
        source=ContextSource.RAG,
        priority=5,
        token_count=len(long_content) // 4
    )

    original_tokens = item.token_count
    target_tokens = 100

    print(f"  Original: {original_tokens} tokens")
    print(f"  Target: {target_tokens} tokens")
    print()

    result = compressor.compress(item, target_tokens)

    print(f"  Compressed: {result.token_count} tokens")
    print(f"  Reduction: {original_tokens - result.token_count} tokens ({(original_tokens - result.token_count) / original_tokens * 100:.1f}%)")
    print()
    print("  Compressed content:")
    print(f"  {result.content[:200]}...")
    print()


def test_llm_summary_vs_truncate():
    """对比 LLM Summary 和 Truncate 策略"""
    print("=== Test: LLM Summary vs Truncate ===")

    llm = SimpleLLM()

    from mini_agent.context import TruncateStrategy

    long_content = """
    机器学习是人工智能的一个子领域，它使计算机能够从数据中学习，而无需显式编程。深度学习是机器学习的一个子集，使用多层神经网络来处理复杂的模式识别任务。

    自然语言处理（NLP）是人工智能的另一个重要分支，专注于使计算机能够理解、解释和生成人类语言。大语言模型（LLM）如GPT系列是NLP领域的重大突破。

    计算机视觉是AI的第三个主要分支，使计算机能够从图像和视频中提取有意义的信息。这包括图像识别、目标检测、图像分割等任务。
    """

    target_tokens = 80

    # Truncate strategy
    truncate_strategy = TruncateStrategy()
    truncate_result = truncate_strategy.compress(long_content, target_tokens)

    # LLM summary strategy
    llm_strategy = LLMSummaryStrategy(llm=llm)
    llm_result = llm_strategy.compress(long_content, target_tokens)

    print(f"  Target: {target_tokens} tokens")
    print()
    print("  Truncate result:")
    print(f"  {truncate_result[:150]}...")
    print()
    print("  LLM Summary result:")
    print(f"  {llm_result[:150]}...")
    print()


if __name__ == "__main__":
    test_llm_summary_with_real_api()
    test_llm_summary_vs_truncate()

    print("=== All LLM tests completed! ===")
