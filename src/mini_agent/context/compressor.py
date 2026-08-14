from abc import ABC, abstractmethod

from mini_agent.context.models import ContextItem


class CompressionStrategy(ABC):
    """压缩策略抽象类"""

    @abstractmethod
    def compress(self, content: str, target_tokens: int) -> str:
        pass


class TruncateStrategy(CompressionStrategy):
    """截断策略：按比例截断文本"""

    def compress(self, content: str, target_tokens: int) -> str:
        current_tokens = max(1, len(content) // 4)
        ratio = target_tokens / current_tokens
        truncated = content[:int(len(content) * ratio)]
        return truncated + "\n...[truncated]"


class LLMSummaryStrategy(CompressionStrategy):
    """LLM 摘要策略：调用 LLM 生成摘要"""

    def __init__(self, llm=None):
        self.llm = llm

    def compress(self, content: str, target_tokens: int) -> str:
        if self.llm is None or self.llm.client is None:
            return TruncateStrategy().compress(content, target_tokens)

        prompt = f"""请将以下内容压缩到约{target_tokens}个token（约{target_tokens * 4}个字符），保留关键信息。

原始内容：
{content}

要求：
1. 保留核心信息和关键细节
2. 去除冗余和重复内容
3. 使用简洁的语言
4. 保持内容的连贯性"""

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM 摘要失败，使用截断策略: {e}")
            return TruncateStrategy().compress(content, target_tokens)


class ContextCompressor:
    """上下文压缩器，支持多种压缩策略"""

    def __init__(self, strategy: CompressionStrategy = None):
        self.strategy = strategy or TruncateStrategy()

    def compress(self, item: ContextItem, target_tokens: int) -> ContextItem:
        if not item.compressible or item.compressed:
            return item

        current_tokens = item.token_count
        if current_tokens <= target_tokens:
            return item

        item.content = self.strategy.compress(item.content, target_tokens)
        item.token_count = target_tokens
        item.compressed = True
        return item
