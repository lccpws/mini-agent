from abc import ABC, abstractmethod

from mini_agent.context.models import ContextItem
from mini_agent.context.policy import ContextPolicy


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


class PriorityCompressor:
    """按优先级压缩，低优先级先压缩"""

    COMPRESSION_FACTORS = {
        "system": 0.0,
        "user": 0.0,
        "tool": 0.2,
        "memory": 0.3,
        "rag": 0.4,
        "history": 0.6,
    }

    def __init__(
        self,
        strategy: CompressionStrategy = None,
        policy: ContextPolicy = None,
        compression_factors: dict[str, float] = None,
    ):
        self.strategy = strategy or TruncateStrategy()
        self.policy = policy or ContextPolicy()
        self.compression_factors = compression_factors or self.COMPRESSION_FACTORS

    def compress(self, item: ContextItem) -> ContextItem:
        """根据优先级压缩单个 item"""
        if not item.compressible or item.compressed:
            return item

        source = item.source.value if hasattr(item.source, 'value') else item.source
        compression_factor = self.compression_factors.get(source, 0.3)

        if compression_factor == 0.0:
            return item

        target_tokens = max(1, int(item.token_count * (1.0 - compression_factor)))

        if item.token_count <= target_tokens:
            return item

        item.content = self.strategy.compress(item.content, target_tokens)
        item.token_count = target_tokens
        item.compressed = True
        return item

    def compress_all(self, items: list[ContextItem]) -> list[ContextItem]:
        """按优先级排序后压缩所有 items（低优先级先压缩）"""
        sorted_items = sorted(
            items,
            key=lambda x: self._get_priority(x),
            reverse=False
        )

        result = []
        for item in sorted_items:
            compressed = self.compress(item)
            result.append(compressed)

        return result

    def _get_priority(self, item: ContextItem) -> float:
        """获取 item 的优先级数值（用于排序）"""
        source = item.source.value if hasattr(item.source, 'value') else item.source
        return self.policy.get_priority(source)
