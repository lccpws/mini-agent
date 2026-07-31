from abc import ABC, abstractmethod
from typing import ClassVar


class TokenCounter(ABC):
    """Token 计数器抽象基类"""

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        pass


class TiktokenCounter(TokenCounter):
    """基于 tiktoken 的精确计数器（OpenAI 模型）"""

    MODEL_TO_ENCODING: ClassVar[dict[str, str]] = {
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4-turbo": "cl100k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }

    def __init__(self, model: str = "gpt-4o"):
        self.encoding_name = self.MODEL_TO_ENCODING.get(model, "cl100k_base")
        self._encoding = None

    def _get_encoding(self):
        if self._encoding is None:
            try:
                import tiktoken
                self._encoding = tiktoken.get_encoding(self.encoding_name)
            except (ImportError, ValueError, OSError):
                self._encoding = False
        return self._encoding

    def count_tokens(self, text: str) -> int:
        encoding = self._get_encoding()
        if encoding is False or encoding is None:
            return max(1, len(text) // 4)
        return len(encoding.encode(text))


class EstimationCounter(TokenCounter):
    """基于字符估算的计数器（回退策略）"""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


class TokenCounterFactory:
    """Token 计数器工厂"""

    TIKTOKEN_MODELS: ClassVar[set[str]] = {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }

    @classmethod
    def create(cls, model: str = "gpt-4o") -> TokenCounter:
        if model in cls.TIKTOKEN_MODELS:
            return TiktokenCounter(model)
        return EstimationCounter()

    @classmethod
    def get_supported_models(cls) -> list[str]:
        return list(cls.TIKTOKEN_MODELS)
