from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import time


class Embedder:
    """Embedding 服务，将文本转换为向量"""

    def __init__(self, model: str = "text-embedding-3-small", max_retries: int = 3):
        env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
        load_dotenv(env_path)
        api_key = os.getenv("OPENAI_EMBEDDING_API_KEY")
        base_url = os.getenv("OPENAI_EMBEDDING_BASE_URL")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        ) if api_key else None
        self.model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.max_retries = max_retries

    def embed(self, text: str) -> list[float]:
        """单条文本向量化（带重试）"""
        if self.client is None:
            return self._mock_embed(text)

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Embedding 失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量向量化（带重试）"""
        if self.client is None:
            return [self._mock_embed(text) for text in texts]

        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                print(f"Batch Embedding 失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(1)
                else:
                    raise

    def _mock_embed(self, text: str) -> list[float]:
        """Mock 向量化（用于测试）"""
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        import random
        random.seed(hash_val)
        return [random.random() for _ in range(10)]
