import json
from pathlib import Path
import numpy as np
import faiss

from mini_agent.memory.embedder import Embedder


class ReflectionVectorIndex:
    """FAISS 向量索引，用于 ReflectionMemory 语义检索"""

    def __init__(self, embedder: Embedder, persist_dir: str):
        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "reflection.faiss"
        self.ids_path = self.persist_dir / "reflection_ids.json"

        self.index: faiss.IndexFlatIP | None = None
        self.ids: list[str] = []
        self.dimension: int = 0

        self._load()

    def add(self, record_id: str, text: str):
        """添加记录到索引"""
        vector = self.embedder.embed(text)
        self.dimension = len(vector)
        vector_np = np.array([vector], dtype=np.float32)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        faiss.normalize_L2(vector_np)
        self.index.add(vector_np)
        self.ids.append(record_id)

    def remove(self, record_id: str):
        """从索引中移除记录（重建索引）"""
        if record_id not in self.ids:
            return

        idx = self.ids.index(record_id)
        self.ids.pop(idx)

        if self.index is not None and self.index.ntotal > 0:
            vectors = []
            for i in range(self.index.ntotal):
                vectors.append(self.index.reconstruct(i))
            vectors.pop(idx)

            if vectors:
                vectors_np = np.array(vectors, dtype=np.float32)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(vectors_np)
            else:
                self.index = None

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """语义检索，返回 (record_id, similarity_score)"""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.embedder.embed(query)
        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_np, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.ids):
                results.append((self.ids[idx], float(score)))
        return results

    def save(self):
        """持久化索引"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.ids_path.write_text(json.dumps(self.ids, ensure_ascii=False))

    def _load(self):
        """加载索引"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d

        if self.ids_path.exists():
            self.ids = json.loads(self.ids_path.read_text())

    def count(self) -> int:
        return len(self.ids)
