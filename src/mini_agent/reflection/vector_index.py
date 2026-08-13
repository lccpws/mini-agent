import json
from pathlib import Path
from datetime import datetime
import numpy as np
import faiss

from mini_agent.memory.embedder import Embedder


class VectorRecord:
    """向量记录，包含原始文本和元数据"""

    def __init__(self, record_id: str, text: str, metadata: dict = None):
        self.id = record_id
        self.text = text
        self.metadata = metadata or {}
        self.created_at = datetime.now()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VectorRecord":
        return cls(
            record_id=data["id"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


class ReflectionVectorIndex:
    """FAISS 向量索引，自包含设计，可靠保存和恢复"""

    def __init__(self, embedder: Embedder, persist_dir: str):
        self.embedder = embedder
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.persist_dir / "reflection.faiss"
        self.records_path = self.persist_dir / "reflection_vectors.json"

        self.index: faiss.IndexFlatIP | None = None
        self.records: list[VectorRecord] = []
        self.id_to_idx: dict[str, int] = {}
        self.dimension: int = 0

        self._load()

    def add(self, record_id: str, text: str, metadata: dict = None) -> int:
        """添加记录，返回向量索引位置"""
        if record_id in self.id_to_idx:
            return self.id_to_idx[record_id]

        vector = self.embedder.embed(text)
        self.dimension = len(vector)
        vector_np = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vector_np)

        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)

        idx = self.index.ntotal
        self.index.add(vector_np)

        vec_record = VectorRecord(record_id, text, metadata)
        self.records.append(vec_record)
        self.id_to_idx[record_id] = idx

        return idx

    def remove(self, record_id: str) -> bool:
        """移除记录（重建索引）"""
        if record_id not in self.id_to_idx:
            return False

        removed_idx = self.id_to_idx.pop(record_id)
        self.records = [r for r in self.records if r.id != record_id]

        if self.index is not None and self.index.ntotal > 0:
            vectors = []
            for i in range(self.index.ntotal):
                if i != removed_idx:
                    vectors.append(self.index.reconstruct(i))

            self.index = None
            self.id_to_idx.clear()

            if vectors:
                vectors_np = np.array(vectors, dtype=np.float32)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.index.add(vectors_np)

                for i, vec_record in enumerate(self.records):
                    self.id_to_idx[vec_record.id] = i

        return True

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float, dict]]:
        """语义检索，返回 (record_id, similarity_score, metadata)"""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_vector = self.embedder.embed(query)
        query_np = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_np)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_np, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.records):
                vec_record = self.records[idx]
                results.append((vec_record.id, float(score), vec_record.metadata))
        return results

    def get_text(self, record_id: str) -> str | None:
        """获取记录的原始文本"""
        if record_id in self.id_to_idx:
            idx = self.id_to_idx[record_id]
            return self.records[idx].text
        return None

    def save(self):
        """持久化索引和记录"""
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))

        records_data = [r.to_dict() for r in self.records]
        self.records_path.write_text(json.dumps(records_data, ensure_ascii=False, indent=2))

    def _load(self):
        """加载索引和记录"""
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.dimension = self.index.d

        if self.records_path.exists():
            records_data = json.loads(self.records_path.read_text())
            self.records = [VectorRecord.from_dict(r) for r in records_data]
            self.id_to_idx = {r.id: i for i, r in enumerate(self.records)}

    def count(self) -> int:
        return len(self.records)

    def contains(self, record_id: str) -> bool:
        return record_id in self.id_to_idx
