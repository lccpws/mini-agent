import json
from pathlib import Path
from mini_agent.knowledge.models import Document, DocumentChunk
from mini_agent.knowledge.extractor import TextExtractor
from mini_agent.knowledge.chunker import Chunker
from mini_agent.memory.embedder import Embedder


class KnowledgeBase:
    """知识库管理器"""

    def __init__(self, persist_dir: str = "knowledge_base"):
        self.persist_dir = Path(persist_dir)
        self.documents_path = self.persist_dir / "documents.json"
        self.vectors_path = self.persist_dir / "vectors.json"
        
        self.extractor = TextExtractor()
        self.chunker = Chunker()
        self.embedder = Embedder()
        
        self.documents: list[Document] = []
        self.chunks: list[DocumentChunk] = []
        self.vectors: list[list[float]] = []
        
        self._load()

    def upload(self, file_path: str, metadata: dict = None) -> Document:
        path = Path(file_path)
        content = self.extractor.extract(file_path)
        
        doc = Document(
            filename=path.name,
            content=content,
            doc_type=path.suffix.lower().lstrip("."),
            metadata=metadata or {}
        )
        
        chunks = self.chunker.chunk(content, document_id=doc.id)
        doc.chunk_count = len(chunks)
        
        vectors = self.embedder.embed_batch([chunk.content for chunk in chunks])
        
        self.documents.append(doc)
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)
        
        self._save()
        
        print(f"上传文档: {doc.filename}")
        print(f"  - 分块数: {doc.chunk_count}")
        print(f"  - 总字符数: {len(content)}")
        
        return doc

    def search(self, query: str, top_k: int = 3) -> list[DocumentChunk]:
        if not self.chunks:
            return []
        
        query_vector = self.embedder.embed(query)
        
        scores = []
        for i, vector in enumerate(self.vectors):
            score = self._cosine_similarity(query_vector, vector)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scores[:top_k]:
            chunk = self.chunks[idx]
            results.append(chunk)
        
        return results

    def list_documents(self) -> list[Document]:
        return self.documents

    def get_document(self, doc_id: str) -> Document | None:
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def delete_document(self, doc_id: str) -> bool:
        doc = self.get_document(doc_id)
        if not doc:
            return False
        
        self.documents = [d for d in self.documents if d.id != doc_id]
        
        indices_to_remove = [
            i for i, chunk in enumerate(self.chunks)
            if chunk.document_id == doc_id
        ]
        
        for idx in sorted(indices_to_remove, reverse=True):
            self.chunks.pop(idx)
            self.vectors.pop(idx)
        
        self._save()
        
        print(f"删除文档: {doc.filename}")
        return True

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)

    def _save(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        docs_data = []
        for doc in self.documents:
            docs_data.append({
                "id": doc.id,
                "filename": doc.filename,
                "content": doc.content,
                "doc_type": doc.doc_type,
                "metadata": doc.metadata,
                "created_at": doc.created_at.isoformat(),
                "chunk_count": doc.chunk_count
            })
        
        chunks_data = []
        for chunk in self.chunks:
            chunks_data.append({
                "id": chunk.id,
                "document_id": chunk.document_id,
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata
            })
        
        self.documents_path.write_text(json.dumps(docs_data, ensure_ascii=False))
        self.vectors_path.write_text(json.dumps({
            "chunks": chunks_data,
            "vectors": self.vectors
        }, ensure_ascii=False))

    def _load(self):
        if self.documents_path.exists():
            docs_data = json.loads(self.documents_path.read_text())
            for item in docs_data:
                doc = Document(
                    id=item["id"],
                    filename=item["filename"],
                    content=item["content"],
                    doc_type=item["doc_type"],
                    metadata=item.get("metadata", {}),
                    chunk_count=item.get("chunk_count", 0)
                )
                self.documents.append(doc)
        
        if self.vectors_path.exists():
            data = json.loads(self.vectors_path.read_text())
            chunks_data = data.get("chunks", [])
            self.vectors = data.get("vectors", [])
            
            for item in chunks_data:
                chunk = DocumentChunk(
                    id=item["id"],
                    document_id=item["document_id"],
                    content=item["content"],
                    chunk_index=item["chunk_index"],
                    metadata=item.get("metadata", {})
                )
                self.chunks.append(chunk)

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "total_chars": sum(len(doc.content) for doc in self.documents)
        }