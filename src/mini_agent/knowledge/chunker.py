from mini_agent.knowledge.models import DocumentChunk


class Chunker:
    """文档分块器"""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, document_id: str = "") -> list[DocumentChunk]:
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append(DocumentChunk(
                    document_id=document_id,
                    content=chunk_text,
                    chunk_index=chunk_index
                ))
                chunk_index += 1
            
            start += self.chunk_size - self.overlap
        
        return chunks