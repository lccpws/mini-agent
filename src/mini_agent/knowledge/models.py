from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Document:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    content: str = ""
    doc_type: str = ""  # txt, md, pdf
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    chunk_count: int = 0

    def __str__(self):
        return f"[{self.doc_type}] {self.filename} ({self.chunk_count} chunks)"


@dataclass
class DocumentChunk:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    content: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def __str__(self):
        return f"Chunk {self.chunk_index}: {self.content[:50]}..."