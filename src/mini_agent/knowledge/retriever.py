from mini_agent.knowledge.base import KnowledgeBase
from mini_agent.knowledge.models import DocumentChunk
from mini_agent.context.models import ContextItem
from mini_agent.context.token_counter import TokenCounter, TokenCounterFactory


class KnowledgeRetriever:
    """知识库检索器"""

    def __init__(self, knowledge_base: KnowledgeBase, token_counter: TokenCounter = None):
        self.kb = knowledge_base
        self.token_counter = token_counter or TokenCounterFactory.create()

    def search(self, query: str, top_k: int = 3) -> list[ContextItem]:
        chunks = self.kb.search(query, top_k)
        return [self._to_context_item(chunk) for chunk in chunks]

    def _to_context_item(self, chunk: DocumentChunk) -> ContextItem:
        doc = self.kb.get_document(chunk.document_id)
        filename = doc.filename if doc else "unknown"
        
        content = f"[知识库:{filename}] {chunk.content}"
        
        return ContextItem(
            id=f"rag_{chunk.id}",
            content=content,
            source="rag",
            priority=0.4,
            token_count=self.token_counter.count_tokens(content),
        )