from mini_agent.context.models import ContextItem, ContextSource


class ContextDebugger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def debug(self, context_items: list[ContextItem]):
        if not self.enabled:
            return

        print("\n===== CONTEXT DEBUG =====")

        grouped = self._group_by_source(context_items)

        for source in [ContextSource.SYSTEM, ContextSource.USER, 
                       ContextSource.MEMORY, ContextSource.HISTORY, 
                       ContextSource.TOOL, ContextSource.RAG]:
            items = grouped.get(source, [])
            if items:
                print(f"\n{source.value.upper()}:")
                for item in items:
                    content = item.content[:200] + "..." if len(item.content) > 200 else item.content
                    print(f"  {content}")

        total_tokens = sum(item.token_count for item in context_items)
        print(f"\nTOTAL TOKENS: {total_tokens}")

        print("=========================\n")

    def _group_by_source(self, context_items: list[ContextItem]) -> dict:
        grouped = {}
        for item in context_items:
            if item.source not in grouped:
                grouped[item.source] = []
            grouped[item.source].append(item)
        return grouped