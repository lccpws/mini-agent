from mini_agent.context.manager import ContextManager
from mini_agent.context.builder import ContextBuilder
from mini_agent.context.selector import ContextSelector
from mini_agent.context.models import ContextItem, ContextSource

def main():
    selector = ContextSelector()
    builder = ContextBuilder(selector)
    manager = ContextManager(builder)
    
    items = [
        ContextItem(
            content="You are a helpful assistant.",
            source=ContextSource.SYSTEM,
            priority=10,
            token_count=100
        ),
        ContextItem(
            content="The user's name is Alice.",
            source=ContextSource.MEMORY,
            priority=5,
            token_count=50
        ),
        ContextItem(
            content="Previous conversation: User asked about weather.",
            source=ContextSource.HISTORY,
            priority=3,
            token_count=80
        ),
        ContextItem(
            content="Tool result: Temperature is 25°C.",
            source=ContextSource.TOOL,
            priority=2,
            token_count=60
        ),
    ]
    
    selected = manager.build_context(items, total_tokens=1000, output_tokens=200)
    
    print("Selected context items:")
    for item in selected:
        print(f"  - [{item.source.value}] {item.content[:50]}... (priority: {item.priority}, tokens: {item.token_count})")

if __name__ == "__main__":
    main()