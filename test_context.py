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
            id="sys1",
            content="You are a helpful assistant.",
            source=ContextSource.SYSTEM,
            priority=10,
            token_count=100
        ),
        ContextItem(
            id="mem1",
            content="The user's name is Alice.",
            source=ContextSource.MEMORY,
            priority=5,
            token_count=50
        ),
        ContextItem(
            id="mem1",
            content="The user prefers Python.",
            source=ContextSource.MEMORY,
            priority=5,
            token_count=40,
            reliability=0.9,
            recency=0.8
        ),
        ContextItem(
            id="hist1",
            content="Previous conversation: User asked about weather.",
            source=ContextSource.HISTORY,
            priority=3,
            token_count=80
        ),
        ContextItem(
            id="tool1",
            content="Tool result: Temperature is 25°C.",
            source=ContextSource.TOOL,
            priority=2,
            token_count=60
        ),
    ]
    
    print(f"输入: {len(items)} items (含重复 id)")
    
    selected = manager.build_context(items, total_tokens=1000, output_tokens=200)
    
    print(f"\n输出: {len(selected)} items (去重后)")
    print("\nSelected context items (sorted by utility):")
    for item in selected:
        score = manager.scorer.score(item)
        utility = manager.scorer.utility(item)
        print(f"  - [{item.source.value}] id={item.id}, {item.content[:40]}... (tokens: {item.token_count}, score: {score:.3f}, utility: {utility:.4f})")

if __name__ == "__main__":
    main()