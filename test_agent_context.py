from mini_agent.context import ContextManager, ContextBuilder, ContextSelector, ContextItem, ContextSource
from mini_agent.state import AgentState
from mini_agent.memory.models import Memory, MemoryType

def test_build_context_items():
    from mini_agent.agent import ReactAgent
    from mini_agent.runner import ReActController
    from mini_agent.llm import LLM
    
    llm = LLM()
    controller = ReActController(llm, None, None)
    agent = ReactAgent(controller)
    
    state = AgentState(
        question="北京天气怎么样？",
        memories=[
            Memory(content="用户在北京工作", memory_type=MemoryType.FACT),
            Memory(content="用户喜欢晴天", memory_type=MemoryType.PREFERENCE)
        ],
        observations=["工具返回：晴，25°C"]
    )
    
    items = agent._build_context_items(state)
    
    print("构建的ContextItems:")
    for item in items:
        print(f"  [{item.source.value}] priority={item.priority}, tokens={item.token_count}")
        print(f"    内容: {item.content[:50]}...")
        print()

if __name__ == "__main__":
    test_build_context_items()