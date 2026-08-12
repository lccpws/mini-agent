"""测试完整 Plan+ReAct 流程：乌鲁木齐旅游问题（带调试）"""
import sys
sys.path.insert(0, 'src')

from mini_agent.llm import LLM
from mini_agent.runner import ReActController
from mini_agent.router import CapabilityRouter
from mini_agent.tool_manager import AgentToolManager
from mini_agent.agent import ReactAgent
from mini_agent.planner import LLMPlanner


def main():
    print("[1] 初始化 LLM...")
    llm = LLM(model="gpt-4o")

    print("[2] 初始化工具管理器...")
    tool_manager = AgentToolManager()
    print(f"    已发现工具: {list(tool_manager.tools.keys())}")

    print("[3] 初始化路由器...")
    router = CapabilityRouter(tool_manager)

    print("[4] 初始化 TaskEngine...")
    from mini_agent.executor import TaskEngine
    task_engine = TaskEngine(llm=llm)

    print("[5] 初始化 Controller...")
    controller = ReActController(llm=llm, router=router, executor=task_engine)

    capabilities = ["answer"]
    for tool in tool_manager.tools.values():
        capabilities.extend(tool.capabilities)
    capabilities = list(set(capabilities))
    print(f"[6] 可用能力: {capabilities}")

    print("[7] 初始化 Planner...")
    planner = LLMPlanner(llm=llm, capabilities=capabilities)

    print("[8] 初始化 Agent...")
    agent = ReactAgent(
        controller=controller,
        planner=planner,
        max_steps=10,
        debug_context=False,
    )

    question = "十一期间乌鲁木齐的气温怎么样，如果是十一期间我要去乌鲁木齐周边旅游的话，我需要注意什么？着装上有什么要求吗，并且给我出一个5天的游玩方案"

    print("[9] 开始执行...")
    print("=" * 60)
    print(f"用户问题: {question}")
    print("=" * 60)

    result = agent.run(question, mode="plan_react")

    print("\n" + "=" * 60)
    print("最终结果:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
