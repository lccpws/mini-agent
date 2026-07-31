import sys
sys.path.insert(0, "/Users/weishuai/project/AiGrow/AI-System-Engineer-Learning/02-Implementation/mini-agent/src")

from mini_agent.llm import LLM
from mini_agent.agent import ReactAgent
from mini_agent.runner import ReActController
from mini_agent.tool_manager import AgentToolManager
from mini_agent.router import CapabilityRouter
from mini_agent.executor import Executor
from mini_agent.guardrails.approval_manager import ApprovalManager


class AutoApprovalManager(ApprovalManager):
    def request_approval(self, tool_name, args):
        print(f'自动批准: {tool_name}')
        return True


def test_react_mode():
    print("=" * 60)
    print("测试 1: 纯 ReAct 模式")
    print("=" * 60)

    manager = AgentToolManager()
    llm = LLM(tools=list(manager.tools.values()))
    router = CapabilityRouter(manager)
    executor = Executor()
    executor.approval_manager = AutoApprovalManager()

    controller = ReActController(llm, router, executor)
    agent = ReactAgent(controller, max_steps=5)

    question = "北京天气怎么样"
    print(f"\n输入问题: {question}\n")

    result = agent.run(question, role="user", mode="react")
    print("\n" + result)
    print("\nMemory Stats:", agent.get_memory_stats())


if __name__ == "__main__":
    test_react_mode()
