import sys
sys.path.insert(0, "/Users/weishuai/project/AiGrow/AI-System-Engineer-Learning/02-Implementation/mini-agent")

from runtime.agent.llm import LLM
from runtime.agent.react_agent import ReactAgent
from runtime.agent.react_controller import ReActController
from runtime.agent.planner import Planner
from runtime.tool_manager import AgentToolManager
from runtime.capability_router import CapabilityRouter
from runtime.executor import Executor
from runtime.risk.approval_manager import ApprovalManager


class AutoApprovalManager(ApprovalManager):
    """自动批准的审批管理器（用于测试）"""
    def request_approval(self, tool_name, args):
        print(f"自动批准: {tool_name}")
        return True


def test_react_mode():
    """测试纯 ReAct 模式"""
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
    print("\nMemory:", agent.get_memory())


def test_plan_react_mode():
    """测试 Plan + ReAct 模式"""
    print("\n" + "=" * 60)
    print("测试 2: Plan + ReAct 模式")
    print("=" * 60)

    manager = AgentToolManager()
    llm = LLM(tools=list(manager.tools.values()))
    router = CapabilityRouter(manager)
    executor = Executor()
    executor.approval_manager = AutoApprovalManager()

    controller = ReActController(llm, router, executor)
    planner = Planner(llm)
    agent = ReactAgent(controller, planner=planner, max_steps=5)

    question = "北京天气怎么样"
    print(f"\n输入问题: {question}\n")

    result = agent.run(question, role="user", mode="plan_react")
    print("\n" + result)
    print("\nMemory:", agent.get_memory())


def test_complex_plan():
    """测试复杂任务的计划"""
    print("\n" + "=" * 60)
    print("测试 3: 复杂任务 - Plan + ReAct")
    print("=" * 60)

    manager = AgentToolManager()
    llm = LLM(tools=list(manager.tools.values()))
    router = CapabilityRouter(manager)
    executor = Executor()
    executor.approval_manager = AutoApprovalManager()

    controller = ReActController(llm, router, executor)
    planner = Planner(llm)
    agent = ReactAgent(controller, planner=planner, max_steps=5)

    question = "帮我查一下北京天气，然后计算 25+30 等于多少"
    print(f"\n输入问题: {question}\n")

    result = agent.run(question, role="user", mode="plan_react")
    print("\n" + result)


if __name__ == "__main__":
    test_react_mode()
    test_plan_react_mode()
    test_complex_plan()
