import sys
sys.path.insert(0, "/Users/weishuai/project/AiGrow/AI-System-Engineer-Learning/02-Implementation/mini-agent")

from runtime.agent.llm import LLM
from runtime.agent.react_agent import ReactAgent
from runtime.tool_manager import AgentToolManager
from runtime.capability_router import CapabilityRouter
from runtime.executor import Executor
from runtime.risk.approval_manager import ApprovalManager


class AutoApprovalManager(ApprovalManager):
    """自动批准的审批管理器（用于测试）"""
    def request_approval(self, tool_name, args):
        print(f"自动批准: {tool_name}")
        return True


def test_scenario(question, description):
    print("\n" + "=" * 60)
    print(f"测试场景: {description}")
    print(f"输入问题: {question}")
    print("=" * 60)

    # 初始化组件
    manager = AgentToolManager()
    llm = LLM(tools=list(manager.tools.values()))
    router = CapabilityRouter(manager)
    executor = Executor()
    executor.approval_manager = AutoApprovalManager()

    # 创建 ReactAgent
    agent = ReactAgent(llm, router, executor)

    # 执行
    result = agent.run(question, role="user")

    # 输出结果
    print("\nExecution Trace:")
    print("-" * 40)
    print(result)

    # 输出 Memory
    print("\nScratchPad Memory:")
    print("-" * 40)
    print(agent.get_memory())
    
    return result


def main():
    # 场景 1: 多步计算
    test_scenario(
        "帮我计算 (15 + 25) * 3 等于多少，然后搜索一下这个结果代表什么含义",
        "多步计算 + 搜索"
    )

    # 场景 2: 条件判断 + 多工具
    test_scenario(
        "北京天气怎么样？如果温度超过20度，帮我搜索适合的户外活动",
        "天气查询 + 条件判断 + 搜索"
    )

    # 场景 3: 连续工具调用
    test_scenario(
        "先搜索什么是 machine learning，然后用计算器算一下 2 的 10 次方",
        "搜索 + 计算"
    )


if __name__ == "__main__":
    main()
