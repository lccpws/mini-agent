from runtime.agent import MiniAgent
from runtime.executor import Executor
from runtime.tool_manager import AgentToolManager


def main():
    agent = MiniAgent()
    question = "帮我执行下python命令：import tempfile" 
    print(agent.run("admin", question))
    # manager = AgentToolManager()
    # print(f"已注册的工具有：{manager.list_tools()}")
    # tools = manager.find_by_capability("weather")
    # print(f"具有 weather capability 的工具有：{[tool.name for tool in tools]}")

    


if __name__ == "__main__":
    main()
