from mini_agent.llm import LLM
from mini_agent.formatter import ObservationFormatter
from mini_agent.router import CapabilityRouter


class ReActController:
    """ReAct 逻辑控制器，负责单步决策"""

    def __init__(self, llm: LLM, router: CapabilityRouter, executor):
        self.llm = llm
        self.router = router
        self.executor = executor

    def step(self, state: dict) -> dict:
        """执行单步决策，返回 decision"""
        decision = self.llm.run(state)
        return decision

    def execute_tool(self, decision: dict, role: str = "user") -> dict:
        """执行工具并返回格式化的 observation"""
        tool_info = self.router.route(decision)
        
        if not tool_info:
            return {
                "tool": decision.get("tool", ""),
                "result": "工具未找到",
                "status": "error"
            }
        
        result = self.executor.execute(
            role=role,
            tool_name=tool_info["tool_name"],
            args=tool_info["args"]
        )
        
        observation = ObservationFormatter.from_executor(tool_info["tool_name"], result)
        return observation.to_dict()
