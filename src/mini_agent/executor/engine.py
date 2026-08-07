from typing import Any
from mini_agent.planner.models import Task, TaskStatus
from mini_agent.tool_manager import AgentToolManager
from mini_agent.router import CapabilityRouter


class TaskEngine:
    def __init__(self, controller=None, tracelog=None, memory_manager=None):
        self.controller = controller
        self.tracelog = tracelog
        self.memory_manager = memory_manager
        self.tool_manager = AgentToolManager()
        self.router = CapabilityRouter(self.tool_manager)

    def execute_task(self, task: Task, state: Any = None, role: str = "user") -> tuple[bool, Any]:
        try:
            if task.capability == "answer":
                result = self._execute_answer(task, state)
            else:
                result = self._execute_tool(task)

            is_valid, error = task.validate_result(result)
            if not is_valid:
                return False, error

            return True, result

        except Exception as e:
            return False, str(e)

    def _execute_tool(self, task: Task) -> Any:
        decision = {
            "type": "tool",
            "tool": task.capability,
            "args": task.input
        }

        tool_info = self.router.route(decision)
        if not tool_info:
            return {"error": f"未找到能力 '{task.capability}' 对应的工具"}

        result = self.tool_manager.execute_tool(
            tool_info["tool_name"],
            **tool_info["args"]
        )
        return result

    def _execute_answer(self, task: Task, state: Any = None) -> str:
        if self.controller is None:
            return "Controller not available"

        if task.dependencies and state:
            summary_parts = []
            for dep_id in task.dependencies:
                dep_task = self._get_task_from_state(dep_id, state)
                if dep_task and dep_task.result:
                    if dep_task.output_schema:
                        schema = dep_task._resolve_schema()
                        if schema:
                            fields = schema.get("properties", {})
                            for field_name in fields:
                                value = dep_task.get_result_field(field_name)
                                if value is not None:
                                    summary_parts.append(f"{dep_task.description}.{field_name}: {value}")
                    else:
                        summary_parts.append(f"{dep_task.description}: {dep_task.result}")
            state["observations"] = "\n".join(summary_parts)

        decision = self.controller.step(state)
        return decision.get("content", "无法生成答案")

    def _get_task_from_state(self, task_id: str, state: Any = None) -> Task | None:
        if state and hasattr(state, "task_graph"):
            return state.task_graph.get_task(task_id)
        return None
