from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from mini_agent.planner.models import Task, TaskStatus, FailureType
from mini_agent.tool_manager import AgentToolManager
from mini_agent.router import CapabilityRouter
from mini_agent.executor.failure import FailureClassifier
from mini_agent.executor.retry import RetryManager
from mini_agent.trace_step import TaskExecutionRecord, ExecutionTrace


class TaskEngine:
    def __init__(self, controller=None, tracelog=None, memory_manager=None, max_workers: int = 4):
        self.controller = controller
        self.tracelog = tracelog
        self.memory_manager = memory_manager
        self.tool_manager = AgentToolManager()
        self.router = CapabilityRouter(self.tool_manager)
        self.retry_manager = RetryManager()
        self.max_workers = max_workers
        self.execution_trace = ExecutionTrace()

    def execute_task(self, task: Task, state: Any = None, role: str = "user") -> tuple[bool, Any, str]:
        record = TaskExecutionRecord(
            task_id=task.id,
            task_description=task.description,
            capability=task.capability
        )
        record.start()

        try:
            if task.capability == "answer":
                result = self._execute_answer(task, state)
            else:
                result = self._execute_tool(task)

            is_valid, error = task.validate_result(result)
            if not is_valid:
                record.complete(False, error, error, FailureType.PERMANENT)
                self.execution_trace.add_record(record)
                return False, error, FailureType.PERMANENT

            record.complete(True, result)
            self.execution_trace.add_record(record)
            return True, result, None

        except Exception as e:
            failure_type = FailureClassifier.classify(e)
            record.complete(False, None, str(e), failure_type)
            self.execution_trace.add_record(record)
            return False, str(e), failure_type

    def execute_task_with_retry(self, task: Task, state: Any = None, role: str = "user") -> tuple[bool, Any, str]:
        record = self.execution_trace.get_record(task.id)
        if not record:
            record = TaskExecutionRecord(
                task_id=task.id,
                task_description=task.description,
                capability=task.capability
            )
            self.execution_trace.add_record(record)

        record.start()
        record.retry_count = task.retry_count

        def execute():
            if task.capability == "answer":
                return self._execute_answer(task, state)
            else:
                return self._execute_tool(task)

        success, result, failure_type = self.retry_manager.execute_with_retry(execute, task)

        if success:
            is_valid, error = task.validate_result(result)
            if not is_valid:
                record.complete(False, error, error, FailureType.PERMANENT)
                return False, error, FailureType.PERMANENT

        record.complete(success, result if success else None, result if not success else None, failure_type)
        record.retry_count = task.retry_count

        return success, result, failure_type

    def execute_tasks_parallel(
        self,
        tasks: list[Task],
        state: Any = None,
        role: str = "user"
    ) -> list[tuple[Task, bool, Any, str]]:
        non_answer_tasks = [t for t in tasks if t.capability != "answer"]
        answer_tasks = [t for t in tasks if t.capability == "answer"]

        results = []

        if non_answer_tasks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {}
                for task in non_answer_tasks:
                    task.status = TaskStatus.RUNNING
                    future = executor.submit(self.execute_task_with_retry, task, state, role)
                    futures[future] = task

                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        success, result, failure_type = future.result()
                        results.append((task, success, result, failure_type))
                    except Exception as e:
                        results.append((task, False, str(e), FailureType.UNKNOWN))

        for task in answer_tasks:
            task.status = TaskStatus.RUNNING
            success, result, failure_type = self.execute_task_with_retry(task, state, role)
            results.append((task, success, result, failure_type))

        return results

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
