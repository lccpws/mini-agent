from typing import Any
from mini_agent.planner.models import Task
from mini_agent.reflection.models import EvaluationResult, ReflectionResult


class Reflection:
    def __init__(self, llm=None):
        self.llm = llm

    def reflect(self, task: Task, result: Any, evaluation: EvaluationResult) -> ReflectionResult:
        if self.llm is None:
            return self._rule_based_reflect(task, result, evaluation)

        return self._llm_reflect(task, result, evaluation)

    def _rule_based_reflect(self, task: Task, result: Any, evaluation: EvaluationResult) -> ReflectionResult:
        feedback_parts = []

        if evaluation.suggestions:
            feedback_parts.extend(evaluation.suggestions)

        improved_input = dict(task.input) if task.input else {}

        if isinstance(result, dict) and "error" in result:
            error_msg = result["error"]
            if "未找到" in error_msg or "不存在" in error_msg:
                feedback_parts.append("工具或资源不存在，尝试使用替代方案")
            elif "超时" in error_msg or "timeout" in error_msg:
                feedback_parts.append("执行超时，尝试简化请求")
            elif "权限" in error_msg or "403" in error_msg:
                feedback_parts.append("权限不足，需要调整访问方式")

        should_retry = (
            evaluation.score < 60
            and len(feedback_parts) > 0
            and task.retry_count < task.max_retry
        )

        return ReflectionResult(
            reflected=len(feedback_parts) > 0,
            feedback="\n".join(feedback_parts),
            improved_input=improved_input,
            should_retry=should_retry,
            root_cause=evaluation.reason
        )

    def _llm_reflect(self, task: Task, result: Any, evaluation: EvaluationResult) -> ReflectionResult:
        prompt = f"""你是一个任务执行反思专家。请分析任务执行失败的原因并提供改进建议。

任务信息：
- 任务ID: {task.id}
- 任务描述: {task.description}
- 任务目标: {task.objective}
- 任务输入: {task.input}
- 已重试次数: {task.retry_count}/{task.max_retry}

执行结果：
{result}

评估结果：
- 分数: {evaluation.score}
- 原因: {evaluation.reason}
- 建议: {evaluation.suggestions}

请返回JSON格式的反思结果：
{{
    "reflected": true,
    "feedback": "详细的反思反馈",
    "improved_input": {{}},
    "should_retry": true/false,
    "root_cause": "根本原因分析",
    "suggested_capability": "建议使用的能力（如果需要更换）"
}}"""

        try:
            import json
            response = self.llm.generate(prompt)
            data = json.loads(response)
            return ReflectionResult(
                reflected=data.get("reflected", True),
                feedback=data.get("feedback", ""),
                improved_input=data.get("improved_input", task.input),
                should_retry=data.get("should_retry", False),
                root_cause=data.get("root_cause", ""),
                suggested_capability=data.get("suggested_capability")
            )
        except Exception:
            return self._rule_based_reflect(task, result, evaluation)
