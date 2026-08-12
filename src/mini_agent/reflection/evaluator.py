from typing import Any
from mini_agent.planner.models import Task
from mini_agent.reflection.models import EvaluationResult


class Evaluator:
    def __init__(self, llm=None, threshold: float = 60.0):
        self.llm = llm
        self.threshold = threshold

    def evaluate(self, task: Task, result: Any, context: str = "") -> EvaluationResult:
        rule_result = self._rule_based_evaluate(task, result)

        if rule_result.score < self.threshold and self.llm:
            llm_result = self._llm_evaluate(task, result, context)
            if llm_result:
                return llm_result

        return rule_result

    def _rule_based_evaluate(self, task: Task, result: Any) -> EvaluationResult:
        if result is None:
            return EvaluationResult(
                score=0,
                passed=False,
                reason="结果为空"
            )

        if isinstance(result, dict):
            if "error" in result:
                return EvaluationResult(
                    score=0,
                    passed=False,
                    reason=f"执行错误: {result['error']}"
                )

            schema = task._resolve_schema()
            if schema:
                required = schema.get("required", [])
                missing = [f for f in required if f not in result]
                if missing:
                    return EvaluationResult(
                        score=30,
                        passed=False,
                        reason=f"缺少必填字段: {', '.join(missing)}",
                        suggestions=[f"补充字段: {f}" for f in missing]
                    )

            return EvaluationResult(
                score=100,
                passed=True,
                reason="结果有效"
            )

        if isinstance(result, str):
            if len(result) < 10:
                return EvaluationResult(
                    score=30,
                    passed=False,
                    reason="结果内容过短",
                    suggestions=["提供更详细的结果"]
                )
            return EvaluationResult(
                score=80,
                passed=True,
                reason="文本结果有效"
            )

        return EvaluationResult(
            score=50,
            passed=True,
            reason="结果类型未知但非空"
        )

    def _llm_evaluate(self, task: Task, result: Any, context: str) -> EvaluationResult | None:
        if self.llm is None:
            return None

        prompt = f"""你是一个任务结果评估专家。请评估以下任务执行结果的质量。

任务信息：
- 任务ID: {task.id}
- 任务描述: {task.description}
- 任务目标: {task.objective}
- 预期输出: {task.expected_output}

执行结果：
{result}

上下文：
{context}

请返回JSON格式的评估结果：
{{
    "score": 0-100的分数,
    "passed": true/false,
    "reason": "评估原因",
    "suggestions": ["改进建议1", "改进建议2"],
    "confidence": 0-1的置信度
}}"""

        try:
            response = self.llm.generate(prompt)
            import json
            data = json.loads(response)
            return EvaluationResult(
                score=data.get("score", 50),
                passed=data.get("passed", False),
                reason=data.get("reason", ""),
                suggestions=data.get("suggestions", []),
                confidence=data.get("confidence", 0.5)
            )
        except Exception:
            return None
