from abc import ABC, abstractmethod
from mini_agent.planner.models import Plan, PlanStep
from mini_agent.planner.validator import PlanValidator, DependencyValidator
from mini_agent.llm import LLM
import json


class BasePlanner(ABC):

    @abstractmethod
    def create_plan(self,
                    goal: str,
                    context: str) -> Plan:
        pass


class LLMPlanner(BasePlanner):
    def __init__(self, llm: LLM, capabilities: list[str] = None):
        self.llm = llm
        self.capabilities = capabilities or []
        self.validator = PlanValidator()
        self.dep_validator = DependencyValidator()

    def create_plan(self, goal: str, context: str) -> Plan:
        prompt = f"""
        你是一个任务规划器。
        目标：
        {goal}
        上下文：
        {context}
        请把目标拆分成可执行步骤。

        可用能力：
        {', '.join(self.capabilities) if self.capabilities else '未指定'}

        要求：
        1. 每个步骤必须有唯一id
        2. 明确任务依赖
        3. 不允许创建不存在的能力
        4. 尽量让没有依赖关系的任务并行
        5. 最后一步必须能够完成最终目标

        输出JSON：
        {{
            "goal": "...",
            "steps": [
                {{
                    "id": "step_1",
                    "task": "...",
                    "dependencies": [],
                    "capability": "search"
                }}
            ]
        }}
        """
        response = self.llm.generate(prompt)
        data = json.loads(response)
        steps = [PlanStep(**step) for step in data["steps"]]
        plan = Plan(goal=data["goal"], steps=steps)

        self.validator.validate(plan, self.capabilities)
        self.dep_validator.validate_no_cycle(plan.steps)

        return plan
