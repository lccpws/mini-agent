from abc import ABC, abstractmethod
from mini_agent.planner.models import Plan, Task, TaskSchemas, PlanQuality
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
    def __init__(self, llm: LLM, capabilities: list[str] = None, max_retries: int = 3):
        self.llm = llm
        self.capabilities = capabilities or []
        self.validator = PlanValidator()
        self.dep_validator = DependencyValidator()
        self.quality_checker = PlanQuality(capabilities)
        self.max_retries = max_retries

    def create_plan(self, goal: str, context: str) -> Plan | None:
        available_schemas = TaskSchemas.list_all()
        retry_context = ""

        for attempt in range(self.max_retries):
            prompt = f"""
            你是一个任务规划器。

            目标：
            {goal}
            上下文：
            {context}
            {retry_context}
            你的职责是：
            将用户目标拆解成可执行的任务图。

            可用能力：
            {', '.join(self.capabilities) if self.capabilities else '未指定'}

            预定义输出 Schema（优先使用字符串引用）：
            {', '.join(available_schemas)}

            要求：
            1. 每个任务必须具有独立的业务目标
            2. 不要把简单的工具调用拆成独立任务
            3. 每个任务必须声明 capability
            4. capability 必须来自系统提供的能力列表
            5. 每个任务必须声明 expected_output
            6. 每个任务必须声明 output_schema，优先使用预定义 schema 的字符串名称（如 "weather"、"search"）
            7. 如果没有合适的预定义 schema，可以自定义 JSON Schema
            8. 必须明确任务依赖关系
            9. 没有依赖关系的任务应该并行
            10. 不要创建不存在的 capability
            11. 不要执行任务，只负责规划
            12. 最终任务必须能够覆盖用户目标

            输出必须严格符合 JSON Schema：
            {{
                "goal": "...",
                "reasoning": "为什么这样拆分任务",
                "tasks": [
                    {{
                        "id": "task_1",
                        "description": "任务描述",
                        "objective": "任务目标",
                        "capability": "search",
                        "dependencies": [],
                        "input": {{}},
                        "expected_output": "预期输出",
                        "output_schema": "search"
                    }}
                ]
            }}
            """

            try:
                response = self.llm.generate(prompt)
                plan = self._parse_plan(response)

                if isinstance(plan, Plan):
                    return plan

                retry_context = f"\n\n上次生成的计划解析失败: {plan}"

            except ValueError as e:
                retry_context = f"\n\n上次生成的计划校验失败: {e}"
                print(f"计划校验失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

            except Exception as e:
                retry_context = f"\n\n生成计划时发生错误: {e}"
                print(f"计划生成失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")

        print(f"已达到最大重试次数 ({self.max_retries})，无法生成有效计划")
        return None

    def _parse_plan(self, output) -> Plan | dict:
        try:
            data = json.loads(output)
            tasks = [Task(**task) for task in data["tasks"]]
            plan = Plan(
                goal=data["goal"],
                tasks=tasks,
                reasoning=data.get("reasoning", "")
            )

            self.validator.validate(plan, self.capabilities)
            self.dep_validator.validate_no_cycle(plan.tasks)

            plan.quality_score = self.quality_checker.evaluate(plan)

            return plan

        except json.JSONDecodeError as e:
            return {"type": "error", "content": f"JSON 解析失败: {e}"}

        except ValueError as e:
            raise e

        except Exception as e:
            return {"type": "error", "content": f"解析失败: {e}"}
