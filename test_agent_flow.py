"""测试 Agent 完整流程"""
import sys
sys.path.insert(0, 'src')

from mini_agent.planner import (
    Plan, Task, TaskGraph, PlanStatus, TaskStatus,
    PlanValidator, DependencyValidator
)
from mini_agent.tools.weather import WeatherTool
from mini_agent.tools.calculator import CalculatorTool


def test_complex_plan_execution():
    """测试复杂计划的完整执行流程"""
    print("=" * 60)
    print("测试：复杂问题的计划制定与执行")
    print("=" * 60)

    question = "帮我查一下北京、上海、广州三个城市的天气，然后计算三个城市温度的平均值，最后告诉我哪个城市最适合出游"
    print(f"\n用户问题: {question}")

    # 1. 模拟 LLM 生成的计划
    print("\n" + "=" * 60)
    print("Phase 1: Planning")
    print("=" * 60)

    plan = Plan(
        goal=question,
        reasoning="需要先查询三个城市的天气，然后计算平均温度，最后综合分析给出建议",
        tasks=[
            Task(
                id="task_1",
                description="查询北京天气",
                objective="获取北京今天的天气和温度信息",
                capability="weather",
                dependencies=[],
                input={"city": "北京"},
                expected_output="北京的天气和温度"
            ),
            Task(
                id="task_2",
                description="查询上海天气",
                objective="获取上海今天的天气和温度信息",
                capability="weather",
                dependencies=[],
                input={"city": "上海"},
                expected_output="上海的天气和温度"
            ),
            Task(
                id="task_3",
                description="查询广州天气",
                objective="获取广州今天的天气和温度信息",
                capability="weather",
                dependencies=[],
                input={"city": "广州"},
                expected_output="广州的天气和温度"
            ),
            Task(
                id="task_4",
                description="计算平均温度",
                objective="计算三个城市温度的平均值",
                capability="calculator",
                dependencies=["task_1", "task_2", "task_3"],
                input={"expression": "(北京温度 + 上海温度 + 广州温度) / 3"},
                expected_output="三个城市的平均温度"
            ),
            Task(
                id="task_5",
                description="综合分析并给出建议",
                objective="根据天气和温度信息，分析哪个城市最适合出游",
                capability="answer",
                dependencies=["task_1", "task_2", "task_3", "task_4"],
                expected_output="出游建议和城市推荐"
            ),
        ]
    )

    plan.status = PlanStatus.RUNNING

    print(f"目标: {plan.goal}")
    print(f"推理: {plan.reasoning}")
    print(f"任务数: {len(plan.tasks)}")
    for task in plan.tasks:
        deps = ", ".join(task.dependencies) if task.dependencies else "无"
        print(f"  - [{task.id}] {task.description}")
        print(f"    目标: {task.objective}")
        print(f"    依赖: {deps}")
        print(f"    能力: {task.capability}")
        print(f"    预期输出: {task.expected_output}")

    # 2. 验证计划
    print("\n" + "=" * 60)
    print("Phase 2: Validation")
    print("=" * 60)

    capabilities = ["weather", "calculator", "answer"]
    validator = PlanValidator()
    dep_validator = DependencyValidator()

    try:
        validator.validate(plan, capabilities)
        print("PlanValidator: 通过 ✓")
    except ValueError as e:
        print(f"PlanValidator: 失败 - {e}")
        return

    try:
        dep_validator.validate_no_cycle(plan.tasks)
        print("DependencyValidator: 无循环依赖 ✓")
    except ValueError as e:
        print(f"DependencyValidator: 失败 - {e}")
        return

    # 3. 创建 TaskGraph 并执行
    print("\n" + "=" * 60)
    print("Phase 3: Execution")
    print("=" * 60)

    task_graph = TaskGraph(plan.tasks)
    weather_tool = WeatherTool()
    calculator_tool = CalculatorTool()

    task_counter = 0
    temperatures = {}

    while True:
        ready_tasks = task_graph.get_ready_tasks()

        if not ready_tasks:
            if task_graph.all_completed():
                plan.status = PlanStatus.COMPLETED
                print("\n所有任务执行完成!")
            else:
                plan.status = PlanStatus.FAILED
                print("\n执行失败")
            break

        for task in ready_tasks:
            task_counter += 1
            task.status = TaskStatus.RUNNING
            print(f"\nTask {task_counter}: [{task.id}] {task.description}")

            try:
                if task.capability == "weather":
                    observation = weather_tool.execute(**task.input)
                    print(f"  执行: weather({task.input})")
                    print(f"  结果: {observation}")

                    city = task.input["city"]
                    if "25度" in observation:
                        temperatures[city] = 25

                elif task.capability == "calculator":
                    expr = task.input.get("expression", "")
                    if "北京温度" in expr:
                        expr = expr.replace("北京温度", str(temperatures.get("北京", 0)))
                    if "上海温度" in expr:
                        expr = expr.replace("上海温度", str(temperatures.get("上海", 0)))
                    if "广州温度" in expr:
                        expr = expr.replace("广州温度", str(temperatures.get("广州", 0)))

                    result = eval(expr)
                    observation = f"计算结果: {result}"
                    print(f"  执行: calculator({expr})")
                    print(f"  结果: {observation}")

                elif task.capability == "answer":
                    summary = "各城市天气:\n"
                    for city, temp in temperatures.items():
                        summary += f"- {city}: {temp}度\n"
                    summary += f"\n平均温度: {sum(temperatures.values()) / len(temperatures)}度"
                    summary += "\n\n建议: 三个城市天气都很好，温度适宜，都可以出游！"

                    observation = summary
                    print(f"  生成综合分析:")
                    print(f"  {observation}")

                task.status = TaskStatus.COMPLETED
                task.result = observation

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                print(f"  失败: {e}")
                break

    # 4. 输出最终结果
    print("\n" + "=" * 60)
    print("Phase 4: Result")
    print("=" * 60)

    print(f"计划状态: {plan.status}")
    print(f"计划版本: {plan.version}")
    print(f"\n任务执行结果:")
    for task in plan.tasks:
        status_icon = "✓" if task.status == TaskStatus.COMPLETED else "✗"
        print(f"  [{status_icon}] {task.id}: {task.description}")
        if task.result:
            result_preview = str(task.result)[:100] + "..." if len(str(task.result)) > 100 else str(task.result)
            print(f"      结果: {result_preview}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    test_complex_plan_execution()
