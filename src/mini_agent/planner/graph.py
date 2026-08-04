from mini_agent.planner.models import Plan, PlanStep



def get_ready_steps(plan: Plan) -> list[PlanStep]:
    """获取所有依赖已满足、可执行的步骤，就是获取待执行的步骤，
    什么是待执行的步骤？就是状态为PENDING的步骤，并且所有依赖的步骤都已经完成"""
    completed = {
        step.id
        for step in plan.steps
        if step.status == "COMPLETED"
    }

    ready = []

    for step in plan.steps:

        if step.status != "PENDING":
            continue

        if all(
            dependency in completed
            for dependency in step.dependencies
        ):

            ready.append(step)

    return ready
