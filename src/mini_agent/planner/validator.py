
class PlanValidator:
    def validate(self, plan, capabilities):

        task_ids = {task.id for task in plan.tasks}
        for task in plan.tasks:

            # 1. Validate required fields
            if not task.description:
                raise ValueError(f"Task '{task.id}' is missing 'description'")
            if not task.objective:
                raise ValueError(f"Task '{task.id}' is missing 'objective'")
            if not task.expected_output:
                raise ValueError(f"Task '{task.id}' is missing 'expected_output'")

            # 2. Validate dependencies
            for dep in task.dependencies:
                if dep not in task_ids:
                    raise ValueError(f"Invalid dependency '{dep}' in task '{task.id}'")

            # 3. Validate capability
            if task.capability and task.capability not in capabilities:
                raise ValueError(f"Invalid capability '{task.capability}' in task '{task.id}'")

        return True


class DependencyValidator:
    def validate_no_cycle(self, tasks):
        graph = {task.id: task.dependencies for task in tasks}
        visited = set()
        visiting = set()

        def dfs(node):
            if node in visiting:
                raise ValueError("Cycle detected in dependencies")
            if node in visited:
                return
            visiting.add(node)
            for neighbor in graph.get(node, []):
                dfs(neighbor)
            visited.add(node)
            visiting.remove(node)

        for node in graph:
            dfs(node)
        return True
