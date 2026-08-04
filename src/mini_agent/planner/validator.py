
class PlanValidator:
    def validate(self, plan, capabilities):

        step_ids = {step.id for step in plan.steps}
        for step in plan.steps:

            # 1. Validate dependencies
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Invalid dependency '{dep}' in step '{step.id}'")

            # 2. Validate capability
            if step.capability and step.capability not in capabilities:
                raise ValueError(f"Invalid capability '{step.capability}' in step '{step.id}'")

        return True

class DependencyValidator:
    def validate_no_cycle(self, steps):
        graph = {step.id: step.dependencies for step in steps}
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
