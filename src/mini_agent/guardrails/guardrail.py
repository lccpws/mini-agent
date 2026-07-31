from mini_agent.tools.base_tool import BaseTool
class Guardrail:
    def validate(self, tool: BaseTool, role):
        if role not in tool.allow_roles:
            return False
        return True