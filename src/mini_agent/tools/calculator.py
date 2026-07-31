from mini_agent.tools.base_tool import BaseTool

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "计算数学表达式"
    capabilities = ["calculator", "math"]

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin", "user"]
    required_permissions = ["network"]
    risk_level = 3

    parameters_schema = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如：2+3*4、100/5"
            }
        },
        "required": ["expression"]
    }

    def execute(self, expression):
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return str(e)


def calculator(expression: str):

    try:
        result = eval(expression)
        return str(result)

    except Exception as e:
        return str(e)