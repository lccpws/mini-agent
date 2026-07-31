from mini_agent.tools.base_tool import BaseTool
from mini_agent.sandbox.python_executor import PythonExecutor
from mini_agent.sandbox.guard import validate_code

class PythonTool(BaseTool):
    name = "python tool"
    description = "python执行工具"
    capabilities = ["python", "python execute"]

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin", "user"]
    required_permissions = ["network"]
    risk_level = 4

    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码"
            }
        },
        "required": ["code"]
    }

    def __init__(self):
        self.executor = PythonExecutor()

    def execute(self, code=None, command=None, **kwargs):
        code = code or command
        if not code:
            return "缺少 code 参数"
        
        is_valid = validate_code(code)
        if not is_valid:
            return "代码不合法"
        
        stdout, stderr = self.executor.execute(code)
        
        if stderr:
            return f"执行错误：{stderr}"
        return stdout if stdout else "执行完成（无输出）"