from mini_agent.tools.base_tool import BaseTool
import os

class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "删除文件工具"
    capabilities = ["delete", "delete file"]

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin"]
    required_permissions = ["network"]
    risk_level = 4

    parameters_schema = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "要删除的文件路径"
            }
        },
        "required": ["file_path"]
    }

    def execute(self, file_path):
        try:
            result = self.delete_file(file_path)
            return str(result)
        except Exception as e:
            return str(e)

    def delete_file(self, file_path):
        if not os.path.exists(file_path):
            return f"文件 {file_path} 不存在"
        os.remove(file_path)
        return f"文件 {file_path} 删除成功"

