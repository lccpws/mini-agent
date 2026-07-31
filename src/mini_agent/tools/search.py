from mini_agent.tools.base_tool import BaseTool

class SearchTool(BaseTool):
    capabilities = ["search"]
    description = "搜索"
    name = "search"

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin", "user"]
    required_permissions = ["network"]
    risk_level = 1

    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词"
            }
        },
        "required": ["query"]
    }

    def execute(self, query):
        return f"搜索结果: {query} 是一个AI系统相关问题"

def search(query: str):
    return f"搜索结果: {query} 是一个AI系统相关问题"