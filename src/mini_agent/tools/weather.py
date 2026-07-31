from mini_agent.tools.base_tool import BaseTool

class WeatherTool(BaseTool):
    name = "weather"
    description = "查询天气"
    capabilities = ["weather"]

    version = "1.0.0"
    author = "MiniAgent"
    allow_roles = ["admin", "user"]
    required_permissions = ["network"]
    risk_level = 1

    parameters_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如：北京、上海、广州"
            }
        },
        "required": ["city"]
    }

    def execute(self, city):
        return f"{city}天气晴朗 25度"
