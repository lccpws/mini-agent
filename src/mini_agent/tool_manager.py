from typing import Any, Dict
from pathlib import Path
import importlib
from mini_agent.tools.base_tool import BaseTool


class AgentToolManager:
    """Agent工具动态管理器"""

    def __init__(self, tools_dir: str = None):
        if tools_dir is None:
            self.tools_dir = Path(__file__).resolve().parent / "tools"
        else:
            self.tools_dir = Path(tools_dir)

        self.tools: Dict[str, BaseTool] = {}
        self._discover_tools()

    def _discover_tools(self):
        """自动发现工具"""
        for tool_file in self.tools_dir.glob("*.py"):
            if tool_file.name.startswith("_"):
                continue

            module_name = f"mini_agent.tools.{tool_file.stem}"
            self._load_tool_module(module_name)
    
    def _load_tool_module(self, module_name: str):
        """加载工具模块"""
        try:
            module = importlib.import_module(module_name)
            
            # 3. 查找工具函数
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseTool)
                    and attr is not BaseTool
                ):
                    tool = attr()
                    self.tools[tool.name] = tool
                                
        except Exception as e:
            print(f"加载工具模块 {module_name} 失败: {e}")

    def get_tool_schemas(self) -> list:
        """获取所有工具的schema（用于LLM）"""
        schemas = []
        
        for tool_name, tool_func in self.tools.items():
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_func.__doc__ or "",
                    "parameters": self._get_function_schema(tool_func)
                }
            }
            schemas.append(schema)
        
        return schemas
    
    def _get_function_schema(self, tool):
        if hasattr(tool, 'parameters_schema'):
            return tool.parameters_schema
        return {"type": "object", "properties": {}}
    
    def execute_tool(self, tool_name: str, **kwargs):
        """执行工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在: {tool_name}")
        
        return self.tools[tool_name].execute(**kwargs)
    
    def find_by_capability(self, capability):
        result = []
        for tool in self.tools.values():
            if capability in tool.capabilities:
                result.append(tool)
        return result
    
    def list_tools(self):
        tool_list = {}
        for tool in self.tools.values():
            tool_list[tool.name] = tool.description
        return tool_list
    
    def get_tool_matedata(self, tool_name):
        """获取工具元数据"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在: {tool_name}")
        
        tool = self.tools[tool_name]
        return {
            "name": tool.name,
            "description": tool.description,
            "capabilities": tool.capabilities,
            "version": tool.version,
            "risk_level": tool.risk_level,
            "author": tool.author,
            "required_permissions": tool.required_permissions,
            "allow_roles": tool.allow_roles
        }
    
    def list_tool_matedata(self):
        """获取所有工具的元数据"""
        return [self.get_tool_matedata(tool_name) for tool_name in self.tools]
    
    def get_tool_by_name(self, tool_name):
        """通过名称获取工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在: {tool_name}")
        return self.tools[tool_name]