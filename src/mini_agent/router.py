import json
from typing import Optional, Dict, Any, Tuple
from mini_agent.tool_manager import AgentToolManager

class CapabilityRouter:
    def __init__(self, tool_manager: AgentToolManager):
        self.tool_manager = tool_manager

    def detect_capability(self, prompt: str) -> str | None:
        pass

    def find_tools(self, capability: str):
        pass

    def select_tool(self, prompt: str, tools: list):
        pass

    def route(self, decision):
        if isinstance(decision, str):
            json_output = self.parse(decision)
        else:
            json_output = decision
        
        if not json_output:
            return None
        
        # 支持 capability 或 tool 字段
        capability = json_output.get("capability") or json_output.get("tool", "")
        
        tools = self.tool_manager.find_by_capability(capability)
        if not tools:
            # 尝试按名称查找
            try:
                tool = self.tool_manager.get_tool_by_name(capability)
                return {
                    "tool_name": tool.name,
                    "args": json_output.get("args", {})
                }
            except ValueError:
                return None
        
        return {
            "tool_name": tools[0].name,
            "args": json_output.get("args", {})
        }

    def parse(self, llm_output: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 返回的 JSON 字符串"""
        try:
            # 1. 清理输出（去除可能的 Markdown 标记、多余空格等）
            cleaned_output = self._clean_output(llm_output)
            
            # 2. 解析 JSON
            parsed = json.loads(cleaned_output)
            
            # 3. 验证必要字段
            if not self._validate(parsed):
                return None
            
            return parsed
            
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return None
        except Exception as e:
            print(f"解析过程中发生错误: {e}")
            return None
    
    def _clean_output(self, output: str) -> str:
        """清理 LLM 输出，提取 JSON 部分"""
        # 移除可能的 Markdown 代码块标记
        output = output.strip()
        
        # 处理 ```json ... ``` 格式
        if output.startswith("```"):
            # 找到第一个 { 和最后一个 }
            start = output.find("{")
            end = output.rfind("}")
            if start != -1 and end != -1:
                output = output[start:end+1]
        
        return output.strip()
    
    def _validate(self, parsed: Dict) -> bool:
        """验证解析结果的必要字段"""
        required_fields = ["capability", "reason", "args"]
        
        for field in required_fields:
            if field not in parsed:
                print(f"缺少必要字段: {field}")
                return False
        
        if not isinstance(parsed.get("args"), dict):
            print("args 字段必须是字典类型")
            return False
        
        return True