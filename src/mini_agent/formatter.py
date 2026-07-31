from dataclasses import dataclass
from typing import Any, Optional
import json


@dataclass
class Observation:
    tool: str
    result: Any
    status: str  # "success" | "error" | "timeout" | "denied"
    error: Optional[str] = None

    def to_dict(self):
        data = {
            "tool": self.tool,
            "result": self.result,
            "status": self.status
        }
        if self.error:
            data["error"] = self.error
        return data

    def to_string(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ObservationFormatter:
    """标准化 observation 格式"""

    @staticmethod
    def format(tool_name: str, result: Any, status: str = "success", error: str = None) -> Observation:
        return Observation(
            tool=tool_name,
            result=result,
            status=status,
            error=error
        )

    @staticmethod
    def from_executor(tool_name: str, result: str) -> Observation:
        """从执行器结果创建 observation"""
        if result.startswith("Tool Error:"):
            return Observation(tool=tool_name, result=result, status="error", error=result)
        elif result.startswith("Tool Timeout:"):
            return Observation(tool=tool_name, result=result, status="timeout", error=result)
        elif result in ["Permission Denied", "Approval Denied"]:
            return Observation(tool=tool_name, result=result, status="denied", error=result)
        elif result.startswith("Unsafe tool input"):
            return Observation(tool=tool_name, result=result, status="denied", error=result)
        else:
            return Observation(tool=tool_name, result=result, status="success")
