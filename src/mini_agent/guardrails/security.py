from enum import Enum
from mini_agent.tools.base_tool import BaseTool

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class RiskEvaluator:
    def evaluate(self, risk_level: RiskLevel) -> str:
        if risk_level == RiskLevel.LOW:
            return "Low risk"
        elif risk_level == RiskLevel.MEDIUM:
            return "Medium risk"
        elif risk_level == RiskLevel.HIGH:
            return "High risk"
        elif risk_level == RiskLevel.CRITICAL:
            return "Critical risk"
    
    def check(self, tool: BaseTool):
        risk_level = tool.risk_level
        return risk_level