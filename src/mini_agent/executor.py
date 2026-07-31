# from mini_agent.registry import get_default_tools, ToolRegistry, search_tool
from mini_agent.tool_manager import AgentToolManager
from mini_agent.models import ToolInvocation
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import json
from datetime import datetime, timezone
from pathlib import Path
from mini_agent.guardrails.guardrail import Guardrail
from mini_agent.guardrails.security import RiskEvaluator
from mini_agent.guardrails.approval_manager import ApprovalManager

class Executor:
    def __init__(self, timeout_seconds=30):
        self.tools = AgentToolManager()
        self.invocation = ToolInvocation()
        self.timeout_seconds = timeout_seconds
        self.pool = ThreadPoolExecutor(max_workers=4)
        self.guardrail = Guardrail()
        self.risk_evaluator = RiskEvaluator()
        self.approval_manager = ApprovalManager()


        self.audit_log_path = (
            Path(__file__).resolve().parent.parent
            / "logs"
            / "tool_audit.jsonl"
        )

    def run_without_llm(self, user_input):
        tool_names = ", ".join(self.tools.names())
        return (
            "MiniAgent is ready. "
            f"Received: {user_input}. "
            f"Available tools: {tool_names}. "
            "Set OPENAI_API_KEY in a .env file to call the OpenAI API."
        )

    def execute(self, role, tool_name, args):
        self.invocation = ToolInvocation(role=role, tool_name=tool_name, args=args)
        self.invocation.start()
        approval = True


        tool = self.tools.get_tool_by_name(tool_name)

        if not self.guardrail.validate(tool, role):
            return "Permission Denied"
        
        if self.risk_evaluator.check(tool) >= 3:
            approval = self.approval_manager.request_approval(tool_name, args)
            if not approval:
                self.write_audit_log(role, tool_name, args, approval)
                return "Approval Denied"
            
        self.write_audit_log(role, tool_name, args, approval)


        if not tool:
            return "Tool not found"
        
        if not self.is_safe_args(args):
            return "Unsafe tool input blocked"
    
        try:
            future = self.pool.submit(self.tools.execute_tool, tool_name, **args)
            result = future.result(timeout=self.timeout_seconds)
            self.invocation.succeed(result)
            print(f"Tool {tool_name} invoked state {self.invocation.state}")
            return result
        except TimeoutError:
            future.cancel()
            self.invocation.fail(f"Tool Timeout: {tool_name} 超过 {self.timeout_seconds} 秒未返回")
            return f"Tool Timeout: {tool_name} 超过 {self.timeout_seconds} 秒未返回"
        except Exception as e:
            self.invocation.fail(e)
            return f"Tool Error: {str(e)}"
        
    def is_safe_args(self, args):
        dangerous_keywords = [
            "eval",
            "exec",
            "__import__",
            "os.system",
            "subprocess",
            "rm -rf",
            "open(",
            "shutil",
        ]

        text = str(args)

        for keyword in dangerous_keywords:
            if keyword in text:
                return False

        return True
    
    def write_audit_log(self, role, tool_name, args, approved: bool=True):
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        tool = self.tools.get_tool_by_name(tool_name)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "tool": tool_name,
            "tool_version": tool.version,
            "tool_author": tool.author,
            "risk_level": tool.risk_level,
            "args": args,
            "approved": approved
        }

        with self.audit_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            