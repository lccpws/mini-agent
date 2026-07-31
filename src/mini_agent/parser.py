import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


class ParseError(Exception):
    """解析错误"""
    pass


class OutputParser:
    """LLM 输出解析器"""

    REQUIRED_FIELDS = ["type"]
    TOOL_FIELDS = ["tool", "args"]
    ANSWER_FIELDS = ["content"]
    COMMON_FIELDS = ["thought"]

    def __init__(self, log_dir: str = None):
        if log_dir is None:
            self.log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        else:
            self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, raw_output: str) -> Dict[str, Any]:
        """解析 LLM 原始输出"""
        # 1. 清理输出
        cleaned = self._clean_output(raw_output)
        
        # 2. 记录原始输出
        self._log_raw_output(raw_output, cleaned)

        # 3. JSON 解析
        parsed = self._parse_json(cleaned)

        # 4. 校验字段
        validated = self._validate_fields(parsed)

        # 5. 标准化输出
        normalized = self._normalize(validated)

        return normalized

    def _clean_output(self, output: str) -> str:
        """清理 LLM 输出"""
        if not output:
            return ""
        
        output = output.strip()

        # 处理 ```json ... ``` 格式
        if output.startswith("```"):
            start = output.find("{")
            end = output.rfind("}")
            if start != -1 and end != -1:
                output = output[start:end + 1]

        # 处理可能的前后缀文本
        if not output.startswith("{"):
            start = output.find("{")
            if start != -1:
                output = output[start:]
        
        if not output.endswith("}"):
            end = output.rfind("}")
            if end != -1:
                output = output[:end + 1]

        return output.strip()

    def _parse_json(self, cleaned: str) -> Dict[str, Any]:
        """解析 JSON"""
        if not cleaned:
            raise ParseError("输出为空")
        
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ParseError(f"JSON 解析失败: {e}")

    def _validate_fields(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        """校验字段完整性"""
        # 检查 type 字段
        if "type" not in parsed:
            raise ParseError("缺少 type 字段")
        
        output_type = parsed["type"]
        if output_type not in ["tool", "answer"]:
            raise ParseError(f"无效的 type 值: {output_type}")

        # 根据 type 校验必填字段
        if output_type == "tool":
            for field in self.TOOL_FIELDS:
                if field not in parsed:
                    raise ParseError(f"tool 类型缺少 {field} 字段")
            
            # 校验 args 是字典
            if not isinstance(parsed.get("args"), dict):
                parsed["args"] = {}

        elif output_type == "answer":
            for field in self.ANSWER_FIELDS:
                if field not in parsed:
                    raise ParseError(f"answer 类型缺少 {field} 字段")

        # 补全 thought 字段
        if "thought" not in parsed:
            parsed["thought"] = ""

        return parsed

    def _normalize(self, validated: Dict[str, Any]) -> Dict[str, Any]:
        """标准化输出"""
        output_type = validated["type"]

        if output_type == "tool":
            return {
                "type": "tool",
                "thought": str(validated.get("thought", "")),
                "tool": str(validated.get("tool", "")),
                "args": validated.get("args", {})
            }
        else:
            return {
                "type": "answer",
                "thought": str(validated.get("thought", "")),
                "content": str(validated.get("content", ""))
            }

    def _log_raw_output(self, raw: str, cleaned: str):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / "llm_output.log"
        
        log_entry = {
            "timestamp": timestamp,
            "raw": raw[:500],  # 截断过长输出
            "cleaned": cleaned[:500]
        }
        
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
