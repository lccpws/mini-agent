import json
from mini_agent.memory.models import Memory, MemoryType, Importance
from mini_agent.parser import OutputParser, ParseError


class MemoryExtractor:
    """记忆提取器，调用 LLM 从对话中提取值得长期保存的信息"""

    def __init__(self, llm):
        self.llm = llm
        self.parser = OutputParser()
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        return """你是一个记忆提取器。分析以下对话内容，提取值得长期保存的信息。

对话内容：
{conversation}

请返回 JSON 格式的记忆列表：
{{
  "memories": [
    {{
      "content": "提取的信息",
      "memory_type": "preference|fact|skill|project",
      "importance": "low|medium|high",
      "reason": "为什么值得保存"
    }}
  ]
}}

提取规则：
1. 用户偏好（喜欢、习惯、偏好）→ preference
2. 事实信息（个人信息、工作相关）→ fact
3. 技能能力（会什么、擅长什么）→ skill
4. 项目相关（项目信息、技术栈）→ project

重要性判断：
- high: 用户明确强调、多次提及、核心信息
- medium: 一般性信息、可能有用
- low: 随口提及、不太重要

如果没有值得保存的信息，返回空列表：
{{"memories": []}}"""

    def extract(self, conversation: list[dict]) -> list[Memory]:
        """从对话中提取值得长期保存的记忆"""
        conversation_text = self._format_conversation(conversation)
        prompt = self.prompt_template.format(conversation=conversation_text)

        try:
            messages = [
                {"role": "system", "content": "你是一个记忆提取器，只输出 JSON 格式。"},
                {"role": "user", "content": prompt}
            ]

            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            output = response.choices[0].message.content
            return self._parse_response(output)

        except Exception as e:
            print(f"记忆提取失败: {e}")
            return []

    def _format_conversation(self, conversation: list[dict]) -> str:
        """格式化对话内容"""
        lines = []
        for msg in conversation:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _parse_response(self, output: str) -> list[Memory]:
        """解析 LLM 响应"""
        try:
            parsed = json.loads(output)
            memories = parsed.get("memories", [])

            result = []
            for item in memories:
                memory = self._to_memory(item)
                if memory:
                    result.append(memory)

            return result

        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return []

    def _to_memory(self, item: dict) -> Memory:
        """转换为 Memory 对象"""
        content = item.get("content", "")
        if not content:
            return None

        memory_type_str = item.get("memory_type", "fact")
        memory_type_map = {
            "preference": MemoryType.PREFERENCE,
            "fact": MemoryType.FACT,
            "skill": MemoryType.SKILL,
            "project": MemoryType.PROJECT
        }
        memory_type = memory_type_map.get(memory_type_str, MemoryType.FACT)

        importance_str = item.get("importance", "medium")
        importance_map = {
            "low": Importance.LOW,
            "medium": Importance.MEDIUM,
            "high": Importance.HIGH
        }
        importance = importance_map.get(importance_str, Importance.MEDIUM)

        return Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            source="llm_extraction",
            metadata={"reason": item.get("reason", "")}
        )
