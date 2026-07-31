import json
from pathlib import Path
from mini_agent.memory.models import Memory, MemoryType, Importance


class MemoryConsolidator:
    """记忆整合器，将琐碎记忆整合为完整记忆"""

    def __init__(self, llm):
        self.llm = llm
        self.prompt_template = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "consolidate_prompt.txt"
        return prompt_path.read_text(encoding="utf-8")

    def consolidate(self, memories: list[Memory]) -> list[Memory]:
        """整合琐碎记忆"""
        if len(memories) < 2:
            return memories

        groups = self._group_memories(memories)

        consolidated = []
        for group in groups:
            if len(group) > 1:
                result = self._consolidate_group(group)
                consolidated.extend(result)
            else:
                consolidated.extend(group)

        return consolidated

    def _group_memories(self, memories: list[Memory]) -> list[list[Memory]]:
        """按类型分组"""
        groups = {}
        for memory in memories:
            key = memory.memory_type.value
            if key not in groups:
                groups[key] = []
            groups[key].append(memory)
        return list(groups.values())

    def _consolidate_group(self, group: list[Memory]) -> list[Memory]:
        """整合一组记忆"""
        memories_text = "\n".join([
            f"- [{m.memory_type.value}] {m.content}"
            for m in group
        ])

        prompt = self.prompt_template.format(memories=memories_text)

        try:
            messages = [
                {"role": "system", "content": "你是一个记忆整合器，只输出 JSON 格式。"},
                {"role": "user", "content": prompt}
            ]

            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                response_format={"type": "json_object"}
            )

            output = response.choices[0].message.content
            return self._parse_response(output, group)

        except Exception as e:
            print(f"记忆整合失败: {e}")
            return group

    def _parse_response(self, output: str, original_group: list[Memory]) -> list[Memory]:
        """解析 LLM 响应"""
        try:
            parsed = json.loads(output)
            items = parsed.get("consolidated", [])

            if not items:
                return original_group

            result = []
            for item in items:
                memory = self._to_memory(item)
                if memory:
                    result.append(memory)

            return result if result else original_group

        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {e}")
            return original_group

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
            source="consolidation",
            metadata={"reason": item.get("reason", "")}
        )
