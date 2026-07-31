import json
from mini_agent.llm import LLM


class Planner:
    """规划器，负责制定执行计划"""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.plan_prompt = self._load_plan_prompt()

    def _load_plan_prompt(self) -> str:
        return """你是一个规划器。根据用户问题，制定执行计划。

系统可用工具：
- weather: 查询天气，参数 {"city": "城市名"}
- calculator: 计算数学表达式，参数 {"expression": "表达式"}
- search: 搜索信息，参数 {"query": "搜索关键词"}
- delete_file: 删除文件，参数 {"file_path": "文件路径"}
- python: 执行 Python 代码，参数 {"code": "代码"}

请返回 JSON 格式的计划：
{
  "reasoning": "为什么制定这个计划",
  "plan": [
    {"step": 1, "tool": "工具名", "args": {"参数": "值"}},
    {"step": 2, "tool": "工具名", "args": {"参数": "值"}}
  ]
}

注意：
1. 最后一步通常是直接回答，使用 tool: "answer"
2. 每一步的 args 必须符合工具的参数定义
3. 如果问题可以直接回答，plan 可以只包含一步 answer"""

    def plan(self, state) -> dict:
        """制定执行计划"""
    
    
        messages = [
            {"role": "system", "content": self.plan_prompt},
            {"role": "user", "content": f"问题：{state}"}
        ]

        if self.llm.client is None:
            return self._mock_plan(state)

        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        output = response.choices[0].message.content
        return self._parse_plan(output)

    def _parse_plan(self, output: str) -> dict:
        """解析计划"""
        try:
            parsed = json.loads(output)
            if "plan" not in parsed:
                return self._default_plan()
            return parsed
        except json.JSONDecodeError:
            return self._default_plan()

    def _default_plan(self) -> dict:
        return {
            "reasoning": "解析失败，使用默认计划",
            "plan": [{"step": 1, "tool": "answer", "args": {}}]
        }

    def _mock_plan(self, state) -> dict:
        """Mock 计划（用于测试）"""
        if "天气" in state.question:
            city = "北京"
            for c in ["上海", "广州", "深圳", "成都", "杭州"]:
                if c in state.question:
                    city = c
                    break
            return {
                "reasoning": "用户询问天气，需要先查询天气，然后回答",
                "plan": [
                    {"step": 1, "tool": "weather", "args": {"city": city}},
                    {"step": 2, "tool": "answer", "args": {}}
                ]
            }
        elif any(op in state.question for op in ["+", "-", "*", "/"]):
            return {
                "reasoning": "用户需要计算，直接调用计算器",
                "plan": [
                    {"step": 1, "tool": "calculator", "args": {"expression": state.question}},
                    {"step": 2, "tool": "answer", "args": {}}
                ]
            }
        else:
            return {
                "reasoning": "问题可以直接回答",
                "plan": [
                    {"step": 1, "tool": "answer", "args": {}}
                ]
            }
