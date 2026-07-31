import os
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from mini_agent.parser import OutputParser, ParseError
from mini_agent.context.debugger import ContextDebugger
from mini_agent.context.token_counter import TokenCounterFactory, TokenCounter


class LLM:
    def __init__(self, model="gpt-4o", tools=None, debug_context: bool = False):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_CHAT_API_KEY"),
            base_url=os.getenv("OPENAI_CHAT_BASE_URL")
        ) if os.getenv("OPENAI_API_KEY") else None
        self.model = model
        self.tools = tools or []
        self.system_prompt = self._load_prompt()
        self.parser = OutputParser()
        self.debugger = ContextDebugger(enabled=debug_context)
        self.token_counter = TokenCounterFactory.create(model)

    def _load_prompt(self):
        prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "react_prompt.txt"
        return prompt_path.read_text(encoding="utf-8")

    def estimate_tokens(self, text: str) -> int:
        return self.token_counter.count_tokens(text)

    def _build_system_prompt(self):
        tools_info = "\n".join([
            f"- {tool.name}: {tool.description}\n  参数: {tool.parameters_schema.get('properties', {})}"
            for tool in self.tools
        ])
        
        return f"""{self.system_prompt}

可用工具及参数：
{tools_info}

请根据工具的参数定义返回正确的参数名。"""

    def run(self, state):
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_message(state)}
        ]

        if hasattr(state, 'context_items') and state.context_items:
            self.debugger.debug(state.context_items)

        if self.client is None:
            return self._mock_response(state)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_format={"type": "json_object"}
        )

        output = response.choices[0].message.content
        return self._parse_response(output)
    
    def generate(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            response_format={"type": "json_object"}
        )
        output = response.choices[0].message.content
        return self._parse_response(output)

    def _build_message(self, state):
        if hasattr(state, 'context_items') and state.context_items:
            return self._build_message_from_context(state.context_items)

        parts = [f"问题：{state.question}"]

        if hasattr(state, 'memories') and state.memories:
            memories_text = "\n".join(memory.content for memory in state.memories)
            parts.append(f"历史记忆：{memories_text}")

        if hasattr(state, 'observations') and state.observations:
            parts.append(f"观察结果：{state.observations}")

        return "\n".join(parts)

    def _build_message_from_context(self, context_items) -> str:
        parts = []
        for item in context_items:
            parts.append(item.content)
        return "\n\n".join(parts)

    def _parse_response(self, output):
        try:
            return self.parser.parse(output)
        except ParseError as e:
            print(f"解析错误: {e}")
            return {
                "type": "answer",
                "thought": f"解析失败: {e}",
                "content": "抱歉，处理过程中出现错误"
            }

    def _mock_response(self, state):
        question = state.question
        if "天气" in question:
            return {
                "type": "tool",
                "thought": "用户询问天气，需要调用天气工具",
                "tool": "weather",
                "args": {"city": "北京"}
            }
        elif any(op in question for op in ["+", "-", "*", "/"]):
            return {
                "type": "tool",
                "thought": "用户需要计算，调用数学工具",
                "tool": "math",
                "args": {"expression": question}
            }
        else:
            return {
                "type": "answer",
                "thought": "直接回答用户问题",
                "content": f"你好，关于「{question}」的问题..."
            }
