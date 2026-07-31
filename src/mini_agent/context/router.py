import json

from mini_agent.context.models import ContextRoute


class ContextRouter:
    """上下文路由器，通过LLM分析query决定需要哪些context源"""

    def __init__(self, llm):
        self.llm = llm

    def route(self, query: str) -> ContextRoute:
        if self.llm.client is None:
            return self._default_route(query)

        prompt = self._build_route_prompt(query)

        try:
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=[
                    {"role": "system", "content": self._get_route_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            output = response.choices[0].message.content
            return self._parse_response(output)
        except Exception as e:
            print(f"路由分析失败: {e}")
            return self._default_route(query)

    def _get_route_system_prompt(self) -> str:
        return """你是一个上下文路由器。分析用户的问题，判断需要哪些上下文信息来回答。

可选的上下文类型：
- system: 系统提示词（总是需要）
- user: 用户问题（总是需要）
- memory: 历史记忆（用户之前的偏好、事实等）
- rag: 知识库文档（外部知识）
- history: 对话历史（之前的观察结果、工具调用等）

请返回JSON格式：
{
    "needs_memory": true/false,
    "needs_rag": true/false,
    "needs_history": true/false,
    "reason": "简短说明原因"
}

注意：
- system和user总是true
- 根据问题类型判断是否需要其他上下文"""

    def _build_route_prompt(self, query: str) -> str:
        return f"用户问题：{query}\n\n请分析这个问题需要哪些上下文信息。"

    def _parse_response(self, output: str) -> ContextRoute:
        try:
            data = json.loads(output)
            return ContextRoute(
                needs_system=True,
                needs_user=True,
                needs_memory=data.get("needs_memory", True),
                needs_rag=data.get("needs_rag", True),
                needs_history=data.get("needs_history", True),
                reason=data.get("reason", "")
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"解析路由结果失败: {e}")
            return ContextRoute()

    def _default_route(self, query: str) -> ContextRoute:
        keywords_memory = ["之前", "上次", "记得", "以前", "记忆"]
        keywords_rag = ["知识", "文档", "资料", "论文", "介绍"]
        keywords_history = ["刚才", "工具", "执行", "调用", "结果"]

        needs_memory = any(kw in query for kw in keywords_memory)
        needs_rag = any(kw in query for kw in keywords_rag)
        needs_history = any(kw in query for kw in keywords_history)

        if not any([needs_memory, needs_rag, needs_history]):
            needs_memory = True
            needs_rag = True
            needs_history = True

        reason = "关键词匹配（默认模式）"
        if needs_memory:
            reason += "，检测到记忆相关关键词"
        if needs_rag:
            reason += "，检测到知识库相关关键词"
        if needs_history:
            reason += "，检测到历史相关关键词"

        return ContextRoute(
            needs_system=True,
            needs_user=True,
            needs_memory=needs_memory,
            needs_rag=needs_rag,
            needs_history=needs_history,
            reason=reason
        )