from mini_agent.llm import LLM


class ContextSummarizer:

    def __init__(self, llm: LLM):
        self.llm = llm
    
    def summarize(self, text: str, target_tokens: int):
        prompt = f"""
            请压缩下面的内容。
            要求：
            1. 保留事实
            2. 保留关键决策
            3. 保留未完成任务
            4. 删除重复内容
            5. 不要添加原文没有的信息
            目标长度：
            约 {target_tokens} tokens
            原文：
            {text}
        """
        return self.llm.generate(prompt)