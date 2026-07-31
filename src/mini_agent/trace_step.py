from dataclasses import dataclass

@dataclass
class TraceStep:
    question: str
    step: int 
    thought: str     # LLM 的思考
    action: str | None = None      # 工具名
    args: dict | None = None       # 工具参数
    observation: str | None = None # 执行结果
    answer: str | None = None      # 最终答案

class TraceLogger:
    def __init__(self):
        self.logs = []
    
    def log(self, step: TraceStep):
        self.logs.append(step)

    def dump(self):
        for tracestep in self.logs:
            print(f"Step {tracestep.step}:")
            print(tracestep)
        