from mini_agent.memory.models import Memory

class MemoryStore:
    def __init__(self):
        self.memorys = []

    def add(self, memory: Memory):
        self.memorys.append(memory)

    def get_content(self):
        return "\n".join(m.content for m in self.memorys)
    
    def list_memory(self):
        return self.memorys
