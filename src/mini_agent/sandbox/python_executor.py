from mini_agent.sandbox.docker_sandbox import DockerSandbox


class PythonExecutor:
    def __init__(self):
        self.sandbox = DockerSandbox()

    def execute(self, code):
        return self.sandbox.execute(code)
