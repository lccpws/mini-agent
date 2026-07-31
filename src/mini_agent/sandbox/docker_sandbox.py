import subprocess
import tempfile
import os
from pathlib import Path


class DockerSandbox:
    def __init__(
        self,
        image: str = "python:3.11-slim",
        timeout: int = 30,
        memory_limit: str = "128m",
        cpu_limit: float = 0.5,
    ):
        self.image = image
        self.timeout = timeout
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit

    def execute(self, code: str) -> tuple:
        with tempfile.TemporaryDirectory() as tmp_dir:
            code_file = Path(tmp_dir) / "code.py"
            code_file.write_text(code, encoding="utf-8")

            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", self.memory_limit,
                "--cpus", str(self.cpu_limit),
                "--read-only",
                "--tmpfs", "/tmp:size=10m",
                "--security-opt", "no-new-privileges",
                "--cap-drop", "ALL",
                "--pids-limit", "50",
                "-v", f"{code_file}:/code.py:ro",
                self.image,
                "python", "/code.py"
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout
                )
                return result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                return None, f"执行超时：超过 {self.timeout} 秒"
            except FileNotFoundError:
                return None, "Docker 未安装或不在 PATH 中"
            except Exception as e:
                return None, f"执行错误：{str(e)}"
