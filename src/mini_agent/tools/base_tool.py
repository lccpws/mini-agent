from abc import ABC
from abc import abstractmethod

class BaseTool(ABC):
    name = ""
    description = ""
    capabilities = []

    version = "0.1.0"
    author = "Mini-Agent"

    allow_roles = []
    required_permissions = []
    risk_level = 1

    @abstractmethod
    def execute(self, **kwargs):
        pass