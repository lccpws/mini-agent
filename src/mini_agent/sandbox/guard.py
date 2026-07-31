import ast

# FORBIDDEN = [
#     "import os",
#     "import subprocess",
#     "open(",
#     "exec(",
#     "eval(",
#     "__import__"
# ]

def validate_code(code):
    tree = ast.parse(code)

    for node in ast.walk(tree):
        # 禁止 import os, subprocess
        if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in ["os", "subprocess"]:
                    return False
        
        # 禁止eval exec等函数调用
        elif isinstance(node, ast.Call):
            if node.func.id in ["open", "exec", "eval", "__import__"]:
                return False


    # for forbidden in FORBIDDEN:
    #     if forbidden in code:
    #         return False
    return True