class ApprovalManager:

    def request_approval(
            self,
            tool_name,
            args
    ):

        print(f"需要审批: {tool_name}")
        result = input("是否批准(y/n): ")
        return result == "y"