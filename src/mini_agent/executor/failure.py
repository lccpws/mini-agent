from mini_agent.planner.models import FailureType


class FailureClassifier:
    TRANSIENT_EXCEPTIONS = (
        TimeoutError,
        ConnectionError,
        ConnectionResetError,
        ConnectionAbortedError,
    )

    PERMANENT_EXCEPTIONS = (
        PermissionError,
        FileNotFoundError,
        ModuleNotFoundError,
    )

    TRANSIENT_KEYWORDS = ["timeout", "超时", "连接", "网络", "临时", "暂时", "重试", "retry"]
    PERMANENT_KEYWORDS = ["403", "404", "权限", "不存在", "not found", "forbidden"]

    @classmethod
    def classify(cls, error: Exception) -> str:
        if isinstance(error, cls.TRANSIENT_EXCEPTIONS):
            return FailureType.TRANSIENT

        if isinstance(error, cls.PERMANENT_EXCEPTIONS):
            return FailureType.PERMANENT

        error_msg = str(error).lower()
        return cls.classify_message(error_msg)

    @classmethod
    def classify_message(cls, error_msg: str) -> str:
        error_msg_lower = error_msg.lower()

        for keyword in cls.TRANSIENT_KEYWORDS:
            if keyword in error_msg_lower:
                return FailureType.TRANSIENT

        for keyword in cls.PERMANENT_KEYWORDS:
            if keyword in error_msg_lower:
                return FailureType.PERMANENT

        return FailureType.UNKNOWN

    @classmethod
    def should_retry(cls, failure_type: str) -> bool:
        return failure_type in [FailureType.TRANSIENT, FailureType.UNKNOWN]
