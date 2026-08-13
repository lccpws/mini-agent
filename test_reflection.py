import tempfile
import shutil
from pathlib import Path
from mini_agent.planner.models import Task, TaskStatus
from mini_agent.reflection.models import EvaluationResult, ReflectionResult, ReflectionRecord, Action
from mini_agent.reflection.evaluator import Evaluator
from mini_agent.reflection.reflection import Reflection
from mini_agent.reflection.memory import ReflectionMemory
from mini_agent.reflection.engine import ReflectionEngine


class TestEvaluator:
    def setup_method(self):
        self.evaluator = Evaluator(threshold=60.0)

    def test_none_result(self):
        task = Task(id="t1", description="test", objective="test")
        result = self.evaluator.evaluate(task, None)
        assert result.score == 0
        assert result.passed is False
        assert "空" in result.reason

    def test_dict_with_error(self):
        task = Task(id="t1", description="test", objective="test")
        result = self.evaluator.evaluate(task, {"error": "something failed"})
        assert result.score == 0
        assert result.passed is False
        assert "错误" in result.reason

    def test_dict_valid(self):
        task = Task(id="t1", description="test", objective="test")
        result = self.evaluator.evaluate(task, {"key": "value"})
        assert result.score == 100
        assert result.passed is True

    def test_dict_missing_required_field(self):
        task = Task(
            id="t1", description="test", objective="test",
            output_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        )
        result = self.evaluator.evaluate(task, {"weather": "sunny"})
        assert result.score == 30
        assert result.passed is False
        assert "city" in result.reason

    def test_dict_all_required_fields(self):
        task = Task(
            id="t1", description="test", objective="test",
            output_schema={"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
        )
        result = self.evaluator.evaluate(task, {"city": "Beijing"})
        assert result.score == 100
        assert result.passed is True

    def test_string_too_short(self):
        task = Task(id="t1", description="test", objective="test")
        result = self.evaluator.evaluate(task, "short")
        assert result.score == 30
        assert result.passed is False
        assert "短" in result.reason

    def test_string_valid(self):
        task = Task(id="t1", description="test", objective="test")
        result = self.evaluator.evaluate(task, "This is a long enough result string")
        assert result.score == 80
        assert result.passed is True

    def test_unknown_type(self):
        task = Task(id="t1", description="test", objective="test")
        result = self.evaluator.evaluate(task, 12345)
        assert result.score == 50
        assert result.passed is True


class TestReflection:
    def setup_method(self):
        self.reflection = Reflection()

    def test_no_error_with_suggestions(self):
        task = Task(id="t1", description="test", objective="test", input={"key": "value"})
        eval_result = EvaluationResult(score=30, passed=False, reason="low score", suggestions=["增加字段"])
        result = self.reflection.reflect(task, {"data": "ok"}, eval_result)
        assert result.reflected is True
        assert result.should_retry is True
        assert result.improved_input == {"key": "value"}

    def test_error_not_found(self):
        task = Task(id="t1", description="test", objective="test")
        eval_result = EvaluationResult(score=0, passed=False, reason="error")
        result = self.reflection.reflect(task, {"error": "未找到相关资源"}, eval_result)
        assert result.reflected is True
        assert "替代方案" in result.feedback

    def test_error_timeout(self):
        task = Task(id="t1", description="test", objective="test")
        eval_result = EvaluationResult(score=0, passed=False, reason="error")
        result = self.reflection.reflect(task, {"error": "执行超时"}, eval_result)
        assert result.reflected is True
        assert "简化" in result.feedback

    def test_error_permission(self):
        task = Task(id="t1", description="test", objective="test")
        eval_result = EvaluationResult(score=0, passed=False, reason="error")
        result = self.reflection.reflect(task, {"error": "权限不足 403"}, eval_result)
        assert result.reflected is True
        assert "权限" in result.feedback

    def test_should_not_retry_when_max_exceeded(self):
        task = Task(id="t1", description="test", objective="test", retry_count=3, max_retry=3)
        eval_result = EvaluationResult(score=30, passed=False, reason="low")
        result = self.reflection.reflect(task, {"error": "fail"}, eval_result)
        assert result.should_retry is False

    def test_should_not_retry_when_score_high(self):
        task = Task(id="t1", description="test", objective="test")
        eval_result = EvaluationResult(score=70, passed=False, reason="medium")
        result = self.reflection.reflect(task, {"data": "ok"}, eval_result)
        assert result.should_retry is False


class TestReflectionMemory:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.memory = ReflectionMemory(persist_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_search(self):
        record = ReflectionRecord(
            id="", capability="weather", failure_type="TRANSIENT",
            error_message="timeout", root_cause="network"
        )
        self.memory.save(record)
        results = self.memory.search_by_capability("weather")
        assert len(results) == 1
        assert results[0].capability == "weather"
        assert results[0].fail_count == 0

    def test_save_duplicate_increments_fail(self):
        record1 = ReflectionRecord(
            id="", capability="weather", failure_type="TRANSIENT",
            error_message="timeout error", root_cause="network"
        )
        record2 = ReflectionRecord(
            id="", capability="weather", failure_type="TRANSIENT",
            error_message="timeout error occurred", root_cause="network"
        )
        self.memory.save(record1)
        self.memory.save(record2)
        results = self.memory.search_by_capability("weather")
        assert len(results) == 1
        assert results[0].fail_count == 1

    def test_persistence(self):
        record = ReflectionRecord(
            id="", capability="search", failure_type="PERMANENT",
            error_message="not found", root_cause="missing"
        )
        self.memory.save(record)
        memory2 = ReflectionMemory(persist_dir=self.tmpdir)
        results = memory2.search_by_capability("search")
        assert len(results) == 1

    def test_apply_experience(self):
        record = ReflectionRecord(
            id="", capability="weather", failure_type="UNKNOWN",
            error_message="err", root_cause="cause",
            alternative_capability="weather_api", alternative_input={"city": "Beijing"}
        )
        self.memory.save(record)
        task = Task(id="t1", description="test", objective="test", capability="weather", input={"city": "Shanghai"})
        records = self.memory.search_by_capability("weather")
        self.memory.apply_experience(task, records[0])
        assert task.capability == "weather_api"
        assert task.input["city"] == "Beijing"
        assert records[0].success_count == 1

    def test_get_most_successful(self):
        for i in range(3):
            record = ReflectionRecord(
                id="", capability="weather", failure_type="UNKNOWN",
                error_message=f"err{i}", root_cause="cause"
            )
            record.success_count = i * 10
            self.memory.save(record)
        top = self.memory.get_most_successful("weather", limit=1)
        assert len(top) == 1
        assert top[0].success_count == 20

    def test_clear(self):
        record = ReflectionRecord(
            id="", capability="weather", failure_type="UNKNOWN",
            error_message="err", root_cause="cause"
        )
        self.memory.save(record)
        self.memory.clear()
        assert len(self.memory.records) == 0

    def test_get_replan_context_empty(self):
        ctx = self.memory.get_replan_context(["weather"])
        assert ctx == ""

    def test_get_replan_context_with_records(self):
        r1 = ReflectionRecord(
            id="", capability="weather", failure_type="TRANSIENT",
            error_message="timeout", root_cause="network slow",
            alternative_capability="weather_api", alternative_input={"city": "Beijing"}
        )
        r1.success_count = 3
        r2 = ReflectionRecord(
            id="", capability="weather", failure_type="PERMANENT",
            error_message="not found", root_cause="invalid city"
        )
        r2.fail_count = 2
        self.memory.save(r1)
        self.memory.save(r2)
        ctx = self.memory.get_replan_context(["weather"])
        assert "weather" in ctx
        assert "历史经验" in ctx
        assert "成功方案" in ctx
        assert "失败原因" in ctx

    def test_get_replan_context_multiple_caps(self):
        r1 = ReflectionRecord(
            id="", capability="search", failure_type="UNKNOWN",
            error_message="err", root_cause="cause"
        )
        r1.fail_count = 1
        self.memory.save(r1)
        ctx = self.memory.get_replan_context(["search", "weather"])
        assert "search" in ctx
        assert "weather" not in ctx


class TestReflectionEngine:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.engine = ReflectionEngine(persist_dir=self.tmpdir, min_improvement=3.0)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_evaluate_and_reflect_pass(self):
        task = Task(id="t1", description="test", objective="test")
        eval_result, refl_result = self.engine.evaluate_and_reflect(task, {"key": "value"})
        assert eval_result.passed is True
        assert eval_result.score == 100
        assert refl_result.reflected is False

    def test_evaluate_and_reflect_fail(self):
        task = Task(id="t1", description="test", objective="test")
        eval_result, refl_result = self.engine.evaluate_and_reflect(task, {"error": "未找到相关资源"})
        assert eval_result.passed is False
        assert refl_result.reflected is True

    def test_score_history_recorded(self):
        task = Task(id="t1", description="test", objective="test")
        self.engine.evaluate_and_reflect(task, {"key": "value"})
        assert "t1" in self.engine.score_history
        assert self.engine.score_history["t1"] == [100.0]

    def test_convergence_stops_retry(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3)
        eval_r = EvaluationResult(score=58.0, passed=False, reason="low")
        refl_r = ReflectionResult(reflected=True, should_retry=True, action=Action.RETRY)

        self.engine.score_history["t1"] = [55.0, 58.0]
        task.retry_count = 1
        assert self.engine.should_retry(task, eval_r, refl_r) is True

        self.engine.score_history["t1"].append(59.0)
        eval_r2 = EvaluationResult(score=59.0, passed=False, reason="low")
        task.retry_count = 2
        assert self.engine.should_retry(task, eval_r2, refl_r) is False

    def test_first_attempt_allows_retry(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3)
        self.engine.score_history["t1"] = [55.0]
        eval_r = EvaluationResult(score=55.0, passed=False, reason="low")
        refl_r = ReflectionResult(reflected=True, should_retry=True, action=Action.RETRY)
        task.retry_count = 0
        assert self.engine.should_retry(task, eval_r, refl_r) is True

    def test_max_retry_stops(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3, retry_count=3)
        eval_r = EvaluationResult(score=55.0, passed=False, reason="low")
        refl_r = ReflectionResult(reflected=True, should_retry=True, action=Action.RETRY)
        assert self.engine.should_retry(task, eval_r, refl_r) is False

    def test_passed_stops(self):
        task = Task(id="t1", description="test", objective="test")
        eval_r = EvaluationResult(score=100.0, passed=True, reason="ok")
        refl_r = ReflectionResult(reflected=False)
        assert self.engine.should_retry(task, eval_r, refl_r) is False

    def test_no_reflection_stops(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3)
        eval_r = EvaluationResult(score=50.0, passed=False, reason="low")
        refl_r = ReflectionResult(reflected=False, should_retry=False, action=Action.NONE)
        assert self.engine.should_retry(task, eval_r, refl_r) is False

    def test_replan_action_stops_retry(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3, retry_count=1)
        eval_r = EvaluationResult(score=30.0, passed=False, reason="low")
        refl_r = ReflectionResult(reflected=True, should_retry=True, action=Action.REPLAN)
        assert self.engine.should_retry(task, eval_r, refl_r) is False
        assert self.engine.should_replan(task, eval_r, refl_r, question="test question") is True

    def test_replan_requires_retry_first(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3, retry_count=0)
        eval_r = EvaluationResult(score=30.0, passed=False, reason="low")
        refl_r = ReflectionResult(reflected=True, should_retry=True, action=Action.REPLAN)
        assert self.engine.should_replan(task, eval_r, refl_r, question="test question") is False

    def test_apply_reflection_updates_input(self):
        task = Task(id="t1", description="test", objective="test", input={"old": "value"})
        refl_r = ReflectionResult(improved_input={"new": "value"})
        updated = self.engine.apply_reflection(task, refl_r)
        assert updated.input == {"new": "value"}

    def test_apply_reflection_updates_capability(self):
        task = Task(id="t1", description="test", objective="test", capability="old_tool")
        refl_r = ReflectionResult(suggested_capability="new_tool")
        updated = self.engine.apply_reflection(task, refl_r)
        assert updated.capability == "new_tool"

    def test_full_retry_flow(self):
        task = Task(id="t1", description="test", objective="test", max_retry=3)

        eval1, refl1 = self.engine.evaluate_and_reflect(task, {"error": "未找到相关资源"})
        assert eval1.passed is False
        assert self.engine.score_history["t1"] == [0.0]
        assert self.engine.should_retry(task, eval1, refl1) is True

        task = self.engine.apply_reflection(task, refl1)
        task.retry_count = 1

        eval2, refl2 = self.engine.evaluate_and_reflect(task, {"error": "超时"})
        assert eval2.passed is False
        assert self.engine.score_history["t1"] == [0.0, 0.0]
        assert self.engine.should_retry(task, eval2, refl2) is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
