import json

from mini_agent.context import (
    ContextItem,
    ContextManager,
    ContextPolicy,
    ContextRoute,
    ContextRouter,
    ContextSelector,
    ContextSource,
)
from mini_agent.context.token_counter import TokenCounterFactory
from mini_agent.knowledge import KnowledgeBase, KnowledgeRetriever
from mini_agent.memory.extractor import MemoryExtractor
from mini_agent.memory.manager import MemoryManager
from mini_agent.memory.models import MemoryType
from mini_agent.planner import LLMPlanner, Plan, PlanStep, PlanStatus, StepStatus, get_ready_steps
from mini_agent.runner import ReActController
from mini_agent.state import AgentState
from mini_agent.trace_step import TraceLogger, TraceStep


class ReactAgent:
    """ReAct Agent，支持纯 ReAct 和 Plan + ReAct 两种模式"""

    def __init__(self, controller: ReActController, planner: LLMPlanner = None, max_steps: int = 5, debug_context: bool = False, knowledge_base: KnowledgeBase = None, policy: ContextPolicy = None):
        self.controller = controller
        self.planner = planner
        self.max_steps = max_steps
        self.tracelog = TraceLogger()
        self.memory_manager = MemoryManager(persist_dir="memory_data")
        self.memory_extractor = MemoryExtractor(controller.llm)
        self.context_manager = ContextManager(
            selector=ContextSelector(),
            policy=policy or ContextPolicy(),
            total_tokens=8000,
            output_tokens=2000
        )
        self.context_router = ContextRouter(controller.llm)
        
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.knowledge_retriever = KnowledgeRetriever(
            self.knowledge_base,
            token_counter=controller.llm.token_counter
        )
        
        if debug_context:
            controller.llm.debugger.enabled = True

    def run(self, question: str, role: str = "user", mode: str = "react") -> str:
        self.memory_manager.add(question, MemoryType.QUESTION, scope="working")

        memories = self.memory_manager.search(question, top_k=3)
        state = AgentState(question=question, memories=memories)

        if memories:
            print(f"\n检索到 {len(memories)} 条相关记忆:")
            for m in memories:
                print(f"  - {m}")

        if mode == "plan_react" and self.planner:
            result = self._run_plan_react(state, role)
        else:
            result = self._run_react(state, role)

        self._extract_long_term_memories()

        return result

    def _extract_long_term_memories(self):
        """从对话历史中提取值得长期保存的记忆"""
        recent_memories = self.memory_manager.short_term.get_recent(10)

        if len(recent_memories) < 2:
            return

        print("\n正在提取长期记忆...")

        conversation = []
        for m in recent_memories:
            if m.memory_type == MemoryType.QUESTION:
                conversation.append({"role": "user", "content": m.content})
            elif m.memory_type == MemoryType.ANSWER:
                conversation.append({"role": "assistant", "content": m.content})

        if not conversation:
            return

        memories = self.memory_extractor.extract(conversation)

        if memories:
            print(f"提取到 {len(memories)} 条长期记忆:")
            for m in memories:
                self.memory_manager.promote_to_long_term(m)
                print(f"  - [{m.memory_type.value}] {m.content}")
        else:
            print("未提取到长期记忆")

    def _build_context_items(self, state: AgentState) -> tuple[list[ContextItem], ContextRoute]:
        """将AgentState转换为ContextItem列表，返回(items, route)"""
        route = self.context_router.route(state.question)
        
        print(f"\n[路由分析] {route.reason}")
        print(f"  - memory: {route.needs_memory}")
        print(f"  - rag: {route.needs_rag}")
        print(f"  - history: {route.needs_history}")
        
        items = []

        system_prompt = self.controller.llm.system_prompt
        items.append(ContextItem(
            content=system_prompt,
            source=ContextSource.SYSTEM,
            priority=100,
            token_count=self.controller.llm.estimate_tokens(system_prompt)
        ))

        items.append(ContextItem(
            content=f"问题：{state.question}",
            source=ContextSource.USER,
            priority=90,
            token_count=self.controller.llm.estimate_tokens(state.question)
        ))

        if route.needs_memory:
            for i, memory in enumerate(state.memories):
                content = f"记忆：{memory.content}"
                items.append(ContextItem(
                    content=content,
                    source=ContextSource.MEMORY,
                    priority=50 - i,
                    token_count=self.controller.llm.estimate_tokens(content)
                ))

        if route.needs_rag:
            rag_items = self.knowledge_retriever.search(state.question, top_k=3)
            items.extend(rag_items)

        if route.needs_history:
            for i, obs in enumerate(state.observations):
                content = f"观察：{obs}"
                items.append(ContextItem(
                    content=content,
                    source=ContextSource.HISTORY,
                    priority=30 - i,
                    token_count=self.controller.llm.estimate_tokens(content)
                ))

        return items, route

    def _run_react(self, state: AgentState, role: str) -> str:
        """纯 ReAct 模式"""
        tool_continuous_times = 0
        last_tool = None

        state.context_items, route = self._build_context_items(state)
        state.context_items = self.context_manager.build_context(
            state.context_items,
            route=route,
            total_tokens=8000,
            output_tokens=2000
        )

        for step in range(self.max_steps):
            if tool_continuous_times >= 3:
                print(f"工具 {last_tool} 连续执行超过 3 次，终止循环")
                break

            decision = self.controller.step(state)
            print(f"第{step+1}步: {decision}")

            trace_step = TraceStep(
                question=state.question,
                step=step + 1,
                thought=decision.get("thought", "")
            )

            if decision["type"] == "answer":
                trace_step.answer = decision.get("content", "")

                self.memory_manager.add(state.question, MemoryType.QUESTION, scope="short_term")
                self.memory_manager.add(trace_step.answer, MemoryType.ANSWER, scope="short_term")

                self.tracelog.log(trace_step)
                break

            if decision["type"] == "tool":
                trace_step.action = decision.get("tool", "")
                trace_step.args = decision.get("args", {})

                if last_tool == trace_step.action:
                    tool_continuous_times += 1
                else:
                    last_tool = trace_step.action
                    tool_continuous_times = 1

                observation = self.controller.execute_tool(decision, role)
                trace_step.observation = observation

                tool_result = f"调用 {trace_step.action} 返回: {observation}"
                self.memory_manager.add(tool_result, MemoryType.TOOL_RESULT, scope="short_term")

                state["observations"] = str(observation)

            self.tracelog.log(trace_step)

        self.tracelog.dump()
        return self._format_trace()

    def _run_plan_react(self, state: AgentState, role: str) -> str:
        """Plan + ReAct 模式"""
        state.context_items, route = self._build_context_items(state)
        state.context_items = self.context_manager.build_context(
            state.context_items,
            route=route,
            total_tokens=8000,
            output_tokens=2000
        )

        goal = state.question
        context = "\n".join([item.content for item in state.context_items])

        max_replan_attempts = 3
        replan_count = 0

        while replan_count <= max_replan_attempts:
            print("=" * 50)
            print(f"Phase 1: Planning (尝试 {replan_count + 1})")
            print("=" * 50)

            plan = self.planner.create_plan(goal, context)
            plan.status = PlanStatus.RUNNING

            print(f"目标: {plan.goal}")
            print(f"步骤数: {len(plan.steps)}")
            for step in plan.steps:
                deps = ", ".join(step.dependencies) if step.dependencies else "无"
                print(f"  - [{step.id}] {step.task} (依赖: {deps}, 能力: {step.capability})")

            print("\n" + "=" * 50)
            print("Phase 2: Execution")
            print("=" * 50)

            step_counter = 0
            execution_failed = False

            while True:
                ready_steps = get_ready_steps(plan)

                if not ready_steps:
                    all_completed = all(s.status == StepStatus.COMPLETED for s in plan.steps)
                    if all_completed:
                        plan.status = PlanStatus.COMPLETED
                        print("\n计划执行完成")
                    else:
                        plan.status = PlanStatus.FAILED
                        print("\n没有可执行的步骤（可能存在循环依赖或步骤失败）")
                    break

                for step in ready_steps:
                    step_counter += 1
                    step.status = StepStatus.RUNNING
                    print(f"\nStep {step_counter}: [{step.id}] {step.task}")

                    trace_step = TraceStep(
                        question=state.question,
                        step=step_counter,
                        thought=f"执行计划步骤: {step.task}"
                    )

                    if step.capability == "answer":
                        if step.dependencies:
                            summary_parts = []
                            for dep_id in step.dependencies:
                                dep_step = next((s for s in plan.steps if s.id == dep_id), None)
                                if dep_step and dep_step.result:
                                    summary_parts.append(f"{dep_step.task}: {dep_step.result}")
                            state["observations"] = "\n".join(summary_parts)

                        decision = self.controller.step(state)
                        answer = decision.get("content", "无法生成答案")

                        self.memory_manager.add(state.question, MemoryType.QUESTION, scope="short_term")
                        self.memory_manager.add(answer, MemoryType.ANSWER, scope="short_term")

                        trace_step.answer = answer
                        step.status = StepStatus.COMPLETED
                        step.result = answer
                        self.tracelog.log(trace_step)
                        print(f"  答案: {answer}")
                        plan.status = PlanStatus.COMPLETED
                        self.tracelog.dump()
                        return self._format_trace()

                    decision = {
                        "type": "tool",
                        "tool": step.capability,
                        "args": {}
                    }

                    trace_step.action = step.capability
                    trace_step.args = decision["args"]

                    try:
                        observation = self.controller.execute_tool(decision, role)
                        trace_step.observation = observation
                        step.status = StepStatus.COMPLETED
                        step.result = str(observation)

                        tool_result = f"调用 {step.capability} 返回: {observation}"
                        self.memory_manager.add(tool_result, MemoryType.TOOL_RESULT, scope="short_term")

                        state["observations"] = str(observation)

                        self.tracelog.log(trace_step)
                        print(f"  结果: {observation}")
                    except Exception as e:
                        step.status = StepStatus.FAILED
                        step.result = str(e)
                        execution_failed = True
                        print(f"  失败: {e}")
                        self.tracelog.log(trace_step)
                        break

                if execution_failed:
                    break

            if plan.status == PlanStatus.COMPLETED:
                break

            if execution_failed:
                replan_count += 1
                if replan_count <= max_replan_attempts:
                    print(f"\n计划执行失败，准备重新规划 (尝试 {replan_count}/{max_replan_attempts})")
                    plan.status = PlanStatus.NEED_REPLAN
                    failed_steps = [s for s in plan.steps if s.status == StepStatus.FAILED]
                    context += f"\n\n上次执行失败的步骤: {', '.join([f'{s.id}: {s.result}' for s in failed_steps])}"
                else:
                    plan.status = PlanStatus.FAILED
                    print(f"\n已达到最大重试次数 ({max_replan_attempts})，计划执行失败")
            else:
                break

        self.tracelog.dump()
        return self._format_trace()

    def _format_trace(self) -> str:
        lines = []
        for step in self.tracelog.logs:
            lines.append(f"STEP {step.step}\n")

            if step.thought:
                lines.append(f"Thought:\n{step.thought}\n")

            if step.action:
                lines.append(f"Action:\n{step.action}\n")

            if step.observation:
                lines.append(f"Observation:\n{step.observation}\n")

            if step.answer:
                lines.append(f"Answer:\n{step.answer}\n")

        return "\n".join(lines)

    def get_memory_stats(self) -> dict:
        return self.memory_manager.stats()
