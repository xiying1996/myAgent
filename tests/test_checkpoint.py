"""
test_checkpoint.py — CheckpointManager + Replay 测试套件
"""

import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from unittest import mock

import pytest

from checkpoint.checkpoint_manager import CheckpointManager
from checkpoint.replay import DebugReplayAgent, RecoveryReplayAgent, ReplayMixin
from core.agent import Agent, HistoryEntry
from core.budget import ExecutionBudget
from core.plan import Plan, Step
from core.state_machine import AgentState, RetryMode
from events.event import Event
from events.event_queue import PriorityEventQueue
from events.event_types import EventType
from events.raw_event_bus import Dispatcher, RawEventBus
from execution.llm_interface import MockLLM
from execution.llm_deepseek import DeepSeekLLM
from execution.step_runner import StepRunner
from scheduler.policy_engine import PolicyEngine
from scheduler.scheduler import Scheduler
from state.dependency_validator import DependencyValidator
from state.state_manager import StateManager


class TestCheckpointManager:
    def _make_agent(self) -> Agent:
        plan = Plan.create([
            Step("s0", "search", {"q": "x"}, output_schema={"url": "str"}),
            Step("s1", "fetch", {"url": ""}, input_bindings={"url": "s0.url"}, dependencies=["s0"]),
        ])
        agent = Agent.create(
            plan=plan,
            budget=ExecutionBudget(max_steps=5, max_llm_calls=4, max_replans=2),
            agent_id="agent_cp",
        )
        return agent

    def test_save_and_load_snapshot_round_trip(self, tmp_path: Path):
        manager = CheckpointManager(snapshot_dir=str(tmp_path), snapshot_interval_steps=1)
        agent = self._make_agent()
        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")
        agent.add_history(HistoryEntry.create("step_started", {"step_id": "s0"}))

        snapshot_id = manager.save_snapshot(agent, last_event_id="evt_1")
        loaded = manager.load_snapshot_from_file(agent.agent_id, snapshot_id)

        assert loaded is not None
        assert loaded.agent_id == agent.agent_id
        assert loaded.plan["steps"][0]["step_id"] == "s0"
        assert loaded.budget["max_steps"] == 5
        assert loaded.retry_mode is None
        assert loaded.history[0]["kind"] == "state_change"
        assert loaded.state_history[-1]["to"] == "WAITING"

    def test_trigger_if_needed_uses_completed_steps(self, tmp_path: Path):
        manager = CheckpointManager(snapshot_dir=str(tmp_path), snapshot_interval_steps=1)
        agent = self._make_agent()
        agent.plan.advance()

        triggered = manager.trigger_if_needed(agent, event_id="evt_step_done")
        assert triggered is True
        assert manager.load_snapshot(agent.agent_id) is not None


class TestReplay:
    def _make_snapshot_agent(self) -> Agent:
        plan = Plan.create([
            Step("s0", "search", {"q": "x"}),
            Step("s1", "fetch", {"url": ""}, dependencies=["s0"]),
        ])
        agent = Agent.create(plan=plan, agent_id="agent_replay")
        agent.transition(AgentState.RUNNING, "start s0")
        agent.transition(AgentState.WAITING, "submit s0")
        agent.plan.advance()
        agent.transition(AgentState.READY, "s0 ok")
        agent.transition(AgentState.RUNNING, "start s1")
        agent.transition(AgentState.WAITING, "submit s1")
        return agent

    def test_restore_from_snapshot_rebuilds_agent(self, tmp_path: Path):
        manager = CheckpointManager(snapshot_dir=str(tmp_path))
        agent = self._make_snapshot_agent()
        snapshot_id = manager.save_snapshot(agent, last_event_id="evt_waiting")
        snapshot = manager.load_snapshot_from_file(agent.agent_id, snapshot_id)

        restored = ReplayMixin.restore_from_snapshot(snapshot)
        assert restored.agent_id == agent.agent_id
        assert restored.state == AgentState.WAITING
        assert restored.current_step().step_id == "s1"
        assert len(restored.state_history) == len(agent.state_history)

    def test_recovery_replay_applies_pending_events(self, tmp_path: Path):
        manager = CheckpointManager(snapshot_dir=str(tmp_path))
        agent = self._make_snapshot_agent()
        snapshot_id = manager.save_snapshot(agent, last_event_id="evt_waiting")
        snapshot = manager.load_snapshot_from_file(agent.agent_id, snapshot_id)

        event_log = [
            {
                "event_id": "evt_waiting",
                "agent_id": agent.agent_id,
                "event_type": "STATE_UPDATE",
                "payload": {},
                "timestamp": 1.0,
                "priority": "normal",
                "is_plan_snapshot": False,
            },
            {
                "event_id": "evt_result",
                "agent_id": agent.agent_id,
                "event_type": "TOOL_RESULT",
                "payload": {"step_id": "s1", "tool_name": "fetch", "result": {"html": "ok"}},
                "timestamp": 2.0,
                "priority": "normal",
                "is_plan_snapshot": False,
            },
        ]

        replay = RecoveryReplayAgent(snapshot=snapshot, event_log=event_log)
        recovered = replay.recover()

        assert recovered.current_step() is None
        assert recovered.state == AgentState.READY
        assert "s1" in recovered.plan.completed_step_ids

    def test_debug_replay_tracks_plan_snapshots(self):
        event_log = [
            {
                "event_id": "evt_plan",
                "agent_id": "a1",
                "event_type": "PLAN_UPDATE",
                "payload": {
                    "step_id": "s0",
                    "plan": {
                        "plan_id": "p1",
                        "current_index": 0,
                        "completed_ids": [],
                        "steps": [{"step_id": "s0"}, {"step_id": "s1"}],
                    },
                    "agent_state": "WAITING",
                },
                "timestamp": 1.0,
                "priority": "normal",
                "is_plan_snapshot": True,
            },
            {
                "event_id": "evt_result_1",
                "agent_id": "a1",
                "event_type": "TOOL_RESULT",
                "payload": {"step_id": "s0", "result": {}},
                "timestamp": 2.0,
                "priority": "normal",
                "is_plan_snapshot": False,
            },
            {
                "event_id": "evt_result_2",
                "agent_id": "a1",
                "event_type": "TOOL_RESULT",
                "payload": {"step_id": "s1", "result": {}},
                "timestamp": 3.0,
                "priority": "normal",
                "is_plan_snapshot": False,
            },
        ]

        replay = DebugReplayAgent(event_log=event_log)
        result = replay.replay_all()

        assert result["plan_snapshot_count"] == 1
        assert result["final_index"] == 2
        assert result["final_state"] == "DONE"


class TestSchedulerCheckpointIntegration:
    class _DummyExecutor:
        def __init__(self):
            self.submitted = []

        def submit(self, action):
            self.submitted.append(action)

    def test_scheduler_emits_plan_snapshot_and_saves_checkpoint(self, tmp_path: Path):
        queue = PriorityEventQueue()
        bus = RawEventBus()
        Dispatcher(queue).attach(bus)
        state_mgr = StateManager(event_log_dir=str(tmp_path / "logs"))
        checkpoint_mgr = CheckpointManager(snapshot_dir=str(tmp_path / "snaps"), snapshot_interval_steps=99)
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=state_mgr,
            step_runner=StepRunner(MockLLM(simulate_delay_s=0)),
            dependency_validator=DependencyValidator(),
            policy_engine=PolicyEngine(),
            checkpoint_manager=checkpoint_mgr,
        )
        dummy = self._DummyExecutor()
        sched.set_tool_executor(dummy)

        plan = Plan.create([Step("s0", "primary", {"q": "x"})])
        agent = Agent.create(plan=plan, agent_id="agent_plan_snapshot")
        state_mgr.add_agent(agent)
        sched._dep_validator.register_plan(plan)
        sched._completed_outputs[agent.agent_id] = {}

        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")
        agent.transition(AgentState.RETRYING, "timeout", retry_mode=RetryMode.REPLAN_MODE)

        sched._dispatch_agent(agent)

        evt = queue.get_nowait()
        assert evt.event_type == EventType.PLAN_UPDATE
        assert evt.is_plan_snapshot is True
        assert evt.payload["plan"]["steps"][0]["fallback_chain"][0]["tool"] == "primary_v2"

        sched._process_event(evt)
        snapshot = checkpoint_mgr.load_snapshot(agent.agent_id)
        assert snapshot is not None
        assert snapshot.last_event_id == evt.event_id

    def test_scheduler_deepseek_replan_is_logged_and_replayable(self, tmp_path: Path):
        queue = PriorityEventQueue()
        bus = RawEventBus()
        Dispatcher(queue).attach(bus)
        state_mgr = StateManager(event_log_dir=str(tmp_path / "logs"))
        checkpoint_mgr = CheckpointManager(snapshot_dir=str(tmp_path / "snaps"), snapshot_interval_steps=99)
        llm = DeepSeekLLM(api_key="test-key")
        sched = Scheduler(
            event_queue=queue,
            raw_event_bus=bus,
            state_manager=state_mgr,
            step_runner=StepRunner(llm),
            dependency_validator=DependencyValidator(),
            policy_engine=PolicyEngine(),
            checkpoint_manager=checkpoint_mgr,
        )
        dummy = self._DummyExecutor()
        sched.set_tool_executor(dummy)

        plan = Plan.create([Step("s0", "primary", {"q": "x"})])
        agent = Agent.create(plan=plan, agent_id="agent_deepseek_snapshot")
        state_mgr.add_agent(agent)
        sched._dep_validator.register_plan(plan)
        sched._completed_outputs[agent.agent_id] = {}

        agent.transition(AgentState.RUNNING, "start")
        agent.transition(AgentState.WAITING, "submitted")
        agent.transition(AgentState.RETRYING, "timeout", retry_mode=RetryMode.REPLAN_MODE)

        raw_response = json.dumps({
            "new_fallbacks": [
                {"tool": "primary_v2", "params": {"q": "x", "mode": "replanned"}},
            ],
            "step_param_updates": {},
            "reasoning": "switch to primary_v2",
            "give_up": False,
        })
        mock_result = mock.MagicMock()
        mock_result.choices = [mock.MagicMock()]
        mock_result.choices[0].message.content = raw_response

        with mock.patch.object(
            llm._client.chat.completions,
            "create",
            return_value=mock_result,
        ):
            sched._dispatch_agent(agent)

        evt = queue.get_nowait()
        assert evt.event_type == EventType.PLAN_UPDATE
        assert evt.payload["llm"]["provider"] == "deepseek"
        assert evt.payload["llm"]["model"] == llm.model_name
        assert evt.payload["llm"]["raw_response"] == raw_response
        assert evt.payload["llm"]["normalized_result"]["new_fallbacks"][0]["tool"] == "primary_v2"
        assert dummy.submitted[0].tool_name == "primary_v2"
        assert dummy.submitted[0].params["mode"] == "replanned"

        sched._process_event(evt)

        snapshot = checkpoint_mgr.load_snapshot(agent.agent_id)
        assert snapshot is not None
        assert snapshot.last_event_id == evt.event_id

        event_log = state_mgr.load_event_log(agent.agent_id)
        replay = DebugReplayAgent(event_log=event_log)
        replay_result = replay.replay_all()
        assert replay_result["latest_llm_output"]["provider"] == "deepseek"
        assert replay_result["latest_llm_output"]["raw_response"] == raw_response

        recovered = RecoveryReplayAgent(snapshot=snapshot, event_log=event_log).recover()
        assert recovered.current_step() is not None
        assert recovered.current_step().fallback_chain[0].tool == "primary_v2"
