from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from benchmarks.run_bench import verify_replay
from harness_agent import agent_runtime, core
from harness_agent.durable_engine import DurableEngine, Step, Tool
from harness_agent.tenant_policy import Principal, TenantPolicy, TenantToolAccess, ToolCall

ACTOR = Principal("alpha", "worker")
REVIEWER = Principal("alpha", "human")
OTHER_TENANT = Principal("beta", "worker")
STEPS = [Step("one", "effect", {"value": 1}), Step("two", "effect", {"value": 2})]


@pytest.fixture
def clock():
    now = [100.]
    return now, lambda: now[0]


@pytest.fixture
def policy(clock):
    _, now = clock
    return TenantPolicy(
        memberships={
            ("alpha", "worker"): ["operator"], ("alpha", "colleague"): ["operator"],
            ("alpha", "human"): ["approver"], ("beta", "worker"): ["operator"],
        },
        grants={("alpha", "operator"): ["effect", "read"], ("beta", "operator"): ["read"]},
        signing_key=b"x" * 32, high_impact={"effect"}, clock=now,
    )


def approve(engine, policy, step=None):
    pending = engine.run(ACTOR, "run", [STEPS[0]] if step is None else step)
    event = pending["authorization"]
    call = ToolCall(ACTOR, event["operation"], event["tool"], event["arguments_hash"])
    return policy.sign(call, REVIEWER)


def test_process_crashes_before_and_after_each_effect(tmp_path):
    assert verify_replay(tmp_path) == (6, 0)


def test_approval_binds_to_run_and_arguments(tmp_path, policy, caplog):
    seen = []
    engine = DurableEngine(
        tmp_path / "runs", policy, {"effect": Tool(lambda args, key, db: seen.append(args))}
    )
    caplog.set_level("INFO", logger="harness_agent.tenant_policy")
    approval = approve(engine, policy)
    assert engine.run(ACTOR, "other", [STEPS[0]], approvals={"one": approval})[
        "authorization"
    ]["decision"] == "approval_invalid"
    assert engine.resume(ACTOR, "run", approvals={"one": approval})["status"] == "completed"
    assert engine.resume(ACTOR, "run")["status"] == "completed"
    assert seen == [{"value": 1}]
    assert '"decision":"approval_invalid"' in caplog.text
    assert approval.signature not in caplog.text
    assert '"value":1' not in caplog.text


def test_policy_denies_role_from_different_tenant(policy):
    call = ToolCall.bind(OTHER_TENANT, "op", "effect", {})
    assert policy.authorize(call)["decision"] == "rbac_denied"


@pytest.mark.parametrize("field,value", [
    ("operation", "another-operation"), ("tool", "read"), ("arguments_hash", "changed"),
    ("principal", Principal("alpha", "colleague")),
])
def test_signature_cannot_be_transferred(policy, field, value):
    call = ToolCall.bind(ACTOR, "op", "effect", {"value": 1})
    approval = policy.sign(call, REVIEWER)
    assert policy.authorize(replace(call, **{field: value}), approval, high_impact=True)[
        "decision"
    ] == "approval_invalid"


def test_approval_expires_and_requires_separate_in_tenant_human(policy, clock):
    call = ToolCall.bind(ACTOR, "op", "effect", {})
    for approver in (ACTOR, Principal("beta", "human"), Principal("alpha", "colleague")):
        with pytest.raises(PermissionError):
            policy.sign(call, approver)
    approval = policy.sign(call, REVIEWER, ttl_seconds=1)
    clock[0][0] += 1
    assert policy.authorize(call, approval)["decision"] == "approval_invalid"


def test_resume_scopes_run_to_owner_and_rejects_plan_drift(tmp_path, policy):
    engine = DurableEngine(tmp_path / "runs", policy, {"effect": Tool(lambda *_: None)})
    approve(engine, policy)
    with pytest.raises(KeyError):
        engine.resume(OTHER_TENANT, "run")
    with pytest.raises(PermissionError):
        engine.run(Principal("alpha", "colleague"), "run", [STEPS[0]])
    with pytest.raises(ValueError, match="definition changed"):
        engine.run(ACTOR, "run", [STEPS[0]], version="2")
    with pytest.raises(ValueError, match="definition changed"):
        engine.run(ACTOR, "run", [Step("one", "effect", {"value": 2})])
    assert engine.run(OTHER_TENANT, "run", [STEPS[0]])["status"] == "blocked"


def test_retry_backoff_savepoint_and_dead_letter(tmp_path, policy, clock):
    now, timestamp = clock
    seen = []

    def fail(args, key, db):
        seen.append(key)
        db.execute("CREATE TABLE effects (value INTEGER)")
        db.execute("INSERT INTO effects VALUES (?)", (args["value"],))
        raise RuntimeError("private detail")

    engine = DurableEngine(tmp_path / "runs", policy, {"effect": Tool(fail)},
                           clock=timestamp, max_attempts=3, base_delay=2, max_delay=3)
    approval = approve(engine, policy)
    for attempts, delay in ((1, 2), (2, 3)):
        outcome = engine.resume(ACTOR, "run", approvals={"one": approval})
        assert (outcome["status"], outcome["attempts"], outcome["ready_at"]) == (
            "retry", attempts, now[0] + delay,
        )
        assert engine.queue(ACTOR) == []
        assert engine.resume(ACTOR, "run") == outcome
        now[0] += delay
        assert engine.queue(ACTOR) == ["run"]
    outcome = engine.resume(ACTOR, "run", approvals={"one": approval})
    assert outcome["status"] == "dead_letter"
    assert outcome["attempts"] == 3
    assert outcome["error_type"] == "RuntimeError"
    assert engine.resume(ACTOR, "run") == outcome
    assert engine.queue(ACTOR, dead_letters=True) == ["run"]
    assert engine.queue(OTHER_TENANT, dead_letters=True) == []
    assert len(seen) == 3 and len(set(seen)) == 1
    with engine._transaction() as db:
        assert db.execute("SELECT name FROM sqlite_master WHERE name='effects'").fetchone() is None


def test_concurrent_workers_commit_each_step_once(tmp_path, policy):
    barrier = Barrier(2)
    seen = []

    def invoke(args, key, db):
        seen.append(key)
        return args

    engine = DurableEngine(tmp_path / "runs", policy, {"read": Tool(invoke)})
    steps = [Step(str(n), "read", {"index": n}) for n in range(4)]

    def run():
        barrier.wait(timeout=5)
        return engine.run(ACTOR, "run", steps)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(run), pool.submit(run)
        assert first.result(timeout=10) == second.result(timeout=10)
    assert len(seen) == len(set(seen)) == len(steps)


def test_runtime_uses_tenant_tools_and_checks_approval_before_invoking(policy, monkeypatch):
    monkeypatch.setattr(core, "SHOW_THINKING", False)
    seen = []

    @tool
    def effect(value: int) -> str:
        """Apply a test effect."""
        seen.append(value)
        return "applied"

    provider = lambda call, arguments: policy.sign(call, REVIEWER)
    for principal, callback, expected in [
        (OTHER_TENANT, provider, "rbac_denied"),
        (ACTOR, None, "approval_required"),
        (ACTOR, provider, "applied"),
    ]:
        access = TenantToolAccess(principal, policy, {"effect": effect}, approval_provider=callback)
        runtime = agent_runtime.AgentRuntime("unused", tenant_access=access)
        message = AIMessage(content="", tool_calls=[
            {"name": "effect", "args": {"value": 7}, "id": "call-1"},
            {"name": "mcp_call_tool", "args": {}, "id": "call-2"},
        ])
        outputs = runtime.run_tool_calls({"messages": [message]})["messages"]
        assert expected in outputs[0].content
        assert "Unknown tool" in outputs[1].content
    assert seen == [7]


def test_checkpointed_transaction_and_retry_resume(tmp_path, policy, clock):
    now, timestamp = clock
    calls = []

    def effect(args, key, db):
        value = args["value"]
        calls.append(value)
        db.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, value INTEGER)")
        db.execute("INSERT INTO effects VALUES (?, ?)", (key, value))
        if value == 2 and calls.count(2) == 1:
            raise OSError
        return value

    engine = DurableEngine(tmp_path / "runs", policy, {"effect": Tool(effect)}, clock=timestamp)
    first_approval = approve(engine, policy, STEPS)
    pending = engine.resume(ACTOR, "run", approvals={"one": first_approval})
    assert (pending["status"], pending["results"]) == ("awaiting_approval", [1])
    call = pending["authorization"]
    second_approval = policy.sign(ToolCall(ACTOR, call["operation"], call["tool"], call["arguments_hash"]),
                                  REVIEWER)
    retry = engine.resume(ACTOR, "run", approvals={"two": second_approval})
    assert (retry["status"], retry["results"]) == ("retry", [1])
    now[0] = retry["ready_at"]
    result = engine.resume(ACTOR, "run", approvals={"two": second_approval})
    assert result["status"] == "completed"
    assert result["results"] == [1, 2]
    assert calls == [1, 2, 2]
    with engine._transaction() as db:
        assert [row[0] for row in db.execute("SELECT value FROM effects ORDER BY value")] == [1, 2]


def test_authorization_exports_correlated_opentelemetry_span(policy, monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from harness_agent import tenant_policy

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tenant_policy, "_TRACER", provider.get_tracer("test"))
    try:
        event = policy.authorize(ToolCall.bind(ACTOR, "op", "effect", {}))
        span, = exporter.get_finished_spans()
        assert span.name == "harness.tool.authorize"
        assert span.attributes["harness.decision"] == event["decision"] == "approval_required"
        assert span.attributes["harness.tenant"] == ACTOR.tenant
        assert event["trace_id"] == format(span.context.trace_id, "032x")
        assert event["span_id"] == format(span.context.span_id, "016x")
    finally:
        provider.shutdown()
