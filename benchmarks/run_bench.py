"""Run from the repository root with python -m benchmarks.run_bench."""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from time import perf_counter_ns

from harness_agent.durable_engine import DurableEngine, Step, Tool
from harness_agent.tenant_policy import Principal, TenantPolicy, ToolCall

ACTOR = Principal("benchmark", "worker")
HUMAN = Principal("benchmark", "reviewer")
STEPS = [Step(str(index), "effect", {"index": index}) for index in range(3)]
CRASH_EXIT = 73
SAMPLES = 200


def policy():
    return TenantPolicy(
        memberships={("benchmark", "worker"): ["operator"], ("benchmark", "reviewer"): ["approver"]},
        grants={("benchmark", "operator"): ["effect"]}, signing_key=b"benchmark-only-key".ljust(32, b"!"),
    )


def replay_engine(root, crash_step=None, after_effect=False, calls=None):
    def invoke(arguments, key, _transaction):
        index = arguments["index"]
        if calls is not None:
            calls.append(index)
        if index == crash_step and not after_effect:
            os._exit(CRASH_EXIT)
        with closing(sqlite3.connect(root / "provider.db")) as provider, provider:
            provider.execute("INSERT INTO effects VALUES (?, ?) ON CONFLICT (key) DO NOTHING", (key, index))
            result = provider.execute("SELECT step FROM effects WHERE key=?", (key,)).fetchone()[0]
        if index == crash_step and after_effect:
            os._exit(CRASH_EXIT)
        return result

    return DurableEngine(root / "runs.db", policy(), {"effect": Tool(invoke)})


def crash_worker(root, crash_step, after_effect):
    replay_engine(root, crash_step, after_effect).run(ACTOR, "replay", STEPS)


def verify_replay(root):
    recoveries = duplicates = 0
    for index in range(len(STEPS)):
        for after_effect in (False, True):
            case = root / f"crash-{index}-{int(after_effect)}"
            case.mkdir()
            with closing(sqlite3.connect(case / "provider.db")) as provider:
                provider.execute("CREATE TABLE effects (key TEXT PRIMARY KEY, step INTEGER NOT NULL)")
            worker = multiprocessing.get_context("spawn").Process(
                target=crash_worker, args=(case, index, after_effect)
            )
            worker.start()
            worker.join(20)
            if worker.is_alive():
                worker.kill()
                worker.join()
            if worker.exitcode != CRASH_EXIT:
                raise RuntimeError(f"expected injected process exit, got {worker.exitcode}")
            calls = []
            result = replay_engine(case, calls=calls).resume(ACTOR, "replay")
            with closing(sqlite3.connect(case / "provider.db")) as provider:
                effects = provider.execute("SELECT step FROM effects ORDER BY step").fetchall()
            duplicates += len(effects) - len(set(effects))
            if result["status"] != "completed" or result["results"] != list(range(len(STEPS))):
                raise RuntimeError(f"bad recovery: {result}")
            if calls != list(range(index, len(STEPS))) or effects != [(i,) for i in range(len(STEPS))]:
                raise RuntimeError(f"bad replay: {calls=}, {effects=}")
            recoveries += 1
    return recoveries, duplicates


def percentiles(samples):
    ordered = sorted(samples)
    return {
        f"p{percentile}_ms": round(ordered[math.ceil(len(ordered) * percentile / 100) - 1] / 1e6, 4)
        for percentile in (50, 95)
    }


def measure(root):
    starts = []
    checkpoint = []

    def empty_step(_arguments, _key, _transaction):
        starts.append(perf_counter_ns())
        return None

    access = policy()
    engine = DurableEngine(root / "latency.db", access, {"effect": Tool(empty_step)})
    for index in range(SAMPLES):
        result = engine.run(ACTOR, f"sample-{index}", [Step("step", "effect", {})])
        checkpoint.append(perf_counter_ns() - starts.pop())
        if result["status"] != "completed":
            raise RuntimeError(f"checkpoint benchmark failed: {result}")

    gate = []
    for index in range(SAMPLES):
        call = ToolCall.bind(ACTOR, str(index), "effect", {"index": index})
        start = perf_counter_ns()
        approval = access.sign(call, HUMAN)
        result = access.authorize(call, approval, high_impact=True)
        gate.append(perf_counter_ns() - start)
        if not result["ok"]:
            raise RuntimeError(f"gate benchmark failed: {result}")
    return percentiles(checkpoint), percentiles(gate)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        recoveries, duplicates = verify_replay(root)
        checkpoint, gate = measure(root)
    print(json.dumps({
        "crash_recoveries": recoveries, "duplicate_side_effects": duplicates,
        "samples_per_latency": SAMPLES, "checkpoint": checkpoint, "hitl_gate": gate,
    }, indent=2))


if __name__ == "__main__":
    main()
