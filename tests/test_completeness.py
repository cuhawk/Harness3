"""Feature A: the completeness envelope.

These tests are deliberately offline - no router, no MCP server, no AgentRuntime. Everything
that touches the filesystem runs against a tempfile workspace and restores the real one in
tearDown, because core.set_workspace mutates module globals for the whole process and a leaked
switch would poison every test that runs after it.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from harness_agent import completeness, core, subagent


class _FakeRuntime:
    """Scripted stand-in for AgentRuntime. Deliberately does NOT define ``completeness`` unless
    one is passed in, so the getattr-default path in _drain_signals stays covered."""

    def __init__(self, model_name, answers, recorder=None):
        self.model_name = model_name
        self._answers = list(answers)
        self.calls = []
        if recorder is not None:
            self.completeness = recorder

    def switch_model(self, model_name):
        self.model_name = model_name

    def invoke(self, user_text):
        self.calls.append((self.model_name, user_text))
        answer = self._answers.pop(0) if self._answers else ""
        return {"messages": [HumanMessage(content=user_text), AIMessage(content=answer)]}


GREP_CAPPED = json.dumps({
    "ok": True,
    "query": "password",
    "matches_total": 912,
    "returned": 100,
    "capped": True,
    "files_scanned": 44,
    "files_skipped_too_large": 0,
    "results": [],
})


class DetectorTests(unittest.TestCase):
    def test_grep_files_capped(self):
        signals = completeness.detect_cap_signals("grep_files", GREP_CAPPED)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["reason"], "capped")
        self.assertEqual(signals[0]["returned"], 100)
        self.assertEqual(signals[0]["total"], 912)
        self.assertEqual(signals[0]["tool"], "grep_files")

    def test_grep_files_uncapped_is_silent(self):
        payload = json.loads(GREP_CAPPED)
        payload.update({"capped": False, "returned": 912})
        self.assertEqual(completeness.detect_cap_signals("grep_files", json.dumps(payload)), [])

    def test_list_tree_truncated(self):
        signals = completeness.detect_cap_signals(
            "list_tree", json.dumps({"ok": True, "count": 500, "truncated": True, "entries": []})
        )
        self.assertEqual([s["reason"] for s in signals], ["truncated"])

    def test_files_skipped_counter(self):
        signals = completeness.detect_cap_signals(
            "extract_findings",
            json.dumps({"ok": True, "files_scanned": 10, "files_skipped_over_cap": 3}),
        )
        self.assertEqual([s["reason"] for s in signals], ["files_skipped_over_cap"])
        self.assertEqual(signals[0]["total"], 3)

    def test_zero_skipped_is_not_a_signal(self):
        self.assertEqual(
            completeness.detect_cap_signals(
                "extract_findings", json.dumps({"ok": True, "files_skipped_over_cap": 0})
            ),
            [],
        )

    def test_plain_string_tool_is_ignored(self):
        self.assertEqual(completeness.detect_cap_signals("run_terminal", "hello world"), [])
        self.assertEqual(completeness.detect_cap_signals("run_terminal", None), [])

    def test_output_trimmed_at_256kb_is_its_own_signal(self):
        # core.to_json trims an oversized tool result into INVALID json with this suffix.
        raw = GREP_CAPPED[:200] + "\n...[truncated]..."
        signals = completeness.detect_cap_signals("grep_files", raw)
        self.assertIn("tool_output_trimmed_256kb", [s["reason"] for s in signals])

    def test_trimmed_but_still_parseable_reports_both(self):
        # A dict small enough that chopping only the closing brace leaves a parseable body is
        # unrealistic, so instead prove the strip-and-retry path: valid JSON + the marker.
        raw = GREP_CAPPED + "\n...[truncated]..."
        reasons = [s["reason"] for s in completeness.detect_cap_signals("grep_files", raw)]
        self.assertEqual(sorted(reasons), ["capped", "tool_output_trimmed_256kb"])

    def test_error_result_is_not_a_cap(self):
        self.assertEqual(
            completeness.detect_cap_signals("grep_files", json.dumps({"ok": False, "error": "boom"})),
            [],
        )

    def test_nested_result_scope(self):
        # jobs.check_job nests the delegation payload one level down.
        signals = completeness.detect_cap_signals(
            "check_job",
            json.dumps({"ok": True, "status": "done", "result": {"truncated": True, "total_chars": 9000}}),
        )
        self.assertEqual([s["reason"] for s in signals], ["truncated"])


class NestedDelegationTests(unittest.TestCase):
    """The envelope has to COMPOSE. MAX_SUBAGENT_DEPTH is 2 and sub-agents can delegate, so a
    sub-delegation is one of the tool results most likely to be carrying a cap - and the harness's
    own envelope is the one shape a key-driven detector would otherwise walk straight past."""

    def test_partial_sub_delegation_is_a_signal(self):
        payload = json.dumps({
            "ok": True, "model": "m", "result": "prose",
            **completeness.envelope(True, [{"tool": "grep_files", "reason": "capped"}]),
        })
        signals = completeness.detect_cap_signals("spawn_subagent", payload)
        self.assertEqual([s["reason"] for s in signals], ["nested_run_partial"])
        self.assertEqual(
            completeness.compute_status(True, signals), completeness.STATUS_PARTIAL
        )

    def test_partial_check_job_poll_is_a_signal(self):
        # delegation_status is the hoisted copy; this record's own "status" means done/running.
        payload = json.dumps({
            "ok": True, "status": "done",
            "delegation_status": completeness.STATUS_PARTIAL,
            "coverage": {"capped": True, "signals": [], "covered": 20, "total": 400},
        })
        signals = completeness.detect_cap_signals("check_job", payload)
        self.assertEqual([s["reason"] for s in signals], ["nested_run_partial"])
        self.assertEqual(signals[0]["returned"], 20)
        self.assertEqual(signals[0]["total"], 400)

    def test_clean_sub_delegation_is_silent(self):
        payload = json.dumps({"ok": True, "result": "prose", **completeness.envelope(True, [])})
        self.assertEqual(completeness.detect_cap_signals("spawn_subagent", payload), [])

    def test_a_spilled_sub_delegation_does_not_infect_the_parent(self):
        # The lossy/lossless decision was already made one level down; reading the inner status
        # rather than re-deriving from the inner signals is what makes that hold.
        inner = completeness.envelope(True, [{
            "tool": "<answer>", "reason": "answer_spilled_to_disk", "lossless": True,
        }])
        payload = json.dumps({"ok": True, "result_truncated": True, "result": "head", **inner})
        self.assertEqual(completeness.detect_cap_signals("spawn_subagent", payload), [])

    def test_a_failed_sub_delegation_is_not_a_cap(self):
        # Same rule as everywhere else here: an error propagates on ok, not as a coverage gap.
        payload = json.dumps({"ok": False, "error": "gave up", **completeness.envelope(False, [])})
        self.assertEqual(completeness.detect_cap_signals("spawn_subagent", payload), [])


class ModelViewTruncationTests(unittest.TestCase):
    """The harness's OWN 12 KB model budget is the largest and commonest truncation in the system,
    and a tool that reports no cap of its own (read_file, list_tree, run_terminal) used to hand the
    model a prefix and the caller a clean verdict."""

    def _big_read_file(self, size=200_000):
        return json.dumps({
            "ok": True, "path": "big.log", "offset": 0, "bytes_read": size,
            "file_size_bytes": size, "content": "x" * size,
        })

    def test_compaction_is_a_signal_even_when_the_tool_reports_nothing(self):
        from harness_agent import agent_runtime

        raw = self._big_read_file()
        self.assertEqual(completeness.detect_cap_signals("read_file", raw), [])  # tool is silent
        model_content = agent_runtime.compact_tool_result_for_model(raw)
        signals = completeness.detect_model_view_signals("read_file", raw, model_content)
        self.assertEqual([s["reason"] for s in signals], ["model_view_truncated"])
        self.assertLess(signals[0]["returned"], signals[0]["total"])
        self.assertEqual(
            completeness.compute_status(True, signals), completeness.STATUS_PARTIAL
        )

    def test_a_result_under_the_budget_is_silent(self):
        from harness_agent import agent_runtime

        raw = json.dumps({"ok": True, "content": "small"})
        model_content = agent_runtime.compact_tool_result_for_model(raw)
        self.assertEqual(completeness.detect_model_view_signals("read_file", raw, model_content), [])


class _AllowAllGate:
    def request(self, tool_name, args):
        return True, "allowed for test"


class _StubTool:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, args):
        return self.payload


class ToolNodeWiringTests(unittest.TestCase):
    """The ONE production site that feeds the recorder, driven for real.

    Everything else in this file exercises the recorder or the detector directly, so the graph
    node itself was unobserved: deleting its record() call left the whole suite green while
    turning the feature into a no-op that reports an authoritative-looking clean verdict on every
    delegation - strictly worse than not having it, because the operator would stop re-checking.
    """

    def setUp(self):
        from harness_agent import agent_runtime, hooks, permissions

        self.agent_runtime = agent_runtime
        self._saved = (agent_runtime.TOOL_BY_NAME, permissions.GATE, hooks.run_hooks, core.SHOW_THINKING)
        self.permissions, self.hooks = permissions, hooks
        permissions.GATE = _AllowAllGate()
        hooks.run_hooks = lambda *a, **k: {}
        core.SHOW_THINKING = False

    def tearDown(self):
        tools, gate, run_hooks, show = self._saved
        self.agent_runtime.TOOL_BY_NAME = tools
        self.permissions.GATE = gate
        self.hooks.run_hooks = run_hooks
        core.SHOW_THINKING = show

    def _drive(self, tool_name, payload):
        self.agent_runtime.TOOL_BY_NAME = {tool_name: _StubTool(payload)}
        runtime = self.agent_runtime.AgentRuntime("stub-model")
        message = AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {}, "id": "call-1"}],
        )
        out = runtime.run_tool_calls({"messages": [message]})
        return runtime, out["messages"][0]

    def test_a_capped_tool_result_reaches_the_recorder(self):
        runtime, _ = self._drive("grep_files", GREP_CAPPED)
        signals = runtime.completeness.drain()
        self.assertEqual([s["reason"] for s in signals], ["capped"])
        self.assertEqual(signals[0]["total"], 912)

    def test_the_model_budget_truncation_is_recorded_too(self):
        big = json.dumps({"ok": True, "path": "big.log", "offset": 0, "content": "x" * 200_000})
        runtime, tool_message = self._drive("read_file", big)
        reasons = [s["reason"] for s in runtime.completeness.drain()]
        self.assertEqual(reasons, ["model_view_truncated"])
        # And the model really did only see a prefix - the signal is not cosmetic.
        self.assertLess(len(tool_message.content), len(big))

    def test_a_clean_tool_result_records_nothing(self):
        runtime, _ = self._drive("grep_files", json.dumps({"ok": True, "capped": False, "results": []}))
        self.assertEqual(runtime.completeness.drain(), [])

    def test_a_blocked_call_is_not_a_coverage_gap(self):
        # A synthesised block/error envelope is this node's own text, not tool evidence.
        self.permissions.GATE = type("Deny", (), {"request": lambda self, n, a: (False, "no")})()
        runtime, _ = self._drive("grep_files", GREP_CAPPED)
        self.assertEqual(runtime.completeness.drain(), [])


class InvariantTests(unittest.TestCase):
    """A3: no code path may report a complete run when a cap signal was recorded."""

    def test_signal_forces_partial(self):
        self.assertEqual(
            completeness.compute_status(True, [{"tool": "grep_files", "reason": "capped"}]),
            "partial",
        )

    def test_clean_run_is_complete(self):
        self.assertEqual(completeness.compute_status(True, []), "complete")

    def test_not_ok_is_failed_even_with_no_signals(self):
        self.assertEqual(completeness.compute_status(False, []), "failed")
        self.assertEqual(
            completeness.compute_status(False, [{"reason": "capped"}]), "failed"
        )

    def test_envelope_shape(self):
        env = completeness.envelope(True, [{"tool": "grep_files", "reason": "capped"}])
        self.assertEqual(env["status"], "partial")
        self.assertIs(env["coverage"]["capped"], True)
        self.assertEqual(env["coverage"]["gaps"], [])
        self.assertIsNone(env["coverage"]["covered"])

    def test_envelope_bounds_signal_flood(self):
        env = completeness.envelope(True, [{"reason": "capped"}] * 50)
        self.assertEqual(len(env["coverage"]["signals"]), 21)
        self.assertEqual(env["coverage"]["signals"][-1]["reason"], "signals_truncated")
        self.assertEqual(env["coverage"]["signals"][-1]["total"], 50)

    def test_status_literal_lives_only_in_completeness_module(self):
        """Source-level guard: the only module allowed to write the string is the one that
        derives it. This is what stops a later feature from hand-stamping a clean status onto
        a new code path and quietly re-opening the laundering hole."""
        pkg = Path(completeness.__file__).parent
        offenders = []
        for path in sorted(pkg.glob("*.py")):
            if path.name == "completeness.py":
                continue
            if '"complete"' in path.read_text(encoding="utf-8", errors="replace"):
                offenders.append(path.name)
        self.assertEqual(offenders, [], f"status literal written outside completeness.py: {offenders}")


class RecorderDrainTests(unittest.TestCase):
    def test_recorded_signal_makes_the_delegation_partial(self):
        recorder = completeness.CompletenessRecorder()
        recorder.record("grep_files", GREP_CAPPED)
        rt = _FakeRuntime("m", ["a confident prose answer with no mention of the cap"], recorder)
        result = subagent._run_subagent_task("count them", "m", runtime_factory=lambda m: rt)
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "partial")
        self.assertIs(result["coverage"]["capped"], True)
        self.assertEqual(result["coverage"]["signals"][0]["reason"], "capped")

    def test_runtime_without_recorder_still_works(self):
        rt = _FakeRuntime("m", ["the answer"])
        result = subagent._run_subagent_task("do it", "m", runtime_factory=lambda m: rt)
        self.assertEqual(result["status"], "complete")
        self.assertIs(result["coverage"]["capped"], False)

    def test_drain_clears_so_a_reused_runtime_does_not_double_count(self):
        recorder = completeness.CompletenessRecorder()
        recorder.record("grep_files", GREP_CAPPED)
        self.assertEqual(len(recorder.drain()), 1)
        self.assertEqual(recorder.drain(), [])

    def test_signals_from_a_rejected_attempt_survive(self):
        recorder = completeness.CompletenessRecorder()
        recorder.record("grep_files", GREP_CAPPED)
        rt = _FakeRuntime("m", ["bad", "GOOD"], recorder)
        result = subagent._run_subagent_task(
            "task", "m",
            validator=subagent.build_validator(require_substrings=["GOOD"]),
            max_attempts=2,
            runtime_factory=lambda m: rt,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)
        # The cap happened during attempt 1, which was REJECTED. It must still be reported.
        self.assertEqual(result["status"], "partial")

    def test_failed_run_is_failed_not_partial(self):
        rt = _FakeRuntime("m", ["bad", "bad", "bad"])
        result = subagent._run_subagent_task(
            "task", "m",
            validator=subagent.build_validator(require_substrings=["GOOD"]),
            max_attempts=3,
            runtime_factory=lambda m: rt,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["status"], "failed")


class _WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        self._previous_workspace = str(core.WORKSPACE)
        self._tmp = tempfile.TemporaryDirectory()
        core.set_workspace(self._tmp.name)
        self.ws = core.WORKSPACE

    def tearDown(self):
        core.set_workspace(self._previous_workspace)
        self._tmp.cleanup()


class AnswerTruncationTests(_WorkspaceTestCase):
    """A4: losing bytes off the ANSWER is a truncation too, and belongs in the same story."""

    def test_spilled_answer_is_disclosed_but_not_partial(self):
        """A spill keeps every byte on disk, so it is REPORTED without downgrading the run.

        The alternative was tried and is worse: spawn_map_reduce spills at 1500 chars by default,
        so counting a spill as a cap made "partial" the verdict of every verbose-but-complete
        sweep - and a status that cries wolf sends the operator back to hand-verifying everything.
        """
        rt = _FakeRuntime("m", ["x" * 500])
        result = subagent._run_subagent_task(
            "do it", "m", 20000, spill_over_chars=50, runtime_factory=lambda m: rt
        )
        self.assertTrue(result["result_truncated"])
        self.assertEqual(result["status"], completeness.STATUS_COMPLETE)
        self.assertIs(result["coverage"]["capped"], False)
        # Disclosed, not silent: the caller is still told the inline text is only a head.
        signals = result["coverage"]["signals"]
        self.assertEqual([s["reason"] for s in signals], ["answer_spilled_to_disk"])
        self.assertIs(signals[0]["lossless"], True)
        self.assertTrue(Path(result["result_path"]).is_file())

    def test_trimmed_answer_is_partial(self):
        # No spilling, but max_history_chars chops the tail - previously flagged nowhere at all.
        rt = _FakeRuntime("m", ["y" * 500])
        result = subagent._run_subagent_task(
            "do it", "m", 50, spill_over_chars=0, runtime_factory=lambda m: rt
        )
        self.assertEqual(result["status"], "partial")
        self.assertEqual([s["reason"] for s in result["coverage"]["signals"]], ["answer_trimmed"])

    def test_short_answer_is_complete(self):
        rt = _FakeRuntime("m", ["short enough"])
        result = subagent._run_subagent_task(
            "do it", "m", 20000, spill_over_chars=6000, runtime_factory=lambda m: rt
        )
        self.assertEqual(result["status"], "complete")


class AllowlistTests(unittest.TestCase):
    """A6: the fields have to reach the MCP caller, through BOTH payload-shaping paths."""

    def test_compact_result_keeps_status_and_coverage(self):
        env = completeness.envelope(True, [{"tool": "grep_files", "reason": "capped"}])
        compact = subagent._compact_result({"ok": True, "model": "m", "result": "r", **env})
        self.assertIn("status", compact)
        self.assertIn("coverage", compact)
        self.assertEqual(compact["status"], "partial")

    def test_single_delegation_json_keeps_status_and_coverage(self):
        rt = _FakeRuntime("m", ["the answer"])
        result = subagent._run_subagent_task("do it", "m", runtime_factory=lambda m: rt)
        round_tripped = json.loads(core.to_json(result))
        self.assertEqual(round_tripped["status"], "complete")
        self.assertIn("coverage", round_tripped)

    def test_every_compact_key_is_declared(self):
        self.assertIn("status", subagent._COMPACT_KEYS)
        self.assertIn("coverage", subagent._COMPACT_KEYS)


class MapReduceCoverageTests(_WorkspaceTestCase):
    """A5: covered/total/gaps come from the harness's own glob expansion, and an unfinished
    sweep hands back a cursor that resumes over exactly the gaps."""

    def setUp(self):
        super().setUp()
        for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
            (self.ws / name).write_text("content", encoding="utf-8")
        self.seen: list[str] = []
        self._real_runner = subagent._run_subagent_task

        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            self.seen.append(task.rsplit("The file in question is: ", 1)[-1].rstrip(")"))
            return {
                "ok": True, "model": model_name or "stub", "attempts": 1, "result": "ok",
                **completeness.envelope(True, []),
            }

        subagent._run_subagent_task = stub

    def tearDown(self):
        subagent._run_subagent_task = self._real_runner
        super().tearDown()

    def _run(self, **kwargs):
        args = {"glob_pattern": "*.txt", "map_prompt": "look at {path}", "max_files": 2}
        args.update(kwargs)
        return json.loads(subagent.spawn_map_reduce.invoke(args))

    def test_first_batch_reports_gaps_and_a_cursor(self):
        out = self._run()
        self.assertEqual(out["coverage"]["covered"], 2)
        self.assertEqual(out["coverage"]["total"], 5)
        self.assertEqual(len(out["coverage"]["gaps"]), 3)
        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["resume_cursor"]["offset"], 2)
        self.assertEqual(self.seen, ["a.txt", "b.txt"])

    def test_resume_processes_exactly_the_next_gap(self):
        first = self._run()
        self.seen.clear()
        second = self._run(offset=first["resume_cursor"]["offset"])
        self.assertEqual(self.seen, ["c.txt", "d.txt"])
        self.assertEqual(second["coverage"]["covered"], 2)
        self.assertEqual(second["coverage"]["gaps"], ["e.txt"])
        self.assertEqual(second["resume_cursor"]["offset"], 4)

    def test_full_sweep_is_complete(self):
        out = self._run(max_files=10)
        self.assertEqual(out["status"], "complete")
        self.assertEqual(out["coverage"]["covered"], 5)
        self.assertEqual(out["coverage"]["gaps"], [])
        self.assertNotIn("resume_cursor", out)

    def test_partial_sub_run_makes_the_sweep_partial(self):
        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            return {
                "ok": True, "model": "stub", "attempts": 1, "result": "ok",
                **completeness.envelope(True, [{"tool": "grep_files", "reason": "capped"}]),
            }

        subagent._run_subagent_task = stub
        out = self._run(max_files=10)
        self.assertEqual(out["status"], "partial")
        self.assertIn("sub_run_partial", [s["reason"] for s in out["coverage"]["signals"]])
        # And the per-file entries kept their own verdict through _compact_result.
        self.assertEqual(out["results"][0]["status"], "partial")

    def test_offset_past_the_end_is_refused_not_crashed(self):
        out = self._run(offset=99)
        self.assertFalse(out["ok"])
        self.assertIn("already complete", out["error"])

    def test_failed_sub_runs_make_the_sweep_partial(self):
        # A failed per-file run has status "failed", so it contributes no partial signal of its
        # own - a sweep where every sub-agent exhausted its attempts used to report clean.
        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            return {
                "ok": False, "model": "stub", "attempts": 3, "error": "gave up",
                **completeness.envelope(False, []),
            }

        subagent._run_subagent_task = stub
        out = self._run(max_files=10)
        self.assertEqual(out["map_failed"], 5)
        self.assertEqual(out["status"], completeness.STATUS_PARTIAL)
        self.assertIn("sub_run_failed", [s["reason"] for s in out["coverage"]["signals"]])

    def test_a_partial_reduce_makes_the_sweep_partial(self):
        # The reduce answer is the headline result, and the envelope used to be built before the
        # reduce stage had even run.
        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            if kwargs.get("entry_point") == "map_reduce_reduce":
                return {
                    "ok": True, "model": "stub", "attempts": 1, "result": "synthesis",
                    **completeness.envelope(True, [{"tool": "grep_files", "reason": "capped"}]),
                }
            return {
                "ok": True, "model": "stub", "attempts": 1, "result": "ok",
                **completeness.envelope(True, []),
            }

        subagent._run_subagent_task = stub
        out = self._run(max_files=10, reduce_prompt="fold them together")
        self.assertEqual(out["reduced"]["status"], completeness.STATUS_PARTIAL)
        self.assertEqual(out["status"], completeness.STATUS_PARTIAL)
        self.assertIn("capped", [s["reason"] for s in out["coverage"]["signals"]])

    def test_a_failed_reduce_makes_the_sweep_partial(self):
        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            ok = kwargs.get("entry_point") != "map_reduce_reduce"
            return {
                "ok": ok, "model": "stub", "attempts": 1, "result": "ok",
                **completeness.envelope(ok, []),
            }

        subagent._run_subagent_task = stub
        out = self._run(max_files=10, reduce_prompt="fold them together")
        self.assertIn("reduce_failed", [s["reason"] for s in out["coverage"]["signals"]])
        self.assertEqual(out["status"], completeness.STATUS_PARTIAL)

    def test_a_clean_full_sweep_with_a_reduce_stays_complete(self):
        out = self._run(max_files=10, reduce_prompt="fold them together")
        self.assertEqual(out["status"], completeness.STATUS_COMPLETE)


class MapReduceTrimSurvivalTests(_WorkspaceTestCase):
    """core.to_json trims the whole reply at MAX_OUTPUT_BYTES, so key ORDER decides whether the
    verdict survives. Appending the envelope after the big "results" list deleted status,
    coverage and resume_cursor at the tool's own documented defaults - i.e. exactly on the large
    sweeps that skipped files, and only on those."""

    def setUp(self):
        super().setUp()
        for i in range(300):
            (self.ws / f"f{i:03d}.txt").write_text("x", encoding="utf-8")
        self._real_runner = subagent._run_subagent_task

        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            return {
                "ok": True, "model": "stub", "attempts": 1, "result": "y" * 1400,
                **completeness.envelope(True, []),
            }

        subagent._run_subagent_task = stub

    def tearDown(self):
        subagent._run_subagent_task = self._real_runner
        super().tearDown()

    def test_the_verdict_survives_a_reply_that_blows_the_output_cap(self):
        raw = subagent.spawn_map_reduce.invoke({
            "glob_pattern": "*.txt", "map_prompt": "look at {path}", "max_files": 200,
        })
        self.assertGreater(len(raw), 0)
        self.assertIn("...[truncated]...", raw)  # the reply really is over the cap
        head = raw[: raw.index("...[truncated]...")]
        for key in ('"status"', '"coverage"', '"resume_cursor"', '"gaps_total"'):
            self.assertIn(key, head, f"{key} fell past the 256 KB trim")
        self.assertLess(head.index('"status"'), head.index('"results"'))


class ParallelFanOutCoverageTests(_WorkspaceTestCase):
    """A batch must not read clean when one of its tasks ran on capped evidence."""

    def setUp(self):
        super().setUp()
        self._real_runner = subagent._run_subagent_task

    def tearDown(self):
        subagent._run_subagent_task = self._real_runner
        super().tearDown()

    def _run_with(self, statuses):
        queue = list(statuses)

        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            signals, ok = queue.pop(0)
            return {
                "ok": ok, "model": "stub", "attempts": 1, "result": "answer",
                **completeness.envelope(ok, signals),
            }

        subagent._run_subagent_task = stub
        return json.loads(subagent.spawn_subagents_parallel.invoke(
            {"tasks_json": json.dumps(["one", "two"])}
        ))

    def test_clean_batch_is_complete(self):
        out = self._run_with([([], True), ([], True)])
        self.assertEqual(out["status"], "complete")

    def test_one_capped_task_makes_the_batch_partial(self):
        out = self._run_with([([], True), ([{"tool": "grep_files", "reason": "capped"}], True)])
        self.assertEqual(out["status"], "partial")
        self.assertIn("capped", [s["reason"] for s in out["coverage"]["signals"]])
        self.assertEqual(out["results"][1]["status"], "partial")

    def test_a_failed_task_is_a_coverage_gap(self):
        out = self._run_with([([], True), ([], False)])
        self.assertEqual(out["status"], "partial")
        self.assertIn("task_failed", [s["reason"] for s in out["coverage"]["signals"]])


class _StubRunnerTestCase(_WorkspaceTestCase):
    """Swaps out the single-task runner so an entry point can be driven with scripted verdicts."""

    def setUp(self):
        super().setUp()
        self._real_runner = subagent._run_subagent_task

    def tearDown(self):
        subagent._run_subagent_task = self._real_runner
        super().tearDown()

    def _install(self, signals_for):
        def stub(task, model_name="", max_history_chars=20000, **kwargs):
            signals, ok = signals_for(task, kwargs)
            return {
                "ok": ok, "model": "stub", "attempts": 1,
                "result": '{"verdict": "false_positive"}',
                **completeness.envelope(ok, signals),
            }

        subagent._run_subagent_task = stub


class WorkflowEnvelopeTests(_StubRunnerTestCase):
    """workflow_run's whole selling point is that intermediate results stay in the harness - which
    is exactly why the caller cannot re-derive completeness itself and needs a top-level verdict."""

    def tearDown(self):
        from harness_agent import workflows

        workflows.set_runtime_provider(None)
        super().tearDown()

    def test_a_capped_middle_step_makes_the_pipeline_partial(self):
        from harness_agent import workflows

        self._install(lambda task, kwargs: (
            [{"tool": "grep_files", "reason": "capped"}] if "second" in task else [], True
        ))
        outcome = workflows._run_steps({"steps": [
            {"id": "a", "mode": "agent", "prompt": "first"},
            {"id": "b", "mode": "agent", "prompt": "second"},
        ]}, {})
        self.assertTrue(outcome["ok"])
        self.assertEqual(outcome["status"], completeness.STATUS_PARTIAL)
        self.assertIn("capped", [s["reason"] for s in outcome["coverage"]["signals"]])

    def test_a_clean_pipeline_is_complete(self):
        from harness_agent import workflows

        self._install(lambda task, kwargs: ([], True))
        outcome = workflows._run_steps(
            {"steps": [{"id": "a", "mode": "agent", "prompt": "only"}]}, {}
        )
        self.assertEqual(outcome["status"], completeness.STATUS_COMPLETE)

    def test_single_template_workflow_carries_a_verdict_and_drains_the_runtime(self):
        # This path runs the shared live runtime directly instead of _run_subagent_task, so it
        # used to return no status at all - and never drained the recorder, so one run's caps
        # leaked into whatever the operator did next.
        from harness_agent import workflows

        recorder = completeness.CompletenessRecorder()

        class _LiveRuntime:
            model_name = "live-model"

            def __init__(self):
                self.completeness = recorder

            def invoke(self, text):
                self.completeness.record("grep_files", GREP_CAPPED)
                return {"messages": [AIMessage(content="the answer")]}

        workflows.set_runtime_provider(_LiveRuntime)
        workflows.workflow_save.invoke({"name": "probe", "prompt_template": "count {thing}"})
        out = json.loads(workflows.workflow_run.invoke(
            {"name": "probe", "variables_json": json.dumps({"thing": "files"})}
        ))
        self.assertEqual(out["status"], completeness.STATUS_PARTIAL)
        self.assertEqual([s["reason"] for s in out["coverage"]["signals"]], ["capped"])
        self.assertEqual(recorder.drain(), [], "the recorder must not keep growing per run")


class VerifyFindingEnvelopeTests(_StubRunnerTestCase):
    """verify_finding is the delegated output a report gets written from, and its three inner runs
    already compute a verdict that used to be discarded five lines later."""

    def test_a_capped_reviewer_makes_the_triage_partial(self):
        from harness_agent import verify_tools

        self._install(lambda task, kwargs: (
            [{"tool": "grep_files", "reason": "capped"}]
            if kwargs.get("entry_point") == "verify_finding_red" else [],
            True,
        ))
        out = json.loads(verify_tools.verify_finding.invoke({"claim": "reflected param"}))
        self.assertEqual(out["verdict"], "false_positive")
        self.assertEqual(out["status"], completeness.STATUS_PARTIAL)
        self.assertIn("capped", [s["reason"] for s in out["coverage"]["signals"]])

    def test_a_failed_reviewer_is_a_coverage_gap(self):
        from harness_agent import verify_tools

        self._install(lambda task, kwargs: (
            [], kwargs.get("entry_point") != "verify_finding_blue"
        ))
        out = json.loads(verify_tools.verify_finding.invoke({"claim": "reflected param"}))
        self.assertIn("reviewer_run_failed", [s["reason"] for s in out["coverage"]["signals"]])

    def test_a_clean_review_is_complete(self):
        from harness_agent import verify_tools

        self._install(lambda task, kwargs: ([], True))
        out = json.loads(verify_tools.verify_finding.invoke({"claim": "reflected param"}))
        self.assertEqual(out["status"], completeness.STATUS_COMPLETE)


class AsyncUnparseableReplyTests(_WorkspaceTestCase):
    """A delegation reply too big for core.to_json comes back as invalid JSON. Treating that as
    {ok: true} with a null verdict restored the laundering end to end on the async path."""

    def test_a_trimmed_reply_is_recorded_partial_not_clean(self):
        from types import SimpleNamespace

        from harness_agent import jobs
        from harness_agent import subagent as subagent_module

        real = subagent_module.spawn_map_reduce
        subagent_module.spawn_map_reduce = SimpleNamespace(
            invoke=lambda kwargs: '{"ok": true, "results": [\n...[truncated]...'
        )
        record = {"job_id": "trimjob", "model": "m"}
        try:
            # On a thread, as production does: _run_job calls permissions.mark_background(), and
            # that ContextVar would otherwise stay set on the test runner's own thread and change
            # how every later permission test resolves an ask-policy tool.
            worker = threading.Thread(
                target=jobs._run_job, args=("trimjob", record, "map_reduce", {})
            )
            worker.start()
            worker.join(timeout=30)
        finally:
            subagent_module.spawn_map_reduce = real

        self.assertEqual(record["delegation_status"], completeness.STATUS_PARTIAL)
        self.assertIs(record["coverage"]["capped"], True)
        out = json.loads(jobs.check_job.invoke({"job_id": "trimjob"}))
        self.assertEqual(out["delegation_status"], completeness.STATUS_PARTIAL)


class AsyncJobHoistTests(_WorkspaceTestCase):
    """check_job replaces a big result with a 2000-char head stub, so the coverage verdict has
    to live ABOVE that stub or the async path silently loses it."""

    def test_delegation_status_survives_the_head_stub(self):
        from harness_agent import jobs

        payload = {
            "ok": True,
            "result": "x" * 5000,  # comfortably past _HEAD_CHARS
            **completeness.envelope(True, [{"tool": "grep_files", "reason": "capped"}]),
        }
        record = {
            "job_id": "testjob1", "kind": "single", "label": "l", "model": "m",
            "status": "done", "ok": True, "result": payload,
            "delegation_status": payload["status"], "coverage": payload["coverage"],
        }
        jobs._write("testjob1", record)
        out = json.loads(jobs.check_job.invoke({"job_id": "testjob1"}))
        self.assertTrue(out["result"]["truncated"])  # the result itself is stubbed
        self.assertEqual(out["delegation_status"], "partial")
        self.assertIs(out["coverage"]["capped"], True)
        # The job record's own status keeps its original running/done/error meaning.
        self.assertEqual(out["status"], "done")


if __name__ == "__main__":
    unittest.main()
