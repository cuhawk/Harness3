"""Delegation outcome ledger.

Three things have to hold for this feature to be worth having:
  - the bucket key (task_shape) is STABLE, or the table is a pile of one-run buckets;
  - the ledger reads Feature A's status rather than deriving a second opinion;
  - a broken ledger is invisible to the delegation it observes - bookkeeping that can break the
    thing it measures is worse than no bookkeeping.
"""
import json
import os
import sqlite3
import tempfile
import unittest

from langchain_core.messages import AIMessage, HumanMessage

from harness_agent import completeness, core, ledger, mcp_server, subagent, tools_registry


class FakeRuntime:
    """Scripted stand-in for AgentRuntime (see tests/test_subagent.py) - the retry loop under
    test must never reach the LiteLLM proxy. ``recorder`` is optional so the no-recorder case
    stays exercised too."""

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


class LedgerTestCase(unittest.TestCase):
    """Every ledger DB lands in a throwaway workspace, and the suite-wide kill switch is lifted
    only here - a leaked workspace or a leaked env var poisons every later test in the process."""

    def setUp(self):
        self._original_workspace = str(core.WORKSPACE)
        self._original_disable = os.environ.get(ledger.DISABLE_ENV)
        os.environ.pop(ledger.DISABLE_ENV, None)
        self._tmp = tempfile.TemporaryDirectory()
        core.set_workspace(self._tmp.name)

    def tearDown(self):
        core.set_workspace(self._original_workspace)
        if self._original_disable is None:
            os.environ.pop(ledger.DISABLE_ENV, None)
        else:
            os.environ[ledger.DISABLE_ENV] = self._original_disable
        self._tmp.cleanup()


class SchemaTests(LedgerTestCase):
    def test_connect_is_idempotent_and_creates_everything(self):
        for _ in range(2):
            conn = ledger._connect()
            try:
                names = {
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
                    )
                }
            finally:
                conn.close()
        self.assertIn("delegations", names)
        self.assertIn("delegation_attempts", names)
        self.assertIn("idx_deleg_shape_model", names)
        self.assertIn("idx_deleg_ts", names)
        self.assertIn("idx_attempts_parent", names)

    def test_db_path_follows_the_workspace(self):
        # Not a module constant: set_workspace rebinds CACHE_DIR, and a captured path would keep
        # writing into whatever workspace the process started in.
        self.assertEqual(ledger._db_path().parent, core.CACHE_DIR)


class TaskShapeTests(LedgerTestCase):
    def test_same_task_yields_the_same_shape(self):
        task = "Summarize the release notes in docs and report the highlights."
        self.assertEqual(ledger.task_shape(task), ledger.task_shape(task))

    def test_paraphrases_with_the_same_verb_share_a_bucket(self):
        a = ledger.task_shape("Summarize what this module does.")
        b = ledger.task_shape("Summarize the behaviour of this module.")
        self.assertEqual(a, b)
        self.assertTrue(a.startswith("summarize|"))

    def test_unknown_leading_verb_collapses_to_other(self):
        # A closed vocabulary is the point: an open one makes one bucket per phrasing.
        self.assertTrue(ledger.task_shape("Frobnicate the widget registry.").startswith("other|"))

    def test_synonyms_map_onto_one_canonical_verb(self):
        self.assertTrue(ledger.task_shape("Grep for TODO markers.").startswith("search|"))
        self.assertTrue(ledger.task_shape("Find the TODO markers.").startswith("search|"))

    def test_glob_flag_changes_only_the_last_segment(self):
        task = "Read this file and report."
        plain = ledger.task_shape(task, glob_involved=False)
        globbed = ledger.task_shape(task, glob_involved=True)
        self.assertTrue(plain.endswith("|noglob"))
        self.assertTrue(globbed.endswith("|glob"))
        self.assertEqual(plain.rsplit("|", 1)[0], globbed.rsplit("|", 1)[0])

    def test_glob_in_the_task_text_is_detected(self):
        self.assertTrue(ledger.task_shape("Read every **/*.py file and report.").endswith("|glob"))
        # A lone asterisk in prose is not a glob.
        self.assertTrue(ledger.task_shape("Read the file and note any * marks.").endswith("|noglob"))

    def test_shape_has_three_segments_and_a_group_or_core(self):
        shape = ledger.task_shape("Read the file.")
        verb, groups, glob = shape.split("|")
        self.assertIn(verb, ledger._VERBS)
        self.assertTrue(groups)
        self.assertIn(glob, ("glob", "noglob"))

    def test_groups_come_from_the_real_selector(self):
        # Not a second scorer: a shape must describe the groups the run actually bound.
        task = "Open the website https://example.com in the browser and screenshot the page."
        groups = tools_registry.select_group_names(task)
        self.assertIn("browser", groups)
        self.assertEqual(ledger.task_shape(task).split("|")[1], "+".join(groups))


class RecordAndStatsTests(LedgerTestCase):
    def _record(self, task, model, ok, attempts, latency, status="complete", reason=""):
        ledger.record_delegation(
            task=task, model=model, ok=ok, accepted=ok, attempts=attempts,
            latency_ms=latency, rejection_reason=reason, status=status,
            validator_kinds=["min_length"], entry_point="spawn_subagent",
            validation_history=[{"attempt": 1, "model": model, "ok": True, "accepted": ok, "reason": reason or "ok"}],
        )

    def setUp(self):
        super().setUp()
        search = "Search the tree for hardcoded secrets."
        summarize = "Summarize the module."
        # 2 shapes x 2 models, mixed outcomes: alpha succeeds on the search shape, beta does not.
        self._record(search, "alpha", True, 1, 100)
        self._record(search, "alpha", True, 2, 300)
        self._record(search, "alpha", False, 3, 500, status="failed", reason="empty answer")
        self._record(search, "beta", False, 3, 900, status="failed", reason="empty answer")
        self._record(summarize, "alpha", True, 1, 50)
        self._record(summarize, "beta", True, 1, 70, status="partial")

    def _stats(self, **kwargs):
        payload = json.loads(ledger.delegation_stats.invoke(kwargs))
        self.assertTrue(payload["ok"], payload)
        return payload

    def test_rows_and_attempts_are_both_stored(self):
        conn = ledger._connect()
        try:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM delegations").fetchone()[0], 6)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM delegation_attempts").fetchone()[0], 6)
            row = conn.execute("SELECT * FROM delegations ORDER BY id LIMIT 1").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["validator_kinds"], "min_length")
        self.assertEqual(row["entry_point"], "spawn_subagent")
        self.assertEqual(row["workspace"], str(core.WORKSPACE))

    def test_rates_attempts_and_median_latency(self):
        payload = self._stats()
        self.assertEqual(payload["runs_total"], 6)
        by_key = {(s["task_shape"], s["model"]): s for s in payload["stats"]}
        search_alpha = next(v for (shape, model), v in by_key.items() if model == "alpha" and shape.startswith("search|"))
        self.assertEqual(search_alpha["runs"], 3)
        self.assertEqual(search_alpha["ok_rate"], round(2 / 3, 3))
        self.assertEqual(search_alpha["attempt_distribution"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(search_alpha["median_latency_ms"], 300)
        self.assertEqual(search_alpha["status_counts"], {"complete": 2, "failed": 1})
        self.assertEqual(search_alpha["top_rejection_reasons"][0]["reason"], "empty answer")

    def test_model_filter_narrows(self):
        payload = self._stats(model="beta")
        self.assertEqual(payload["runs_total"], 2)
        self.assertTrue(all(s["model"] == "beta" for s in payload["stats"]))

    def test_shape_filter_narrows(self):
        payload = self._stats(task_shape="summarize|")
        self.assertEqual(payload["runs_total"], 2)
        self.assertTrue(all(s["task_shape"].startswith("summarize|") for s in payload["stats"]))

    def test_min_runs_hides_small_buckets(self):
        payload = self._stats(min_runs=3)
        self.assertEqual([s["runs"] for s in payload["stats"]], [3])

    def test_capped_listing_reports_the_true_total(self):
        payload = self._stats(max_rows=1)
        self.assertEqual(payload["returned"], 1)
        self.assertGreater(payload["matches_total"], 1)
        self.assertTrue(payload["capped"])
        self.assertIn("LISTING TRUNCATED", payload["note"])

    def test_empty_ledger_is_not_an_error(self):
        conn = ledger._connect()
        try:
            conn.execute("DELETE FROM delegations")
            conn.commit()
        finally:
            conn.close()
        payload = self._stats()
        self.assertEqual(payload["runs_total"], 0)
        self.assertEqual(payload["stats"], [])


class BestEffortTests(LedgerTestCase):
    """C5: a broken ledger must leave the delegation result untouched."""

    def _run(self):
        return subagent._run_subagent_task(
            "Summarize this module.",
            "fake-model",
            runtime_factory=lambda m: FakeRuntime(m, ["a perfectly good answer"]),
        )

    def test_result_is_byte_identical_when_the_ledger_cannot_open(self):
        healthy = json.dumps(self._run(), sort_keys=True, default=str)

        def explode():
            raise sqlite3.OperationalError("unable to open database file")

        original = ledger._connect
        ledger._connect = explode
        try:
            broken = json.dumps(self._run(), sort_keys=True, default=str)
        finally:
            ledger._connect = original
        self.assertEqual(healthy, broken)

    def test_bad_argument_types_do_not_escape(self):
        # Nothing about a delegation should depend on the ledger being called correctly.
        ledger.record_delegation(
            task=None, model=object(), ok="yes", accepted=None, attempts="three",
            latency_ms=None, validator_kinds="not-a-list", validation_history="not-a-list",
        )

    def test_kill_switch_writes_nothing(self):
        os.environ[ledger.DISABLE_ENV] = "1"
        try:
            self._run()
        finally:
            os.environ.pop(ledger.DISABLE_ENV, None)
        self.assertFalse(ledger._db_path().exists())


class StatusFromFeatureATests(LedgerTestCase):
    """The ledger must READ the completeness verdict, never re-derive one."""

    def test_a_capped_run_is_recorded_partial(self):
        recorder = completeness.CompletenessRecorder()
        recorder.record("grep_files", json.dumps({
            "ok": True, "matches_total": 900, "returned": 100, "capped": True,
        }))
        result = subagent._run_subagent_task(
            "Search the tree for hardcoded secrets.",
            "fake-model",
            runtime_factory=lambda m: FakeRuntime(m, ["found some"], recorder=recorder),
        )
        self.assertEqual(result["status"], completeness.STATUS_PARTIAL)

        conn = ledger._connect()
        try:
            row = conn.execute("SELECT status, model, attempts, ok FROM delegations").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["status"], result["status"])
        self.assertEqual(row["model"], "fake-model")
        self.assertEqual(row["attempts"], 1)
        self.assertEqual(row["ok"], 1)

    def test_a_failed_run_is_recorded_with_its_attempts(self):
        result = subagent._run_subagent_task(
            "Summarize this module.",
            "fake-model",
            validator=lambda text: (False, "not good enough"),
            validator_kinds=["min_length"],
            max_attempts=2,
            runtime_factory=lambda m: FakeRuntime(m, ["short", "still short"]),
        )
        self.assertFalse(result["ok"])
        conn = ledger._connect()
        try:
            row = conn.execute("SELECT * FROM delegations").fetchone()
            attempts = conn.execute(
                "SELECT attempt, accepted FROM delegation_attempts ORDER BY attempt"
            ).fetchall()
        finally:
            conn.close()
        self.assertEqual(row["status"], completeness.STATUS_FAILED)
        self.assertEqual(row["ok"], 0)
        self.assertEqual(row["accepted"], 0)
        self.assertEqual(row["attempts"], 2)
        self.assertIn("not good enough", row["rejection_reason"])
        self.assertEqual(row["validator_kinds"], "min_length")
        self.assertEqual([a["attempt"] for a in attempts], [1, 2])
        self.assertTrue(all(a["accepted"] == 0 for a in attempts))

    def test_validator_kinds_reach_the_row_through_validation_kwargs(self):
        # The whole point of putting validator_kinds in _validation_kwargs' return: it is
        # splatted into _run_subagent_task at every entry point, so no call site is edited.
        kwargs = subagent._validation_kwargs(
            10, "", "", False, "", "", "Summarize this module.", 0, "",
        )
        self.assertEqual(kwargs["validator_kinds"], ["min_length"])
        subagent._run_subagent_task(
            "Summarize this module.",
            "fake-model",
            runtime_factory=lambda m: FakeRuntime(m, ["a sufficiently long answer here"]),
            **kwargs,
        )
        conn = ledger._connect()
        try:
            row = conn.execute("SELECT validator_kinds, entry_point FROM delegations").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["validator_kinds"], "min_length")
        self.assertEqual(row["entry_point"], "")  # direct call, no tool wrapper

    def test_no_criteria_records_an_empty_validator_kinds(self):
        self.assertEqual(subagent._validation_kwargs(0, "", "", False, "", "", "t", 0, ""), {})
        subagent._run_subagent_task(
            "Summarize this module.", "fake-model",
            runtime_factory=lambda m: FakeRuntime(m, ["answer"]),
        )
        conn = ledger._connect()
        try:
            row = conn.execute("SELECT validator_kinds FROM delegations").fetchone()
        finally:
            conn.close()
        self.assertEqual(row["validator_kinds"], "")


class RegistrationTests(unittest.TestCase):
    """A tool registered but absent from the preset is invisible to the MCP host, and a preset
    name absent from the registry is silently dropped. Both halves have to be there."""

    def test_tool_is_in_the_registry(self):
        self.assertIn("delegation_stats", tools_registry.TOOL_BY_NAME)
        self.assertIn(ledger.delegation_stats, tools_registry.DELEGATION_TOOLS)

    def test_tool_is_in_the_delegate_preset(self):
        self.assertIn("delegation_stats", mcp_server.PRESETS["delegate"])

    def test_preset_names_all_resolve(self):
        for name in mcp_server.PRESETS["delegate"]:
            self.assertIn(name, tools_registry.TOOL_BY_NAME, f"{name} would be silently dropped")

    def test_not_added_to_delegate_lite(self):
        self.assertNotIn("delegation_stats", mcp_server.PRESETS["delegate-lite"])


if __name__ == "__main__":
    unittest.main()
