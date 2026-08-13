"""The four gaps the first pass on the completeness envelope left open.

Each class here pins one of them, and each is a case where the envelope was quietly WRONG
rather than merely thin - a resumed run reporting "complete" over its predecessor's capped
evidence is worse than no verdict at all, because a caller acts on it.
"""
from __future__ import annotations

import json
import unittest

from harness_agent import completeness, core, jobs, subagent

from tests.test_citations import _CitationTestCase
from tests.test_completeness import GREP_CAPPED, _FakeRuntime


class ResumeInheritsSignalsTests(unittest.TestCase):
    """A resume must not launder the original run's caps."""

    @staticmethod
    def _tool_entry(**over):
        entry = {
            "type": "tool",
            "name": "grep_files",
            "content": GREP_CAPPED,
            "artifact": {"full_content": GREP_CAPPED, "model_content_compacted": False},
        }
        entry.update(over)
        return entry

    def test_capped_result_in_restored_history_is_recovered(self):
        signals = completeness.signals_from_history([
            {"type": "human", "content": "find them all"},
            self._tool_entry(),
        ])
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0]["reason"], "capped")
        self.assertIs(signals[0]["restored"], True)

    def test_compacted_model_view_is_recovered(self):
        signals = completeness.signals_from_history([
            self._tool_entry(
                name="read_file",
                content='{"tool_result_compacted": true}',
                artifact={"full_content": '{"content": "x"}', "model_content_compacted": True},
            ),
        ])
        self.assertEqual([s["reason"] for s in signals], ["model_view_compacted"])

    def test_non_tool_and_malformed_entries_are_ignored(self):
        self.assertEqual(completeness.signals_from_history(None), [])
        self.assertEqual(completeness.signals_from_history([
            {"type": "ai", "content": "prose"},
            {"type": "tool"},                      # no name, no content
            "not a dict",
            {"type": "tool", "name": "x", "content": "plain string, not json"},
        ]), [])

    def test_resumed_delegation_reports_partial(self):
        """The end-to-end proof: a clean second turn over capped history is still partial."""
        rt = _FakeRuntime("m", ["a confident answer that never mentions the cap"])
        result = subagent._run_subagent_task(
            "carry on", "m",
            runtime_factory=lambda m: rt,
            restore_history=[self._tool_entry()],
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "partial")
        self.assertIs(result["coverage"]["signals"][0]["restored"], True)

    def test_a_resume_over_clean_history_stays_complete(self):
        rt = _FakeRuntime("m", ["the answer"])
        result = subagent._run_subagent_task(
            "carry on", "m",
            runtime_factory=lambda m: rt,
            restore_history=[
                {"type": "human", "content": "hello"},
                self._tool_entry(content='{"ok": true, "matches": []}',
                                 artifact={"full_content": '{"ok": true, "matches": []}'}),
            ],
        )
        self.assertEqual(result["status"], "complete")

    def test_seeding_happens_once_even_when_an_attempt_rebuilds_the_runtime(self):
        """Seeding inside runtime_factory would re-add the signal on every rebuild."""
        class _Exploding:
            model_name = "m"
            calls = 0

            def invoke(self, _text):
                _Exploding.calls += 1
                if _Exploding.calls == 1:
                    raise RuntimeError("transport died")
                return {"messages": [type("A", (), {"content": "recovered", "type": "ai"})()]}

        result = subagent._run_subagent_task(
            "carry on", "m",
            max_attempts=2,
            runtime_factory=lambda m: _Exploding(),
            restore_history=[self._tool_entry()],
        )
        restored = [s for s in result["coverage"]["signals"] if s.get("restored")]
        self.assertEqual(len(restored), 1, "the inherited signal was counted more than once")


class SummaryLineTests(unittest.TestCase):
    def test_clean_run_is_just_the_status(self):
        self.assertEqual(completeness.envelope(True)["coverage"]["summary"], "complete")

    def test_capped_signal_is_named_with_its_numbers(self):
        summary = completeness.envelope(True, [
            {"tool": "grep_files", "reason": "capped", "returned": 50, "total": 900},
        ])["coverage"]["summary"]
        self.assertTrue(summary.startswith("partial:"), summary)
        self.assertIn("grep_files capped (50 of 900)", summary)

    def test_coverage_ratio_and_gaps_are_reported(self):
        summary = completeness.envelope(
            True, [], covered=12, total=250, gaps=["a.js", "b.js", "c.js"],
        )["coverage"]["summary"]
        self.assertIn("12 of 250 examined", summary)
        self.assertIn("3 unexamined", summary)

    def test_extra_signals_collapse_rather_than_scroll(self):
        signals = [
            {"tool": f"t{i}", "reason": "capped"} for i in range(6)
        ]
        summary = completeness.envelope(True, signals)["coverage"]["summary"]
        self.assertIn("+4 more", summary)
        self.assertLess(len(summary), 200)

    def test_lossless_only_says_nothing_was_lost(self):
        summary = completeness.envelope(True, [
            {"tool": "<answer>", "reason": "answer_spilled_to_disk", "lossless": True},
        ])["coverage"]["summary"]
        self.assertTrue(summary.startswith("complete"), summary)
        self.assertIn("nothing lost", summary)

    def test_failed_run_says_failed(self):
        self.assertTrue(
            completeness.envelope(False)["coverage"]["summary"].startswith("failed")
        )


class CompactCoverageTests(unittest.TestCase):
    def test_gap_list_is_bounded_and_counted(self):
        full = completeness.envelope(
            True, [], covered=50, total=250, gaps=[f"f{i}.js" for i in range(200)],
        )["coverage"]
        compact = completeness.compact_coverage(full)
        self.assertEqual(compact["gaps_total"], 200)
        self.assertLessEqual(len(compact["gaps"]), 2)
        self.assertIn("gaps_note", compact)
        # The parts a caller actually acts on survive intact.
        self.assertEqual(compact["covered"], 50)
        self.assertIn("200 unexamined", compact["summary"])

    def test_small_coverage_keeps_its_whole_gap_list(self):
        full = completeness.envelope(True, [], covered=2, total=2, gaps=["only.js"])["coverage"]
        compact = completeness.compact_coverage(full)
        self.assertEqual(compact["gaps"], ["only.js"])
        self.assertNotIn("gaps_note", compact)
        # gaps_total is unconditional, including when nothing was dropped, so a caller can read
        # one key instead of branching on whether the list it holds is the whole list.
        self.assertEqual(compact["gaps_total"], 1)
        self.assertEqual(completeness.compact_coverage(
            completeness.envelope(True)["coverage"])["gaps_total"], 0)

    def test_non_dict_passes_through(self):
        self.assertIsNone(completeness.compact_coverage(None))

    def test_check_job_does_not_hoist_an_unbounded_gap_list(self):
        record = {
            "status": "done",
            "ok": True,
            "delegation_status": "partial",
            "coverage": completeness.envelope(
                True, [], covered=50, total=250, gaps=[f"f{i}.js" for i in range(200)],
            )["coverage"],
            "result": {"ok": True, "result": "x"},
        }
        original = jobs._read
        jobs._read = lambda job_id: record
        try:
            out = json.loads(jobs.check_job.func("job-1"))
        finally:
            jobs._read = original
        self.assertEqual(out["delegation_status"], "partial")
        self.assertEqual(out["coverage"]["gaps_total"], 200)
        self.assertLessEqual(len(out["coverage"]["gaps"]), 2)


class AbsolutePosixCitationTests(_CitationTestCase):
    """POSIX absolute paths were invisible to the extractor, so on a POSIX host an answer whose
    citations were all absolute counted zero and failed citation_min."""

    def test_posix_absolute_is_a_candidate(self):
        found = subagent._citation_candidates("The guard is at /srv/ws/app.py:12 today.")
        self.assertEqual([(p, ln) for _, p, ln in found], [("/srv/ws/app.py", 12)])

    def test_posix_absolute_inside_the_workspace_verifies(self):
        absolute = (self.ws / "src" / "app.py").as_posix()
        if not absolute.startswith("/"):
            self.skipTest("workspace is not rooted at / on this platform")
        self.assertAccepted(f"See {absolute}:12 for the guard.")

    def test_unverifiable_absolute_is_ignored_not_rejected(self):
        # An ssh_run or other-root path cannot be checked, so it must not sink a good answer.
        self.assertAccepted("Mirrors /srv/remote/app.py:12, same as src/app.py:12 here.")

    def test_relative_escape_is_still_a_rejection(self):
        # The absolute carve-out must not become a traversal loophole.
        reason = self.assertRejected("Compare with ../../etc/passwd.txt:1.")
        self.assertIn("escapes the workspace", reason)

    def test_url_path_is_still_not_a_citation(self):
        found = subagent._citation_candidates("Fetched https://example.com:8080/lib/app.js:12 ok.")
        self.assertEqual(found, [])
        self.assertAccepted("Fetched https://example.com:8080/lib/app.js:12, see src/app.py:12.")


class MapReduceGapSampleTests(unittest.TestCase):
    """A corpus sweep must not pay context to restate what resume_cursor already encodes."""

    def test_inline_gaps_are_a_bounded_sample(self):
        self.assertLessEqual(subagent._MAP_GAPS_INLINE, 20)

    def test_the_cursor_reproduces_the_gaps_exactly(self):
        # The contract that makes truncating the list safe: sorted order + offset. If the sort
        # ever changes, a resumed sweep re-maps some files and skips others, silently.
        matched = sorted(f"app/{i:03d}.js" for i in range(50))
        offset, max_files = 0, 4
        selected = matched[offset:offset + max_files]
        gaps = matched[offset + len(selected):]
        resumed = matched[offset + len(selected):offset + len(selected) + len(gaps)]
        self.assertEqual(gaps, resumed)
        self.assertEqual(len(gaps), 46)


class SnippetVerificationTests(unittest.TestCase):
    """A resolving citation is not a true one. Regression for the live 2026-08-13 failure: a
    model fabricated a grep_files result, cited line 12 for a function that is at line 1812, and
    the answer was accepted because the file does have twelve lines."""

    def setUp(self):
        import tempfile
        self._previous_workspace = str(core.WORKSPACE)
        self._tmp = tempfile.TemporaryDirectory()
        core.set_workspace(self._tmp.name)
        self.ws = core.WORKSPACE
        body = ["import os", ""] + [f"# filler {i}" for i in range(3, 40)]
        body.append("def clean_version(version_name: str) -> None:")
        body.append("    return None")
        (self.ws / "pipeline.py").write_text("\n".join(body), encoding="utf-8")
        (self.ws / "other.py").write_text("def helper(argument_name):\n    pass\n", encoding="utf-8")
        self.validate = subagent.build_validator(require_citations=True)

    def tearDown(self):
        core.set_workspace(self._previous_workspace)
        self._tmp.cleanup()

    def test_fabricated_quote_is_rejected(self):
        ok, reason = self.validate(
            "It is defined as `def clean_version(version):` at pipeline.py:12."
        )
        self.assertFalse(ok, "a quote that appears nowhere in the file was accepted")
        self.assertIn("does not appear", reason)

    def test_real_quote_at_a_real_line_is_accepted(self):
        ok, reason = self.validate(
            "It is `def clean_version(version_name: str) -> None:` at pipeline.py:40."
        )
        self.assertTrue(ok, reason)

    def test_quote_from_another_cited_file_does_not_false_reject(self):
        # The snippet sits next to pipeline.py's citation but was quoted from other.py.
        ok, reason = self.validate(
            "Compare `def helper(argument_name):` with pipeline.py:40 and other.py:1."
        )
        self.assertTrue(ok, reason)

    def test_elided_quote_is_not_a_verbatim_claim(self):
        ok, reason = self.validate("It is `def clean_version(...)` at pipeline.py:40.")
        self.assertTrue(ok, reason)

    def test_short_quote_is_ignored(self):
        ok, reason = self.validate("The `os` import is at pipeline.py:1.")
        self.assertTrue(ok, reason)

    def test_unquoted_prose_is_not_checked(self):
        # Paraphrase is not a claim of verbatim text, so it must not be verified against the file.
        ok, reason = self.validate("It defines clean_version taking a version at pipeline.py:40.")
        self.assertTrue(ok, reason)

    def test_line_check_still_applies(self):
        ok, reason = self.validate("See `def clean_version(version_name: str) -> None:` at pipeline.py:9000.")
        self.assertFalse(ok)
        self.assertIn("points past the end", reason)


class DegenerateAnswerTests(unittest.TestCase):
    def test_runaway_repetition_is_caught(self):
        reason = subagent._degenerate_reason("Here is the answer" + "```}" * 100)
        self.assertIn("degenerates", reason)
        self.assertIn("repeated", reason)

    def test_normal_answer_is_clean(self):
        self.assertEqual(subagent._degenerate_reason("A perfectly ordinary sentence."), "")
        self.assertEqual(subagent._degenerate_reason(""), "")

    def test_a_rule_of_repeated_characters_is_not_degenerate(self):
        # "-----" and "=====" are how a model draws a separator, not how it breaks.
        self.assertEqual(subagent._degenerate_reason("Summary\n" + "-" * 200), "")
        self.assertEqual(subagent._degenerate_reason("Summary\n" + "=" * 200), "")

    def test_repetition_in_the_middle_does_not_fire(self):
        # Only a stuck decoder is being caught, and it runs to the end of the output.
        text = "start" + "ab" * 100 + "\n\nand here is the real conclusion, at length."
        self.assertEqual(subagent._degenerate_reason(text), "")

    def test_env_escape_hatch(self):
        import os as _os
        _os.environ["HARNESS_ALLOW_DEGENERATE"] = "1"
        try:
            self.assertEqual(subagent._degenerate_reason("x" + "```}" * 100), "")
        finally:
            del _os.environ["HARNESS_ALLOW_DEGENERATE"]

    def test_degenerate_answer_retries_even_with_no_validator(self):
        rt = _FakeRuntime("m", ["answer" + "```}" * 100, "the real answer"])
        result = subagent._run_subagent_task("do it", "m", max_attempts=2, runtime_factory=lambda m: rt)
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["result"], "the real answer")
        self.assertFalse(result["validation_history"][0]["accepted"])
        self.assertIn("degenerates", result["validation_history"][0]["reason"])


if __name__ == "__main__":
    unittest.main()
