"""Feature B: the citation validator.

Entirely offline - the check is deterministic filesystem verification, so every test runs
against a tempfile workspace and restores the real one in tearDown (core.set_workspace mutates
module globals for the whole process, and a leaked switch poisons every later test).

The bulk of this file is the false-positive suite. A citation check that trips on a URL port or
a timestamp is worse than no check at all: it would fail good answers and burn the retry budget
doing it. Every one of those cases is paired with a REAL citation and asserted ACCEPTED, which
is the only way to prove the offending token was ignored rather than merely tolerated.
"""
from __future__ import annotations

import tempfile
import unittest

from harness_agent import core, subagent


class _CitationTestCase(unittest.TestCase):
    def setUp(self):
        self._previous_workspace = str(core.WORKSPACE)
        self._tmp = tempfile.TemporaryDirectory()
        core.set_workspace(self._tmp.name)
        self.ws = core.WORKSPACE
        (self.ws / "src").mkdir(parents=True, exist_ok=True)
        (self.ws / "src" / "app.py").write_text(
            "\n".join(f"line {i}" for i in range(1, 31)), encoding="utf-8"
        )
        (self.ws / "README.md").write_text(
            "\n".join(f"readme {i}" for i in range(1, 6)), encoding="utf-8"
        )
        self.validate = subagent.build_validator(require_citations=True)

    def tearDown(self):
        core.set_workspace(self._previous_workspace)
        self._tmp.cleanup()

    def assertAccepted(self, text):
        ok, reason = self.validate(text)
        self.assertTrue(ok, f"expected accepted, got rejection: {reason}")

    def assertRejected(self, text):
        ok, reason = self.validate(text)
        self.assertFalse(ok, "expected rejection, answer was accepted")
        return reason


class ValidCitationTests(_CitationTestCase):
    def test_forward_slash_relative(self):
        self.assertAccepted("The bug is in src/app.py:12, right where the guard is missing.")

    def test_backslash_relative(self):
        self.assertAccepted("The bug is in src\\app.py:12.")

    def test_windows_absolute(self):
        absolute = str(self.ws / "src" / "app.py")
        self.assertAccepted(f"See {absolute}:12 for the guard.")

    def test_bare_filename_in_workspace_root(self):
        self.assertAccepted("Documented at README.md:3.")

    def test_line_range(self):
        self.assertAccepted("The loop spans src/app.py:10-14.")

    def test_trailing_sentence_period_does_not_break_the_match(self):
        # A model writes "see src/app.py:12." far more often than "see src/app.py:12" - if the
        # period defeated the regex the citation would be silently uncounted, not rejected,
        # which is the worst of both worlds.
        self.assertAccepted("The guard is missing; see src/app.py:12.")

    def test_dotted_filename(self):
        (self.ws / "src" / "config.local.py").write_text("a\nb\nc\n", encoding="utf-8")
        self.assertAccepted("Overridden in src/config.local.py:2.")


class BadCitationTests(_CitationTestCase):
    def test_hallucinated_path(self):
        reason = self.assertRejected("The handler lives in src/nope.py:3.")
        self.assertIn("src/nope.py:3", reason)
        self.assertIn("no such file", reason)

    def test_line_past_end_of_file(self):
        reason = self.assertRejected("Documented at README.md:900.")
        self.assertIn("README.md:900", reason)
        self.assertIn("5 lines", reason)

    def test_range_end_past_end_of_file(self):
        reason = self.assertRejected("It spans src/app.py:10-900.")
        self.assertIn("10-900", reason)

    def test_path_escaping_the_workspace(self):
        reason = self.assertRejected("Compare with ../../etc/passwd.txt:1.")
        self.assertIn("escapes the workspace", reason)

    def test_no_citations_at_all(self):
        reason = self.assertRejected("I looked at the code and it seems fine.")
        self.assertIn("0 verified file citation(s)", reason)

    def test_below_citation_min(self):
        validate = subagent.build_validator(require_citations=True, citation_min=2)
        ok, reason = validate("Only one reference here: src/app.py:12.")
        self.assertFalse(ok)
        self.assertIn("1 verified file citation(s)", reason)
        self.assertIn(">= 2", reason)

    def test_citation_min_counts_distinct_references(self):
        validate = subagent.build_validator(require_citations=True, citation_min=2)
        # The same file+line written two ways is ONE citation, not two.
        self.assertFalse(validate("src/app.py:12 and src\\app.py:12")[0])
        self.assertTrue(validate("src/app.py:12 and README.md:3")[0])

    def test_bad_citation_is_named_before_the_count_complaint(self):
        # citation_min is satisfied by the good one, so the only thing that can reject this is
        # the specific bad reference - and the feedback has to name it for the retry loop.
        reason = self.assertRejected("Real: src/app.py:12. Invented: src/ghost.py:4.")
        self.assertIn("src/ghost.py:4", reason)


class FalsePositiveTests(_CitationTestCase):
    """B4: things that LOOK like path:line must be ignored, never rejected.

    Each answer carries one real citation, so an accepted verdict proves the decoy was skipped
    at the shape pre-filter instead of being verified and failed.
    """

    def test_http_url_with_port(self):
        self.assertAccepted("The service on http://host:8080 is fine; see src/app.py:12.")

    def test_https_url_with_port_and_path(self):
        self.assertAccepted("Loaded from https://example.com:8080/x.js:12 - our copy is src/app.py:12.")

    def test_bare_host_with_port(self):
        self.assertAccepted("It dials example.com:8080 from src/app.py:12.")

    def test_clock_time(self):
        self.assertAccepted("The cron fires at 10:30 per src/app.py:12.")

    def test_timestamp(self):
        self.assertAccepted("Logged 2026-08-13 10:30:15 by src/app.py:12.")

    def test_note_prefix(self):
        self.assertAccepted("Note: this matters. See src/app.py:12.")

    def test_todo_prefix(self):
        self.assertAccepted("TODO: fix later, tracked at src/app.py:12.")

    def test_ratio(self):
        self.assertAccepted("The ratio is 3:2, computed in src/app.py:12.")

    def test_version_with_dotted_number(self):
        self.assertAccepted("Pinned to v1.2:3 in src/app.py:12.")

    def test_key_value_prose(self):
        self.assertAccepted("key: value pairs are parsed in src/app.py:12.")

    def test_mailto(self):
        self.assertAccepted("Owner mailto:someone@x.com, code at src/app.py:12.")

    def test_decoys_alone_do_not_satisfy_citation_min(self):
        # The mirror image of the tests above: ignoring a decoy must not mean COUNTING it.
        reason = self.assertRejected("See http://host:8080 and 10:30 and 3:2.")
        self.assertIn("0 verified file citation(s)", reason)

    def test_host_port_outside_the_known_tld_list(self):
        # The TLD allowlist only ever covered 13 suffixes, so every internal hostname - the ones
        # this operator's answers are full of - was verified as a file and then REJECTED. That is
        # not a miss, it is an active rejection of a correct answer that burns the retry ladder.
        self.assertAccepted("The API is on api.staging.corp:8443, handled in src/app.py:12.")
        self.assertAccepted("Redis lives at cache.svc.internal:6379; see src/app.py:12.")
        self.assertAccepted("Metrics on collector.telemetry.cloud:9090 per src/app.py:12.")

    def test_dotted_identifiers_and_properties(self):
        self.assertAccepted("org.slf4j.Logger:5 is the logger used in src/app.py:12.")
        self.assertAccepted("It sets margin.top:12 in the stylesheet, built by src/app.py:12.")

    def test_a_decoy_does_not_veto_answers_that_also_cite_properly(self):
        # The validator returns on the FIRST bad candidate, so one unresolvable token used to sink
        # an answer no matter how many real citations sat beside it.
        self.assertAccepted(
            "Config at conf.d.local:80 - the real code is src/app.py:12 and README.md:3."
        )


class BareFilenameTests(_CitationTestCase):
    """A model that has already named the directory goes on citing basenames, and those are
    references to files it genuinely read. core.resolve_path joins onto the workspace ROOT, so
    every one of them used to be rejected with "no such file" - feedback that is simply untrue
    and gives the retry loop nothing correct to act on."""

    def test_basename_of_a_file_in_a_subdirectory_resolves(self):
        self.assertAccepted("The guard is missing at app.py:12.")

    def test_basename_line_number_is_still_verified(self):
        reason = self.assertRejected("The guard is at app.py:900.")
        self.assertIn("app.py:900", reason)
        self.assertIn("30 lines", reason)

    def test_an_unresolvable_bare_name_is_ignored_not_rejected(self):
        # Indistinguishable from a hostname or a dotted identifier, so it may not reject - but it
        # must not COUNT either, which is what keeps citation_min doing the real work.
        reason = self.assertRejected("It is handled in ghost.py:4.")
        self.assertIn("0 verified file citation(s)", reason)

    def test_a_path_with_directories_is_still_a_hard_rejection(self):
        # A model writing directory segments is claiming a specific location; a miss there is a
        # fabricated citation, which is the case the feature exists to catch.
        reason = self.assertRejected("It is handled in src/ghost.py:4.")
        self.assertIn("no such file", reason)

    def test_the_same_basename_in_two_places_accepts_if_either_fits(self):
        (self.ws / "vendor").mkdir()
        (self.ws / "vendor" / "app.py").write_text("only one line", encoding="utf-8")
        self.assertAccepted("See app.py:25 for the loop.")


class CandidateExtractionTests(_CitationTestCase):
    """Direct tests of the pre-filter, so a regex change that silently stops matching (rather
    than starting to reject) is still caught."""

    def test_extracts_path_and_line(self):
        found = subagent._citation_candidates("look at src/app.py:12 please")
        self.assertEqual(found, [("src/app.py:12", "src/app.py", 12)])

    def test_range_reports_the_end_line(self):
        found = subagent._citation_candidates("src/app.py:10-14")
        self.assertEqual(found[0][2], 14)

    def test_url_yields_no_candidate(self):
        self.assertEqual(subagent._citation_candidates("https://example.com:8080/x.js:12"), [])

    def test_zero_line_number_is_ignored(self):
        self.assertEqual(subagent._citation_candidates("src/app.py:0"), [])

    def test_does_not_match_mid_filename(self):
        # If the engine were allowed to start after a dot it would "verify" local.py while the
        # model actually wrote config.local.py - a different file entirely.
        found = subagent._citation_candidates("config.local.py:2")
        self.assertEqual(found[0][1], "config.local.py")


class GuardTests(_CitationTestCase):
    def test_require_citations_alone_builds_a_validator(self):
        # The regression this guards: if require_citations is missing from build_validator's
        # "no criteria" guard, asking for citations alone returns None and the check never runs.
        self.assertIsNotNone(subagent.build_validator(require_citations=True))

    def test_no_criteria_still_returns_none(self):
        self.assertIsNone(subagent.build_validator())

    def test_unreadable_file_fails_open(self):
        # A local IO problem must not discard an otherwise-good answer (same posture as a judge
        # outage). Simulated by making the size check itself blow up.
        original = subagent._citation_line_count
        subagent._citation_line_count = lambda rp: -1
        try:
            self.assertAccepted("src/app.py:99999 is where it happens.")
        finally:
            subagent._citation_line_count = original

    def test_criteria_chain_still_short_circuits_before_citations(self):
        validate = subagent.build_validator(min_length=500, require_citations=True)
        ok, reason = validate("src/app.py:12")
        self.assertFalse(ok)
        self.assertIn("too short", reason)


class ValidationKwargsTests(_CitationTestCase):
    def test_citation_criteria_reach_the_validator(self):
        kwargs = subagent._validation_kwargs(
            0, "", "", False, "", "", "task", 0, "",
            require_citations=True, citation_min=1,
        )
        self.assertIn("validator", kwargs)
        self.assertFalse(kwargs["validator"]("no citations here")[0])
        self.assertTrue(kwargs["validator"]("see src/app.py:12")[0])

    def test_no_criteria_still_returns_empty(self):
        self.assertEqual(subagent._validation_kwargs(0, "", "", False, "", "", "task", 0, ""), {})


if __name__ == "__main__":
    unittest.main()
