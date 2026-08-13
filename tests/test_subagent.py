import unittest

from langchain_core.messages import AIMessage, HumanMessage

from harness_agent import core, subagent


class FakeRuntime:
    """A stand-in for AgentRuntime that returns scripted answers, so the retry loop can be
    tested without hitting the LiteLLM proxy. ``answers`` is consumed one per invoke(); each
    invoke records the model it ran under and the turn input it received."""

    def __init__(self, model_name, answers):
        self.model_name = model_name
        self._answers = list(answers)
        self.calls = []  # (model_name, turn_input)

    def switch_model(self, model_name):
        self.model_name = model_name

    def invoke(self, user_text):
        self.calls.append((self.model_name, user_text))
        answer = self._answers.pop(0) if self._answers else ""
        return {"messages": [HumanMessage(content=user_text), AIMessage(content=answer)]}


class BuildValidatorTests(unittest.TestCase):
    def test_no_criteria_returns_none(self):
        self.assertIsNone(subagent.build_validator())

    def test_min_length_rejects_short(self):
        v = subagent.build_validator(min_length=10)
        ok, reason = v("short")
        self.assertFalse(ok)
        self.assertIn("too short", reason)
        self.assertTrue(v("this is definitely long enough")[0])

    def test_require_substrings_all_must_be_present(self):
        v = subagent.build_validator(require_substrings=["alpha", "beta"])
        self.assertFalse(v("only alpha here")[0])
        self.assertTrue(v("alpha and BETA (case-insensitive)")[0])

    def test_require_json_and_keys(self):
        v = subagent.build_validator(require_keys=["total", "items"])
        self.assertFalse(v("not json")[0])
        self.assertFalse(v('{"total": 1}')[0])  # missing "items"
        self.assertTrue(v('{"total": 1, "items": []}')[0])

    def test_require_json_unwraps_code_fence(self):
        v = subagent.build_validator(require_json=True)
        self.assertTrue(v('```json\n{"a": 1}\n```')[0])

    def test_require_regex(self):
        v = subagent.build_validator(require_regex=r"CVE-\d{4}-\d+")
        self.assertFalse(v("no id here")[0])
        self.assertTrue(v("found cve-2021-44228")[0])


class ModelSequenceTests(unittest.TestCase):
    def test_base_then_escalate_then_stick(self):
        seq = subagent._model_sequence("base", ["strong", "stronger"], attempts=4)
        self.assertEqual(seq, ["base", "strong", "stronger", "stronger"])

    def test_no_escalation_repeats_base(self):
        seq = subagent._model_sequence("base", [], attempts=3)
        self.assertEqual(seq, ["base", "base", "base"])

    def test_truncated_to_attempts(self):
        seq = subagent._model_sequence("base", ["a", "b", "c"], attempts=2)
        self.assertEqual(seq, ["base", "a"])


class RetryLoopTests(unittest.TestCase):
    def test_no_validator_single_attempt(self):
        rt = FakeRuntime("m", ["the answer"])
        result = subagent._run_subagent_task("do it", "m", runtime_factory=lambda m: rt)
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 1)
        self.assertEqual(result["result"], "the answer")
        self.assertEqual(len(rt.calls), 1)

    def test_retries_until_valid_and_feeds_back_reason(self):
        rt = FakeRuntime("m", ["too short", "still no", '{"total": 5}'])
        validator = subagent.build_validator(require_keys=["total"])
        result = subagent._run_subagent_task(
            "return json", "m", validator=validator, max_attempts=3,
            runtime_factory=lambda m: rt,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(len(rt.calls), 3)
        # First turn is the raw task; later turns carry the rejection feedback.
        self.assertEqual(rt.calls[0][1], "return json")
        self.assertIn("not accepted", rt.calls[1][1])

    def test_gives_up_after_max_attempts(self):
        rt = FakeRuntime("m", ["bad", "bad", "bad"])
        validator = subagent.build_validator(require_substrings=["GOOD"])
        result = subagent._run_subagent_task(
            "task", "m", validator=validator, max_attempts=3,
            runtime_factory=lambda m: rt,
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["result"], "bad")
        self.assertEqual(len(result["validation_history"]), 3)

    def test_escalates_model_across_attempts(self):
        rt = FakeRuntime("base", ["bad", "bad", "GOOD"])
        validator = subagent.build_validator(require_substrings=["GOOD"])
        result = subagent._run_subagent_task(
            "task", "base", validator=validator, max_attempts=3,
            escalation_models=["strong", "stronger"],
            runtime_factory=lambda m: rt,
        )
        self.assertTrue(result["ok"])
        self.assertEqual([c[0] for c in rt.calls], ["base", "strong", "stronger"])
        self.assertEqual(result["model"], "stronger")


class CitationRetryTests(unittest.TestCase):
    """The citation check has to compose with the existing retry loop like any other criterion:
    reject, feed the reason back, accept the corrected answer."""

    def setUp(self):
        import tempfile

        self._previous_workspace = str(core.WORKSPACE)
        self._tmp = tempfile.TemporaryDirectory()
        core.set_workspace(self._tmp.name)
        (core.WORKSPACE / "src").mkdir(parents=True, exist_ok=True)
        (core.WORKSPACE / "src" / "app.py").write_text("a\nb\nc\n" * 10, encoding="utf-8")

    def tearDown(self):
        core.set_workspace(self._previous_workspace)
        self._tmp.cleanup()

    def test_retries_until_the_answer_cites_a_real_file(self):
        rt = FakeRuntime("m", ["no citations here", "found it in src/app.py:12"])
        validator = subagent.build_validator(require_citations=True)
        result = subagent._run_subagent_task(
            "explain the guard", "m", validator=validator, max_attempts=2,
            runtime_factory=lambda m: rt,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["attempts"], 2)
        self.assertIn("not accepted", rt.calls[1][1])
        self.assertIn("verified file citation", rt.calls[1][1])


if __name__ == "__main__":
    unittest.main()
