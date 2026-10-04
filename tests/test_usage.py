"""The per-reply token/speed footer must report the SERVER's numbers, and must not lie
about speed when a turn spent most of its time in tools.

Two failure modes this pins down, both of which look fine on screen:
  1. A streamed OpenAI-compatible response carries no usage block unless the request asks
     for one, so the footer silently degrades to a chars/4 estimate. stream_usage=True on
     the client is the only thing preventing that, and nothing else in the suite would
     notice if it were dropped.
  2. tok/s over WALL time counts tool execution as generation time. A turn with two model
     calls around a slow tool would report a fraction of the model's real decode speed.
"""
import time
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk

from harness_agent import agent_runtime, core, usage


def _msg(usage_metadata=None, response_metadata=None, content="hello"):
    return AIMessage(
        content=content,
        usage_metadata=usage_metadata,
        response_metadata=response_metadata or {},
    )


class UsageExtractionTests(unittest.TestCase):
    def test_server_reported_usage_is_used_verbatim(self):
        counts, estimated = usage.usage_from_message(_msg({
            "input_tokens": 1234,
            "output_tokens": 56,
            "total_tokens": 1290,
            "input_token_details": {"cache_read": 900},
            "output_token_details": {"reasoning": 12},
        }))
        self.assertFalse(estimated)
        self.assertEqual(counts["input"], 1234)
        self.assertEqual(counts["output"], 56)
        self.assertEqual(counts["total"], 1290)
        self.assertEqual(counts["cached_input"], 900)
        self.assertEqual(counts["reasoning"], 12)

    def test_falls_back_to_the_raw_openai_shape(self):
        # What a non-streaming call (and some proxies) leave in response_metadata instead.
        counts, estimated = usage.usage_from_message(_msg(None, {"token_usage": {
            "prompt_tokens": 9,
            "completion_tokens": 7,
            "total_tokens": 16,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }}))
        self.assertFalse(estimated)
        self.assertEqual((counts["input"], counts["output"], counts["total"]), (9, 7, 16))
        self.assertEqual(counts["reasoning"], 3)

    def test_missing_usage_estimates_and_flags_it(self):
        counts, estimated = usage.usage_from_message(_msg(None, None, content="x" * 40))
        self.assertTrue(estimated)
        self.assertEqual(counts["output"], 10)
        # Input is left at zero rather than guessed: the prompt is the whole conversation
        # plus every tool schema, and a chars/4 guess of that is not worth showing.
        self.assertEqual(counts["input"], 0)

    def test_a_dead_call_does_not_crash_the_accounting(self):
        counts, estimated = usage.usage_from_message(None)
        self.assertTrue(estimated)
        self.assertEqual(counts["output"], 0)

    def test_estimated_numbers_are_marked_in_the_footer(self):
        turn = usage.Usage(turns=1)
        turn.record_call(_msg(None, None, content="abcd"), started=0.0, first_chunk_at=0.0, ended=1.0)
        line = turn.format_line()
        self.assertIn("~", line)
        self.assertIn("estimated", line)

        reported = usage.Usage(turns=1)
        reported.record_call(
            _msg({"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}),
            started=0.0, first_chunk_at=0.0, ended=1.0,
        )
        self.assertNotIn("~", reported.format_line())


class SpeedAccountingTests(unittest.TestCase):
    def _call(self, turn, out_tokens, started, first_chunk_at, ended):
        turn.record_call(
            _msg({"input_tokens": 100, "output_tokens": out_tokens, "total_tokens": 100 + out_tokens}),
            started=started, first_chunk_at=first_chunk_at, ended=ended,
        )

    def test_tok_s_measures_decode_not_wall_clock(self):
        turn = usage.Usage(turns=1)
        self._call(turn, 100, started=0.0, first_chunk_at=1.0, ended=3.0)  # 2s of decode
        turn.wall_seconds = 60.0  # the other 57s were a slow tool
        self.assertAlmostEqual(turn.tokens_per_second, 50.0)
        self.assertAlmostEqual(turn.ttft_seconds, 1.0)

    def test_a_multi_hop_turn_sums_every_model_call(self):
        turn = usage.Usage(turns=1)
        self._call(turn, 20, started=0.0, first_chunk_at=0.5, ended=1.0)
        self._call(turn, 80, started=10.0, first_chunk_at=10.5, ended=12.0)
        self.assertEqual(turn.calls, 2)
        self.assertEqual(turn.output_tokens, 100)
        self.assertEqual(turn.input_tokens, 200)
        self.assertAlmostEqual(turn.decode_seconds, 2.0)
        self.assertAlmostEqual(turn.tokens_per_second, 50.0)
        # ttft stays the FIRST call's, not the last one's.
        self.assertAlmostEqual(turn.ttft_seconds, 0.5)

    def test_a_call_that_produced_nothing_stays_out_of_the_denominator(self):
        turn = usage.Usage(turns=1)
        self._call(turn, 100, started=0.0, first_chunk_at=0.0, ended=2.0)
        turn.record_call(None, started=2.0, first_chunk_at=None, ended=8.0)
        self.assertAlmostEqual(turn.decode_seconds, 2.0)
        self.assertAlmostEqual(turn.tokens_per_second, 50.0)
        self.assertAlmostEqual(turn.llm_seconds, 8.0)

    def test_session_totals_accumulate_turns(self):
        session = usage.Usage()
        for _ in range(3):
            turn = usage.Usage(turns=1)
            self._call(turn, 10, started=0.0, first_chunk_at=0.0, ended=1.0)
            turn.wall_seconds = 5.0
            session.merge(turn)
        self.assertEqual(session.turns, 3)
        self.assertEqual(session.output_tokens, 30)
        self.assertAlmostEqual(session.wall_seconds, 15.0)
        self.assertIn("3 turns", session.format_totals())


class _FakeLLM:
    """Stands in for the bound ChatOpenAI. Yields chunks with a real delay so the timing
    split (ttft vs decode) is measured, not asserted into existence."""

    def __init__(self, chunks, first_chunk_delay=0.05, per_chunk_delay=0.01):
        self.chunks = chunks
        self.first_chunk_delay = first_chunk_delay
        self.per_chunk_delay = per_chunk_delay

    def stream(self, _messages):
        time.sleep(self.first_chunk_delay)
        for index, chunk in enumerate(self.chunks):
            if index:
                time.sleep(self.per_chunk_delay)
            yield chunk


class RuntimeTurnTests(unittest.TestCase):
    def setUp(self):
        self._show_thinking = core.SHOW_THINKING
        core.SHOW_THINKING = False

    def tearDown(self):
        core.SHOW_THINKING = self._show_thinking

    def _runtime(self, chunks):
        runtime = agent_runtime.AgentRuntime("stub-model")
        runtime._llm_for_tools = lambda tools: _FakeLLM(chunks)
        return runtime

    def test_the_client_asks_the_server_for_usage(self):
        # Without this the streamed response carries no usage at all - see usage.py.
        bound = agent_runtime.AgentRuntime("stub-model")._build_llm("stub-model", [])
        client = getattr(bound, "bound", bound)
        self.assertTrue(client.stream_usage)

    def test_a_turn_reports_server_tokens_and_a_real_speed(self):
        chunks = [
            AIMessageChunk(content="hel"),
            AIMessageChunk(content="lo", usage_metadata={
                "input_tokens": 41, "output_tokens": 9, "total_tokens": 50,
            }),
        ]
        runtime = self._runtime(chunks)
        runtime.invoke("hi")

        turn = runtime.turn_usage
        self.assertEqual(turn.calls, 1)
        self.assertEqual((turn.input_tokens, turn.output_tokens, turn.total_tokens), (41, 9, 50))
        self.assertFalse(turn.estimated)
        self.assertIsNotNone(turn.tokens_per_second)
        self.assertGreater(turn.ttft_seconds, 0.0)
        self.assertGreater(turn.wall_seconds, 0.0)
        self.assertIn("41 in / 9 out / 50 tok", turn.format_line())

    def test_each_turn_starts_from_zero_and_the_session_keeps_the_total(self):
        chunks = [AIMessageChunk(content="hi", usage_metadata={
            "input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
        })]
        runtime = self._runtime(chunks)
        runtime.invoke("one")
        runtime.invoke("two")

        self.assertEqual(runtime.turn_usage.output_tokens, 5, "per-turn counters must reset")
        self.assertEqual(runtime.session_usage.turns, 2)
        self.assertEqual(runtime.session_usage.output_tokens, 10)

    def test_a_failed_turn_still_reports_what_it_burned(self):
        class _Boom(_FakeLLM):
            def stream(self, messages):
                yield from super().stream(messages)
                raise RuntimeError("backend died mid-turn")

        runtime = agent_runtime.AgentRuntime("stub-model")
        chunk = AIMessageChunk(content="partial", usage_metadata={
            "input_tokens": 7, "output_tokens": 3, "total_tokens": 10,
        })
        runtime._llm_for_tools = lambda tools: _Boom([chunk])
        with self.assertRaises(RuntimeError):
            runtime.invoke("hi")
        # The call never completed, so there are no tokens to report - but the turn's wall
        # clock is still closed out rather than left at zero from the previous turn.
        self.assertGreater(runtime.turn_usage.wall_seconds, 0.0)
        self.assertEqual(runtime.session_usage.turns, 1)


if __name__ == "__main__":
    unittest.main()
