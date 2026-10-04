"""Per-turn token accounting and generation-speed measurement for a chat turn.

The numbers rendered under a reply are the SERVER's, not a local tokenizer's guess. An
OpenAI-compatible streaming response only carries usage if the request asks for it
(``stream_options={"include_usage": true}``, which langchain-openai spells
``stream_usage=True`` - see ``agent_runtime.AgentRuntime._build_llm``). Verified against
both backends this harness actually talks to: the LiteLLM proxy on :4000 and a local
llama.cpp server each emit a final usage-only chunk when asked, and langchain merges it
into the accumulated message's ``usage_metadata``. Without that flag ``usage_metadata`` is
``None`` and the only honest fallback left is a chars/4 estimate - which is why anything
not server-reported renders with a leading ``~``.

Timing is measured client-side around the stream loop and split into time-to-first-token
and decode time, because a turn that calls tools spends most of its wall clock *inside
tools*. Dividing output tokens by wall time would report a "tok/s" several times below the
model's real generation speed, so tok/s here is output tokens over decode time only -
the same quantity llama.cpp reports as ``predicted_per_second``. Wall time is still shown
separately, since that is what the operator actually waited.

A turn is one ``AgentRuntime.invoke``/``invoke_streaming`` call, which can make several
model calls (one per agent<->tool hop), so every field accumulates across the hops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Chars per token for the fallback estimate. Deliberately crude: it is only reached when
# the backend refuses to report usage at all, and its output is always marked with "~".
_ESTIMATE_CHARS_PER_TOKEN = 4


def _int(value: Any) -> int:
    """Coerce a provider-supplied count to a non-negative int (they are occasionally null,
    a float, or a string, and never worth crashing a turn over)."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return number if number > 0 else 0


def _message_text(message: Any) -> str:
    """Best-effort plain text of an AIMessage, whether .content is a string or a list of
    content blocks. Only used for the estimate fallback."""
    # .text is a property in langchain-core 1.x and a method in older releases; the 1.x
    # value is a str that is still callable for compatibility, so test for str FIRST or
    # every call emits a deprecation warning.
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    if callable(text):
        try:
            return str(text() or "")
        except Exception:
            pass
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    return str(content or "")


def usage_from_message(message: Any) -> tuple[dict[str, int], bool]:
    """Pull token counts off one accumulated AIMessage/AIMessageChunk.

    Returns ``(counts, estimated)``. Three sources are tried in descending order of
    trust, because which one is populated depends on the backend and on whether the call
    was streamed:

    1. ``usage_metadata`` - langchain's normalized shape, what the streaming path fills in.
    2. ``response_metadata["token_usage"]`` - the raw OpenAI shape, what a non-streaming
       call leaves behind (and what some proxies put there even when streaming).
    3. A chars/4 estimate of the output text, flagged as estimated so the caller can mark
       it. Input tokens are left at 0 rather than guessed: the prompt is the whole
       conversation plus every tool schema, and a chars/4 guess of that is not a number
       worth showing.
    """
    counts = {"input": 0, "output": 0, "total": 0, "reasoning": 0, "cached_input": 0}
    if message is None:
        return counts, True

    meta = getattr(message, "usage_metadata", None)
    if isinstance(meta, dict) and (meta.get("input_tokens") or meta.get("output_tokens")):
        counts["input"] = _int(meta.get("input_tokens"))
        counts["output"] = _int(meta.get("output_tokens"))
        counts["total"] = _int(meta.get("total_tokens")) or counts["input"] + counts["output"]
        counts["reasoning"] = _int((meta.get("output_token_details") or {}).get("reasoning"))
        counts["cached_input"] = _int((meta.get("input_token_details") or {}).get("cache_read"))
        return counts, False

    response_meta = getattr(message, "response_metadata", None) or {}
    raw = response_meta.get("token_usage") or response_meta.get("usage")
    if isinstance(raw, dict) and (raw.get("prompt_tokens") or raw.get("completion_tokens")):
        counts["input"] = _int(raw.get("prompt_tokens"))
        counts["output"] = _int(raw.get("completion_tokens"))
        counts["total"] = _int(raw.get("total_tokens")) or counts["input"] + counts["output"]
        counts["reasoning"] = _int((raw.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        counts["cached_input"] = _int((raw.get("prompt_tokens_details") or {}).get("cached_tokens"))
        return counts, False

    text = _message_text(message)
    counts["output"] = (len(text) + _ESTIMATE_CHARS_PER_TOKEN - 1) // _ESTIMATE_CHARS_PER_TOKEN
    counts["total"] = counts["output"]
    return counts, True


def _fmt_seconds(seconds: float) -> str:
    if seconds >= 60:
        minutes, rest = divmod(seconds, 60)
        return f"{int(minutes)}m{rest:04.1f}s"
    if seconds >= 10:
        return f"{seconds:.1f}s"
    return f"{seconds:.2f}s"


@dataclass
class Usage:
    """Token/time accumulator for one turn, or - once merged - for a whole session."""

    turns: int = 0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    estimated: bool = False
    ttft_seconds: float | None = None
    decode_seconds: float = 0.0
    thinking_seconds: float = 0.0
    llm_seconds: float = 0.0
    wall_seconds: float = 0.0

    def record_call(
        self,
        message: Any,
        *,
        started: float,
        first_chunk_at: float | None,
        ended: float,
        first_content_at: float | None = None,
    ) -> None:
        """Fold one completed model call into this turn.

        ``first_chunk_at`` is the first chunk of ANY kind, which marks the end of prefill.
        It must not be narrowed to chunks carrying visible text: a reasoning model generates
        its thinking phase token by token and the server bills those tokens inside
        completion_tokens, so gating on visible content moves real generation time into ttft
        while its tokens stay in the numerator - measured at a 2.1x overstatement on Qwen3.8
        (152.3 reported against 72.5 true). See the comment at the stream loop in
        agent_runtime for why the reasoning deltas cannot simply be detected instead.

        So ``ended - first_chunk_at`` is decode time and ``first_chunk_at - started`` is
        time-to-first-token (prompt processing plus queueing). A call that produced no
        chunk at all contributes its latency to ``llm_seconds`` but nothing to decode
        time, which keeps it out of the tok/s denominator instead of deflating it.

        ``first_content_at`` is the first VISIBLE chunk. The gap between the two is the
        thinking phase; it stays inside decode time (it generated tokens) but is reported
        separately, because a turn that spends most of its decode window thinking looks
        identical in the footer to one that streamed an answer.
        """
        counts, estimated = usage_from_message(message)
        self.calls += 1
        self.input_tokens += counts["input"]
        self.output_tokens += counts["output"]
        self.total_tokens += counts["total"]
        self.reasoning_tokens += counts["reasoning"]
        self.cached_input_tokens += counts["cached_input"]
        self.estimated = self.estimated or estimated
        self.llm_seconds += max(0.0, ended - started)
        if first_chunk_at is not None:
            self.decode_seconds += max(0.0, ended - first_chunk_at)
            if first_content_at is not None and first_content_at > first_chunk_at:
                self.thinking_seconds += first_content_at - first_chunk_at
            if self.ttft_seconds is None:
                self.ttft_seconds = max(0.0, first_chunk_at - started)

    def merge(self, other: "Usage") -> None:
        """Accumulate a finished turn into a running total. ``ttft_seconds`` is left alone:
        a first-token latency is a property of one call, and averaging it across a session
        would invite reading it as the next turn's expected latency."""
        self.turns += max(1, other.turns)
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.estimated = self.estimated or other.estimated
        self.decode_seconds += other.decode_seconds
        self.thinking_seconds += other.thinking_seconds
        self.llm_seconds += other.llm_seconds
        self.wall_seconds += other.wall_seconds

    @property
    def tokens_per_second(self) -> float | None:
        if self.decode_seconds <= 0 or self.output_tokens <= 0:
            return None
        return self.output_tokens / self.decode_seconds

    def _mark(self) -> str:
        return "~" if self.estimated else ""

    def format_line(self) -> str:
        """The one-line footer under a reply. Compact by design - it prints after every
        turn, so anything that is zero or unknown is omitted rather than shown as 0."""
        mark = self._mark()
        total = self.total_tokens or (self.input_tokens + self.output_tokens)
        parts = [f"{mark}{self.input_tokens:,} in / {mark}{self.output_tokens:,} out / {mark}{total:,} tok"]
        if self.cached_input_tokens:
            parts.append(f"{self.cached_input_tokens:,} cached")
        if self.reasoning_tokens:
            parts.append(f"{self.reasoning_tokens:,} reasoning")
        speed = self.tokens_per_second
        parts.append(f"{mark}{speed:.1f} tok/s" if speed is not None else "tok/s n/a")
        if self.ttft_seconds is not None:
            parts.append(f"ttft {_fmt_seconds(self.ttft_seconds)}")
        # The thinking share of decode. Without it a turn that spent 90% of its decode
        # window in <think> is indistinguishable from one that streamed an answer, and the
        # tok/s is read as answer throughput when it is mostly reasoning throughput.
        if self.thinking_seconds > 0:
            parts.append(f"think {_fmt_seconds(self.thinking_seconds)}")
        parts.append(f"{_fmt_seconds(self.wall_seconds)} wall")
        if self.calls > 1:
            # ttft is the FIRST call's only (see merge), so on a multi-call turn the prompt
            # processing for calls 2..N is invisible. Total in-model time exposes it: the
            # remainder against wall is time spent inside tools.
            parts.append(f"{self.calls} llm calls, {_fmt_seconds(self.llm_seconds)} in model")
        if self.estimated:
            parts.append("~ = estimated (backend reported no usage)")
        return "  ·  ".join(parts)

    def format_totals(self) -> str:
        """Session-cumulative view, shown by bare ``/usage``."""
        if not self.turns:
            return "session totals: no turns yet"
        mark = self._mark()
        total = self.total_tokens or (self.input_tokens + self.output_tokens)
        speed = self.tokens_per_second
        speed_text = f"{mark}{speed:.1f} tok/s avg" if speed is not None else "tok/s n/a"
        return (
            f"session totals: {self.turns} turns  ·  {self.calls} llm calls  ·  "
            f"{mark}{self.input_tokens:,} in / {mark}{self.output_tokens:,} out / {mark}{total:,} tok  ·  "
            f"{speed_text}  ·  {_fmt_seconds(self.llm_seconds)} model  ·  {_fmt_seconds(self.wall_seconds)} wall"
        )
