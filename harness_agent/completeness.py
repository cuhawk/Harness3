"""Completeness envelope: the deterministic layer stamps how complete a delegated run was.

Why this exists
---------------
Several tools already report their own caps honestly - ``grep_files``/``find_files`` return
``matches_total``/``returned``/``capped``, ``spawn_map_reduce`` reports files it never examined.
That information used to die INSIDE the sub-agent: the cheap model read the capped result,
wrote a confident prose answer, and the cap never reached the caller. One router model has
already reported a ``max_results``-capped grep count as the complete total.

So the model is deliberately NOT the party that writes the completeness flag. The Harness
observes every raw tool result as it comes back, records any cap/truncation signal it can see
in the result's own key shape, and derives ``status`` from that. A model has no way to launder
a cap because it never touches this value.

What "complete" actually means
------------------------------
``status == "complete"`` means NO CAP SIGNAL WAS OBSERVED. It does NOT mean the search was
exhaustive. Several tools cap or drop data with no machine-readable signal at all, and this
detector cannot invent one for them:

  - ``fs_tools.list_dir``        - clamps max_entries to 1000 and emits no truncation flag
  - ``graphql_tools.graphql_schema_search`` - slices its result list silently
  - ``memory_search`` / ``rag_search`` / ``searxng_search`` - top-k retrieval, no denominator
  - ``fs_tools.read_file``       - a partial read (offset/max_bytes) is indistinguishable
                                   from a whole one at the envelope level
  - ``spawn_map_reduce``         - swallows path-resolution failures with a bare ``continue``

Treating "no signal" as proof of exhaustiveness would reproduce, one layer up, exactly the
laundering this module exists to stop. Say "no cap was observed", never "nothing was missed".

Lossy vs lossless signals
-------------------------
Not everything worth REPORTING is worth DOWNGRADING a run for. A capped grep means the model
answered off evidence that no longer exists anywhere; an answer spilled to disk means every byte
was kept and the caller has the path. Spending the same alarm on both is how a status field stops
being read: with map_reduce's 1500-char default, every verbose-but-perfectly-complete sweep would
report as partial, and the operator goes straight back to hand-verifying everything - the exact
tax this feature exists to remove.

So a signal carries ``lossless: True`` when the data it reports on is still fully available. Those
signals are listed in ``coverage`` (the caller is told), but they do not make a run partial and do
not set ``coverage.capped``. Anything that does not say otherwise is treated as lossy, so the
default for a signal from a tool this module has never heard of is the safe one.
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any

# The only three status values in the system. Exported as constants so no other module has to
# hand-write one of the literals - see compute_status' docstring for why that matters.
STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_FAILED = "failed"

# core.trim_text appends exactly this when a tool result blows past MAX_OUTPUT_BYTES (256 KB by
# default). That leaves the result as INVALID JSON, so the detector has to strip it before it
# can parse anything - and the trim is itself an uncounted truncation worth reporting.
_TRIM_MARKER = "\n...[truncated]..."

# Keys whose truthy presence means "this result is not the whole story". Matched on the key
# SHAPE, never on a tool name allowlist: any tool that adopts one of these conventions is
# covered the day it is written, without touching this file.
_SKIPPED_KEY_RE = re.compile(r"^(files_skipped|skipped_)")

# How many signals survive into the envelope. A 500-tool-call run that caps on every call would
# otherwise flood the MCP caller with the very noise this feature exists to summarize.
_MAX_SIGNALS = 20


def detect_cap_signals(tool_name: str, content: Any) -> list[dict[str, Any]]:
    """Inspect one raw tool result and report every cap/truncation signal it advertises.

    Fails soft by design: plain-string tools, binary blobs and malformed JSON all return ``[]``
    rather than raising. A detector that can break a delegation is worse than one that misses a
    signal, since it would take down runs that were previously fine.
    """
    text = content if isinstance(content, str) else str(content)
    signals: list[dict[str, Any]] = []

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except Exception:
        if text.endswith(_TRIM_MARKER):
            try:
                parsed = json.loads(text[: -len(_TRIM_MARKER)])
            except Exception:
                parsed = None
            # Whether or not the stripped body parses, the trim itself already lost bytes.
            signals.append({
                "tool": tool_name,
                "reason": "tool_output_trimmed_256kb",
                "returned": None,
                "total": None,
            })

    if not isinstance(parsed, dict):
        return signals

    # jobs.check_job nests the real payload one level down under "result", so a cap reported by
    # the delegation it is reporting on would otherwise be invisible. One level only - deeper
    # recursion would start matching unrelated user data that happens to use these key names.
    for scope in (parsed, parsed.get("result") if isinstance(parsed.get("result"), dict) else None):
        if not isinstance(scope, dict):
            continue
        signals.extend(_signals_for_scope(tool_name, scope))
    return signals


def detect_model_view_signals(tool_name: str, content: Any, model_content: Any) -> list[dict[str, Any]]:
    """Report the truncation the HARNESS ITSELF performs before the model sees a tool result.

    This is the largest and most frequent truncation in the system and the one most likely to be
    laundered: ``agent_runtime.compact_tool_result_for_model`` slices any result over
    ``core.MODEL_TOOL_RESULT_BYTES`` (12 KB) down to a preview, and a tool that reports no cap of
    its own - read_file, list_tree under max_entries, any run_terminal with 20 KB of stdout -
    hands the model a prefix and the caller a clean verdict. The model answers "3 errors" off the
    first 12 KB of a 200 KB log and nothing anywhere records that it only saw 6% of it.

    Lossy on purpose even though the full text survives in ``ToolMessage.artifact``: the artifact
    is not what the model reasoned from, and the answer is what the caller acts on.
    """
    raw = content if isinstance(content, str) else str(content)
    shown = model_content if isinstance(model_content, str) else str(model_content)
    if shown == raw:
        return []
    return [{
        "tool": tool_name or "unknown",
        "reason": "model_view_truncated",
        "returned": len(shown.encode("utf-8", errors="replace")),
        "total": len(raw.encode("utf-8", errors="replace")),
    }]


def _signals_for_scope(tool_name: str, d: dict[str, Any]) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    # An error is NOT a cap. ok=false already propagates on its own channel, and counting it
    # here would report every failed tool call as a coverage gap.
    if d.get("capped"):
        signals.append({
            "tool": tool_name,
            "reason": "capped",
            "returned": d.get("returned"),
            "total": d.get("matches_total") or d.get("files_matched") or d.get("total"),
        })
    for key in ("truncated", "truncated_results"):
        if d.get(key):
            signals.append({
                "tool": tool_name,
                "reason": key,
                "returned": d.get("returned") or d.get("count"),
                "total": d.get("total_chars") or d.get("total"),
            })
    for key, value in d.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if value > 0 and _SKIPPED_KEY_RE.match(str(key)):
            signals.append({"tool": tool_name, "reason": str(key), "returned": None, "total": value})
    signals.extend(_nested_envelope_signals(tool_name, d))
    return signals


def _nested_envelope_signals(tool_name: str, d: dict[str, Any]) -> list[dict[str, Any]]:
    """Recognise the harness's OWN envelope, so the verdict COMPOSES instead of being laundered.

    MAX_SUBAGENT_DEPTH defaults to 2 and the delegation tools are selectable inside a sub-run, so
    "sub-agent delegates a capped sweep, then answers from it" is a supported path - and a
    delegation result is precisely the tool result most likely to be carrying a cap. Without this
    the outer run reads its own richest cap report as an ordinary dict and finds nothing in it.

    Reading the inner ``status`` rather than re-deriving from the inner signals is deliberate: the
    lossy/lossless decision was already made one level down by the same function, so a nested run
    that merely spilled its answer to disk correctly contributes nothing here.
    """
    coverage = d.get("coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    # delegation_status is jobs.check_job's hoisted copy (its own "status" means running/done).
    inner = d.get("delegation_status") or d.get("status")
    if inner != STATUS_PARTIAL and not coverage.get("capped"):
        return []
    return [{
        "tool": tool_name,
        "reason": "nested_run_partial",
        "returned": coverage.get("covered"),
        "total": coverage.get("total"),
    }]


class CompletenessRecorder:
    """Run-scoped collector of cap signals, owned by one AgentRuntime.

    Thread safety is real, not theoretical: ``spawn_map_reduce``/``spawn_subagents_parallel``
    run many delegations concurrently. Each gets its OWN runtime (and therefore its own
    recorder) because the runtime is constructed inside the worker thread, so the lock here
    only guards the graph's own tool loop - but LangGraph is free to change that, and the cost
    of the lock is nil.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._signals: list[dict[str, Any]] = []

    def record(self, tool_name: str, content: Any, model_content: Any = None) -> None:
        """Observe one tool result. ``model_content`` is the compacted text the MODEL will
        actually receive, when the caller has it - passing it is what makes the harness's own
        12 KB budget visible instead of being the one truncation nobody reports."""
        try:
            found = detect_cap_signals(tool_name or "unknown", content)
            if model_content is not None:
                found = found + detect_model_view_signals(tool_name or "unknown", content, model_content)
        except Exception:
            return  # observing a run must never be able to break it
        if not found:
            return
        with self._lock:
            self._signals.extend(found)

    def drain(self) -> list[dict[str, Any]]:
        """Return everything recorded so far and reset.

        Draining rather than snapshotting is what makes an AgentRuntime reusable across
        validator retries without double-counting: the retry loop drains after every attempt,
        including attempts that raised, so a signal is attributed to the run exactly once.
        """
        with self._lock:
            out = self._signals
            self._signals = []
        return out


def lossy_signals(signals: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """The signals that represent data the caller can no longer get at.

    Absence of the flag means lossy, so a signal from a tool this module has never seen defaults
    to the safe reading. See the module docstring for why the distinction exists at all.
    """
    return [s for s in (signals or []) if not (isinstance(s, dict) and s.get("lossless"))]


def compute_status(ok: bool, signals: list[dict[str, Any]]) -> str:
    """The single place a delegation's completeness verdict is decided.

    The model is never the party that writes this flag - it is derived from cap/truncation
    signals the deterministic layer observed. No caller may synthesise a status string of its
    own; call envelope() instead. This is the A3 invariant: there is no code path that returns
    a clean verdict while a LOSSY signal exists, because there is only one path.
    """
    if not ok:
        return STATUS_FAILED
    if lossy_signals(signals):
        return STATUS_PARTIAL
    return STATUS_COMPLETE


def envelope(
    ok: bool,
    signals: list[dict[str, Any]] | None = None,
    covered: int | None = None,
    total: int | None = None,
    gaps: list[Any] | None = None,
) -> dict[str, Any]:
    """Build the ``{status, coverage}`` pair. The ONLY constructor of that pair anywhere.

    ``covered``/``total``/``gaps`` stay None/empty unless the Harness genuinely knows them from
    its own enumeration - today that is only spawn_map_reduce's glob expansion. A single
    delegation has no denominator, and inventing one (say, by summing a grep's matches_total)
    would be a fabricated coverage ratio, which is worse than an honest null.
    """
    signals = list(signals or [])
    shown = signals[:_MAX_SIGNALS]
    if len(signals) > _MAX_SIGNALS:
        shown.append({"reason": "signals_truncated", "total": len(signals)})
    return {
        "status": compute_status(ok, signals),
        "coverage": {
            # capped answers "was something lost", so a disclosed-but-lossless signal (a spilled
            # answer) is listed below without raising the alarm.
            "capped": bool(lossy_signals(signals)),
            "signals": shown,
            "covered": covered,
            "total": total,
            "gaps": list(gaps or []),
        },
    }
