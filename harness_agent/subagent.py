"""Sub-agent delegation: let the agent spawn an isolated sub-run of itself for a focused,
self-contained sub-task instead of flooding the main conversation with intermediate tool calls.
Also exposes a parallel fan-out variant that runs several independent sub-tasks concurrently in
a thread pool, sharing a single depth-guard increment across the whole batch.

Result-quality retry loop
-------------------------
The LiteLLM proxy already loops on the *transport* failure: a NON-200 response (rate limit,
5xx, timeout) trips ``allowed_fails`` and fails over to the next model/key pool. That loop
cannot see the *content* of a 200 response, so it is blind to the case the operator cares
about here - the model answered with HTTP 200 but the answer is wrong, malformed, or
incomplete ("still needs more data").

``_run_subagent_task`` closes that gap. After each 200 it runs the returned text through a
validator; if the validator rejects it, the loop re-invokes the SAME runtime (so the model
keeps its own prior answer + the rejection reason in context - exactly what the "needs more
data" case wants) and, on the next attempt, optionally escalates to a stronger model. It
gives up after ``max_attempts`` and returns the best (last) result it got, flagged
``ok=false`` with the full per-attempt validation history so the caller can see why.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from langchain_core.tools import tool

from . import completeness
from . import core
from . import ledger
from . import sessions

# A validator takes the candidate final text and returns (accepted, reason). ``reason`` is
# fed back to the model verbatim on rejection, so write it as an instruction to the model
# ("missing required key 'total'"), not as a log line.
Validator = Callable[[str], "tuple[bool, str]"]


# =============================================================================
# Result validation (the "200 but bad/incomplete data" check)
# =============================================================================

def _strip_code_fence(text: str) -> str:
    """Peel a ```lang ... ``` fence off a model answer so json.loads sees raw JSON.

    Models routinely wrap JSON in a markdown fence even when told not to; treating that as
    "not valid JSON" would reject perfectly good answers, so unwrap it before parsing.
    """
    stripped = (text or "").strip()
    match = re.match(r"^```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?```$", stripped, re.S)
    return match.group(1).strip() if match else stripped


# --- Citation checking ------------------------------------------------------
#
# A delegated answer is only worth trusting if its file references are real. The other checks
# police the SHAPE of an answer; this one is the only check that leaves the string and touches
# the filesystem, which is why it runs last in the chain.
#
# The dominant risk is the opposite of the obvious one: not missing a fake citation, but
# rejecting a good answer because something that merely LOOKS like "path:line" (a URL port, a
# timestamp, a version, "Note: this") got treated as a citation and then failed verification.
# The defence is two-layered and the second layer is the important one:
#   1. a shape pre-filter, and
#   2. the rule that anything failing the pre-filter is IGNORED, never rejected.
# Only a candidate that passes the pre-filter AND then fails filesystem verification can fail
# an answer. A URL therefore cannot poison an answer that also carries two real citations.
_CITATION_RE = re.compile(
    # The token must begin on a boundary. Blocking ':' and '/' here is what makes URLs
    # unmatchable: every segment of "https://example.com:8080/x.js:12" is preceded by '/' or
    # sits behind the scheme's ':', and the scheme itself has no ".ext" before its colon.
    # Blocking '.' stops a match starting mid-filename ("local.py" out of "config.local.py"),
    # which would otherwise verify a path the model never wrote.
    r"(?<![\w.+@$\\/:])"
    r"(?P<path>"
    # Absolute forms: a Windows drive, or a POSIX root. Without the "/" branch the path group
    # cannot start with a separator, so on a POSIX host - where every absolute citation looks
    # like /srv/ws/app.py:12 - NO absolute citation was a candidate at all, and an answer whose
    # references were all absolute counted zero and failed citation_min. A URL cannot sneak in
    # this way: the boundary lookbehind above rejects a match whose preceding character is a
    # word char or a separator, which is every "/" inside "https://host/path".
    r"(?:[A-Za-z]:[\\/]|/)?"
    r"(?:[\w.\-+@$]+[\\/])*"                        # optional directory segments, / or \
    r"[\w\-+@$]+(?:\.[\w\-+@$]+)*"                  # stem, may itself contain dots
    r"\.(?P<ext>[A-Za-z0-9_]{1,10})"                # an extension is mandatory
    r")"
    r":(?P<line>\d+)(?:-(?P<end>\d+))?"
    # Reject "12px" and "12.5" but deliberately ALLOW a trailing sentence period, since
    # "see src/app.py:12." is how a model actually writes one.
    r"(?!\w)(?!\.\d)"
)

# An extension-only token with one of these and no path separator is a hostname, not a file:
# "example.com:8080" must not be read as line 8080 of a file called example.com. This is only a
# fast path now - the rule that decides the hard cases is in _verify_citation: a separator-less
# token that matches no file in the workspace is IGNORED, never rejected, because there is no way
# to tell "api.staging.corp:8443" or "org.slf4j.Logger:5" from an invented bare filename, and
# guessing wrong in the rejecting direction burns the whole retry ladder on a correct answer.
_TLD_LIKE = {"com", "net", "org", "io", "dev", "app", "co", "uk", "ai", "gov", "edu", "sh", "me"}

# A Windows drive, a UNC/rooted backslash, or a POSIX root. Used only to decide whether an
# out-of-workspace path is unverifiable (ignore) or an escape attempt (reject).
_ABSOLUTE_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|[\\/])")

# Counting the lines of a huge file to check one citation is not worth the stall; over this
# size the citation is accepted unverified (see _citation_line_count).
_MAX_CITATION_FILE_BYTES = 20_000_000

# Bare-basename resolution ("see subagent.py:225" after naming harness_agent/ a sentence earlier)
# walks the workspace, so it is bounded: a bug-bounty workspace can be a multi-GB tree and a
# validator that stalls on it is a validator that gets turned off. Hitting the limit only means
# some bare citations go uncounted - it can never turn into a rejection.
_INDEX_SCAN_LIMIT = 20_000
_INDEX_PATHS_PER_NAME = 8
_INDEX_SKIP_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".agent_cache",
    ".venv", "venv", ".tox", ".mypy_cache", ".pytest_cache",
}


def _citation_candidates(text: str) -> list[tuple[str, str, int]]:
    """Every (raw_match, path, line) in ``text`` that is plausibly a real file citation.

    Candidates that fail the shape pre-filter are dropped SILENTLY - they are not citations,
    so they are neither counted toward citation_min nor grounds for rejection. That is the
    whole false-positive defence; do not turn any of these `continue`s into a rejection.

    For a range (path:10-14) the END line is what gets verified: if the last line exists the
    whole range is inside the file.
    """
    found: list[tuple[str, str, int]] = []
    for m in _CITATION_RE.finditer(text or ""):
        path = m.group("path")
        ext = m.group("ext")
        if ext.isdigit():
            continue  # "v1.2:3", "3.14:15" - a version or a decimal, not a file
        if ("/" not in path and "\\" not in path) and ext.lower() in _TLD_LIKE:
            continue  # a bare hostname with a port
        line = int(m.group("end") or m.group("line"))
        if not (1 <= line <= 10_000_000):
            continue  # ":0" is not a line number; the huge end is a sanity bound
        found.append((m.group(0), path, line))
    return found


def _citation_line_count(rp: Path) -> int:
    """Line count of ``rp``, or -1 meaning "unknown - accept the citation".

    Fails OPEN on an unreadable or oversized file, mirroring _judge_validator's behavior on a
    judge outage: a local IO problem must not discard an otherwise-good answer.
    """
    try:
        if rp.stat().st_size > _MAX_CITATION_FILE_BYTES:
            return -1
        return len(rp.read_bytes().decode("utf-8", errors="replace").splitlines())
    except Exception:
        return -1


def _basename_index() -> dict[str, list[Path]]:
    """basename -> the files with that name anywhere in the workspace (bounded, see the limits).

    Exists because ``core.resolve_path`` joins a relative path onto the workspace ROOT, so a bare
    "subagent.py:225" only ever resolved for root-level files - while a model that has already
    written "harness_agent/" earlier in the paragraph goes on citing basenames, and every one of
    those references is to a file it genuinely read.
    """
    index: dict[str, list[Path]] = {}
    scanned = 0
    stack = [core.WORKSPACE]
    while stack and scanned < _INDEX_SCAN_LIMIT:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                entries = sorted(it, key=lambda e: e.name)
        except Exception:
            continue  # unreadable directory: skip it, never fail the answer over it
        for entry in entries:
            scanned += 1
            if scanned >= _INDEX_SCAN_LIMIT:
                break
            try:
                if entry.is_dir(follow_symlinks=False):
                    if entry.name not in _INDEX_SKIP_DIRS:
                        stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    bucket = index.setdefault(entry.name, [])
                    if len(bucket) < _INDEX_PATHS_PER_NAME:
                        bucket.append(Path(entry.path))
            except Exception:
                continue
    return index


def _citation_targets(path: str, cache: dict[str, Any]) -> tuple[list[Path], str]:
    """Which real files a citation could be pointing at. Returns (candidates, problem).

    A non-empty ``problem`` is a hard rejection. An EMPTY candidate list with no problem means
    "this is not a file reference" - ignore it silently.

    The split is the whole false-positive defence, one layer below the shape pre-filter:
      - a path with directory segments ("src/ghost.py") is a claim about a specific location, so
        a miss is a fabricated citation and rejecting it is right;
      - a bare token ("api.staging.corp", "org.slf4j.Logger", "margin.top") is indistinguishable
        from a hostname, a dotted identifier or a CSS property, so a miss means ignore it.
    """
    if "/" in path or "\\" in path:
        try:
            rp = core.resolve_path(path)
        except PermissionError:
            # An ABSOLUTE path outside the workspace is unverifiable, not fabricated - it is how
            # an agent cites a file it read over ssh_run, or one on another root - so it falls
            # under the same "can't tell, so ignore" rule as a bare token. A RELATIVE path that
            # escapes ("../../etc/passwd.txt") is a different animal: it is anchored to the
            # workspace by construction and climbing out of it is unambiguous, so it still fails.
            if _ABSOLUTE_PATH_RE.match(path):
                return [], ""
            return [], "escapes the workspace"
        except Exception as exc:
            return [], f"is not a usable path ({exc})"
        return ([rp], "") if rp.is_file() else ([], "no such file")

    root = core.WORKSPACE / path
    if root.is_file():
        return [root], ""
    index = cache.get("index")
    if index is None:
        index = _basename_index()
        cache["index"] = index
    return list(index.get(path, [])), ""


def _verify_citation(path: str, line: int, cache: dict[str, Any]) -> tuple[str, str] | None:
    """Check one citation against the real filesystem. Returns (problem, identity), or None.

    ``None`` means "not a citation at all, ignore it" - see _citation_targets. ``problem`` is ""
    when the citation is good, otherwise a fragment naming what is wrong. ``identity`` is the
    resolved path, used to count DISTINCT citations - so "src/app.py:12" and ".\\src\\app.py:12"
    are one citation, not two.

    Everything workspace-derived is resolved here, at validate time: core.set_workspace rebinds
    core.WORKSPACE, and a path resolved when the validator was BUILT would point at the old one.
    """
    targets, problem = _citation_targets(path, cache)
    if problem:
        return problem, path
    if not targets:
        return None
    line_cache: dict[str, int] = cache.setdefault("lines", {})
    longest = -1
    for rp in targets:
        key = str(rp)
        n_lines = line_cache.get(key)
        if n_lines is None:
            n_lines = _citation_line_count(rp)
            line_cache[key] = n_lines
        if n_lines < 0:
            return "", key  # unreadable or oversized: accepted unverified
        if line <= n_lines:
            return "", key
        # Same basename in several directories: the citation is good if ANY copy has the line,
        # so only report the failure once every candidate has been ruled out.
        longest = max(longest, n_lines)
    return f"points past the end of the file ({longest} lines)", str(targets[0])


# --- Quoted-snippet verification -------------------------------------------
#
# Verifying that a citation POINTS somewhere real is a much weaker guarantee than it sounds:
# "scm_pipeline.py:12" passes as long as the file has twelve lines. Caught live on 2026-08-13 -
# a model fabricated a whole grep_files result, cited line 12 for a function that is actually at
# line 1812, and the answer was accepted, because the reference resolved.
#
# A quote is a claim of verbatim text, so it can be checked against the file. Only backticked
# text counts (prose paraphrase is not a claim), and a miss is a rejection ONLY when the text
# appears in none of the files the answer cited - a snippet attached to the nearest citation but
# quoted from another of them must not fail the answer.
_QUOTED_RE = re.compile(r"`{1,3}([^`\n]{8,200})`{1,3}")
_SNIPPET_WINDOW = 200
_ELLIPSES = ("...", "…")


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _quoted_near(body: str, raw: str) -> str:
    """The backticked fragment adjacent to citation ``raw``, or "" if there is none worth checking."""
    idx = body.find(raw)
    if idx < 0:
        return ""
    window = body[max(0, idx - _SNIPPET_WINDOW): idx + len(raw) + _SNIPPET_WINDOW]
    for match in _QUOTED_RE.finditer(window):
        snippet = match.group(1).strip()
        if any(e in snippet for e in _ELLIPSES):
            continue  # an elided quote is not a verbatim claim
        if _CITATION_RE.search(snippet):
            continue  # the quoted thing is the citation itself, not code from the file
        if len(snippet) >= 8:
            return snippet
    return ""


def _file_text(rp: Path, cache: dict[str, Any]) -> str:
    texts: dict[str, str] = cache.setdefault("texts", {})
    key = str(rp)
    if key not in texts:
        try:
            if rp.stat().st_size > _MAX_CITATION_FILE_BYTES:
                texts[key] = ""  # too big to scan: fail open, same as the line-count check
            else:
                texts[key] = _normalize_for_match(
                    rp.read_bytes().decode("utf-8", errors="replace")
                )
        except Exception:
            texts[key] = ""
    return texts[key]


def _snippet_problem(identity: str, snippet: str, cache: dict[str, Any]) -> str:
    """"" if the quote is real (or unverifiable), else a fragment naming the fabrication."""
    needle = _normalize_for_match(snippet)
    if not needle:
        return ""
    seen: list[str] = cache.setdefault("verified_paths", [])
    for key in [identity] + [p for p in seen if p != identity]:
        haystack = _file_text(Path(key), cache)
        if not haystack:
            return ""  # unreadable or oversized somewhere in the chain: do not reject on it
        if needle in haystack:
            return ""
    return f"the quoted text {core.trim_text(snippet, 80)!r} does not appear in that file"


# --- Degenerate output ------------------------------------------------------
#
# Runaway repetition is a generation failure, not an answer: mistral-large emits corrupted
# tool-call syntax as its final answer roughly one run in three, and it arrives as a fragment
# repeated until the token budget runs out. Nothing else in the chain catches it - the text is
# long, contains the required substrings, and can even carry a valid citation.
#
# Checked on the TAIL only, because that is what a stuck decoder produces, and a bounded window
# keeps this O(window x unit) with no backtracking. A run of one repeated character is exempt:
# "-----" and "=====" are how a model draws a rule, not how it breaks.
# How many unexamined paths a map_reduce reply shows inline. Enough to see WHAT was missed
# (which directory, which extension); the count and the cursor carry the rest.
_MAP_GAPS_INLINE = 10

_DEGENERATE_WINDOW = 8000
_DEGENERATE_MAX_UNIT = 40
_DEGENERATE_MIN_REPEATS = 20


def _degenerate_reason(text: str) -> str:
    if os.environ.get("HARNESS_ALLOW_DEGENERATE"):
        return ""
    tail = (text or "")[-_DEGENERATE_WINDOW:]
    n = len(tail)
    for unit in range(2, _DEGENERATE_MAX_UNIT + 1):
        if n < unit * _DEGENERATE_MIN_REPEATS:
            continue
        block = tail[-unit:]
        if len(set(block.strip())) < 2:
            continue
        repeats = 1
        i = n - unit
        while i - unit >= 0 and tail[i - unit:i] == block:
            repeats += 1
            i -= unit
        if repeats >= _DEGENERATE_MIN_REPEATS:
            return (
                f"the answer degenerates into {block!r} repeated {repeats} times; it was not "
                "finished - return the complete answer as prose, with no repeated fragment"
            )
    return ""


def build_validator(
    *,
    min_length: int = 0,
    require_substrings: list[str] | None = None,
    require_regex: str = "",
    require_json: bool = False,
    require_keys: list[str] | None = None,
    require_citations: bool = False,
    citation_min: int = 1,
) -> Validator | None:
    """Assemble a deterministic content validator from simple acceptance criteria.

    Returns ``None`` when no criteria are requested (the common case - the loop then makes a
    single attempt and returns whatever it gets, i.e. the pre-existing behavior). Checks run
    cheapest-first and short-circuit on the first failure so the feedback names one concrete
    problem at a time.
    """
    require_substrings = [s for s in (require_substrings or []) if s]
    require_keys = [k for k in (require_keys or []) if k]
    want_json = require_json or bool(require_keys)

    # require_citations MUST be in this guard: leave it out and asking for citations alone
    # returns None, i.e. no validator at all, and the feature is a silent no-op that every
    # unit test of the closure still passes.
    if not (min_length or require_substrings or require_regex or want_json or require_citations):
        return None

    compiled_regex = re.compile(require_regex, re.I | re.S) if require_regex else None

    def validate(text: str) -> tuple[bool, str]:
        body = (text or "").strip()
        if not body:
            return False, "the answer was empty; produce the requested output"
        if min_length and len(body) < min_length:
            return False, f"the answer is too short ({len(body)} chars, need >= {min_length}); it looks incomplete - provide the full result"
        if want_json:
            try:
                parsed = json.loads(_strip_code_fence(body))
            except Exception as exc:
                return False, f"the answer is not valid JSON ({exc}); reply with only a valid JSON value and nothing else"
            missing = [k for k in require_keys if not (isinstance(parsed, dict) and k in parsed)]
            if missing:
                return False, f"the JSON is missing required key(s) {missing}; include every required field"
        low = body.lower()
        for needle in require_substrings:
            if needle.lower() not in low:
                return False, f"the answer is missing required content {needle!r}; make sure it is present"
        if compiled_regex is not None and not compiled_regex.search(body):
            return False, f"the answer did not match the required pattern /{require_regex}/; adjust it to satisfy the requirement"
        if require_citations:
            # Last in the chain on purpose: it is the only check that hits the filesystem. The
            # cache holds both the per-file line counts and the (lazily built, at most once)
            # basename index, so a 30-citation answer walks the workspace no more than once.
            cache: dict[str, Any] = {}
            verified: set[tuple[str, int]] = set()
            quoted: list[tuple[str, str]] = []
            for raw, path, line in _citation_candidates(body):
                checked = _verify_citation(path, line, cache)
                if checked is None:
                    continue  # not a file reference; ignoring it is the false-positive defence
                problem, identity = checked
                if problem:
                    # Name the offending citation before complaining about the count, so the
                    # retry loop gets something the model can actually act on.
                    return False, (
                        f"the citation {raw!r} is not usable ({problem}); cite only files you "
                        f"actually read, as path:line"
                    )
                verified.add((identity, line))
                cache.setdefault("verified_paths", []).append(identity)
                quoted.append((raw, identity))
            # Second pass, deliberately: a snippet is only a fabrication if it appears in NONE of
            # the cited files, and the last citation in the answer is as eligible a source as the
            # nearest one. Checking inline would reject a quote whose file is cited a sentence
            # later - which is how "compare `x` with a.py:1 and b.py:2" is normally written.
            for raw, identity in quoted:
                snippet = _quoted_near(body, raw)
                if not snippet:
                    continue
                snippet_problem = _snippet_problem(identity, snippet, cache)
                if snippet_problem:
                    return False, (
                        f"the citation {raw!r} is wrong: {snippet_problem}; quote only text "
                        "you actually read, at the line where it really appears"
                    )
            if len(verified) < citation_min:
                return False, (
                    f"the answer has {len(verified)} verified file citation(s), need >= "
                    f"{citation_min}; cite the file and line for each claim, as path:line, using "
                    "the workspace-relative path"
                )
        return True, "ok"

    return validate


def _judge_validator(task: str, judge_model: str) -> Validator:
    """An LLM-as-judge acceptance check: a second model decides if the answer is correct and
    complete for ``task``. Use for semantic correctness that no regex can capture.

    Runs the judge as a bare (tool-less) LLM call rather than a full AgentRuntime - it only
    needs to render a verdict, and giving it tools would let it wander. If the judge itself
    errors (proxy down, unparseable verdict), it ACCEPTS with a flag rather than nuking an
    otherwise-good result on infrastructure trouble - the reason string records the failure.
    """
    def validate(text: str) -> tuple[bool, str]:
        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=judge_model,
                base_url=core.LITELLM_BASE_URL,
                api_key=core.LITELLM_API_KEY,
                temperature=0,
                timeout=60,
                max_retries=2,
            )
            prompt = (
                "You are a strict acceptance judge for another model's answer. Decide whether "
                "the answer below fully and correctly completes the task. Reject it if it is "
                "wrong, partial, or would need more work to be usable.\n\n"
                f"TASK:\n{task}\n\nCANDIDATE ANSWER:\n{text}\n\n"
                'Reply with ONLY a JSON object: {"accepted": true|false, "reason": "<one sentence; '
                'if rejecting, say concretely what is wrong or missing>"}'
            )
            raw = str(getattr(llm.invoke(prompt), "content", "") or "")
            verdict = json.loads(_strip_code_fence(raw))
            accepted = bool(verdict.get("accepted"))
            reason = str(verdict.get("reason") or ("ok" if accepted else "rejected by judge"))
            return accepted, reason
        except Exception as exc:
            # Don't let a judge outage discard a real result; accept but surface the failure.
            return True, f"judge unavailable ({exc!r}); accepted without semantic check"

    return validate


def _model_sequence(base_model: str, escalation: list[str] | None, attempts: int) -> list[str]:
    """The model to use on each attempt: base first, then walk the escalation list, then
    stick on the last model for any remaining attempts (so extra attempts still retry with
    feedback even once we're out of stronger models to escalate to)."""
    escalation = core.SUBAGENT_ESCALATION_MODELS if escalation is None else [m for m in escalation if m]
    seq = [base_model]
    for model in escalation:
        if len(seq) >= attempts:
            break
        seq.append(model)
    while len(seq) < attempts:
        seq.append(seq[-1])
    return seq[:attempts]


def _persist_session(runtime: Any, session_id: str, session_name: str, model: str) -> tuple[str, str]:
    """Persist a sub-agent's full history (including tool calls/results) so a later
    ``resume_subagent`` can pick the same conversation back up. Returns (session_id, error).

    Only runs when the caller asked for a session by name or id - an unnamed one-shot
    delegation stays exactly as cheap as before and leaves nothing on disk. Storage is
    ``<workspace>/.agent_cache/sessions/<id>.json`` (sessions.SessionStore), so a session
    survives this MCP server process; the delegation's browser instances and other live
    process state do NOT, since only messages are serialized.
    """
    if runtime is None or not (session_id or session_name):
        return "", ""
    try:
        store = sessions.SessionStore()
        if session_id:
            session, _ = store.load(session_id)
        else:
            session = store.create(session_name, model)
        session.model_name = model
        store.save(session.id, session, runtime.history)
        return session.id, ""
    except Exception as exc:
        # A persistence failure must not discard an otherwise-good result - surface it instead.
        return "", repr(exc)


def _spill_result(text: str, head_chars: int) -> dict[str, Any]:
    """Park an oversized answer on disk and return a head + pointer instead of the whole thing.

    Trimming a long answer silently DESTROYS the tail; writing it out keeps every byte while
    costing the caller only the head. The reader then pulls the slice it actually needs.
    """
    import hashlib
    import uuid as _uuid

    raw = text.encode("utf-8")
    try:
        out_dir = core.CACHE_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"result-{_uuid.uuid4().hex[:12]}.txt"
        path.write_text(text, encoding="utf-8")
        return {
            "result": text[:head_chars],
            "result_truncated": True,
            "result_path": str(path),
            "result_bytes": len(raw),
            "result_sha256": hashlib.sha256(raw).hexdigest()[:16],
            "note": (
                f"Answer was {len(raw)} bytes, over the {head_chars}-char budget. Only the head is "
                f"inline; the COMPLETE answer is at result_path - read the part you need from there."
            ),
        }
    except Exception:
        # Never lose a good answer to a disk problem - fall back to plain trimming.
        return {"result": core.trim_text(text, head_chars)}


def _result_payload(final_text: str, max_chars: int, spill_over_chars: int) -> dict[str, Any]:
    text = final_text or ""
    if spill_over_chars and len(text) > spill_over_chars:
        return _spill_result(text, spill_over_chars)
    return {"result": core.trim_text(text, max_chars)}


def _answer_signals(final_text: str, payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Cap signals for the ANSWER itself, so answer-level truncation lands in the same
    status/coverage story as a truncated tool result rather than sitting beside it.

    Two cases, and they are NOT equally serious - see completeness.py on lossy vs lossless:
      - spilled: the answer went to disk and only a head is inline (result_truncated=True).
        Every byte survives at result_path, so this is DISCLOSED, not lost: it is reported in
        coverage and marked lossless, and it does not make the run partial. Marking it partial
        would make "partial" the default verdict of any verbose sweep (spawn_map_reduce spills at
        1500 chars by default), which is how a status field stops being read at all.
      - trimmed: core.trim_text chopped the tail and set no flag at all. This one destroys bytes
        outright, so it is lossy and does downgrade the run.
    Detected by comparing what we are returning against what the model actually said, rather
    than by re-deriving the byte thresholds - that also catches _spill_result's disk-failure
    fallback, which trims without setting result_truncated.
    """
    text = final_text or ""
    shown = payload.get("result") or ""
    if payload.get("result_truncated"):
        return [{
            "tool": "<answer>",
            "reason": "answer_spilled_to_disk",
            "returned": len(shown),
            "total": payload.get("result_bytes"),
            "lossless": True,
        }]
    if shown != text:
        return [{
            "tool": "<answer>",
            "reason": "answer_trimmed",
            "returned": len(shown),
            "total": len(text),
        }]
    return []


def _ledger_reason(out: dict[str, Any], history: list[dict[str, Any]]) -> str:
    """The one line that explains this run's outcome, for the ledger's rejection_reason.

    On failure that is the give-up error. On a run that succeeded only on attempt N, the
    interesting thing is why attempt N-1 was rejected - "succeeded eventually" and "succeeded
    first time" are the two outcomes the ledger exists to tell apart.
    """
    if not out.get("ok"):
        return str(out.get("error") or "")
    if len(history) > 1:
        prior = history[-2] if isinstance(history[-2], dict) else {}
        return str(prior.get("reason") or prior.get("error") or "")
    return ""


def _finish_envelope(
    out: dict[str, Any],
    final_text: str,
    run_signals: list[dict[str, Any]],
    *,
    covered: int | None = None,
    total: int | None = None,
    gaps: list[Any] | None = None,
    task: str = "",
    history: list[dict[str, Any]] | None = None,
    started_at: float | None = None,
    validator_kinds: list[str] | None = None,
    glob_involved: bool = False,
    entry_point: str = "",
) -> dict[str, Any]:
    """The single exit point of _run_subagent_task: stamp status/coverage onto the envelope.

    Both the success and the give-up return paths funnel through here so the completeness
    verdict is computed once, from the same inputs, for every entry point (single delegation,
    resume, map_reduce per file, parallel per task, workflow step, verify_finding). Anything
    that must happen on EVERY delegation regardless of outcome belongs in this function.
    """
    signals = list(run_signals)
    signals.extend(_answer_signals(final_text, out))
    out.update(completeness.envelope(bool(out.get("ok")), signals, covered=covered, total=total, gaps=gaps))

    # The outcome ledger hangs off this one site precisely because every entry point funnels
    # through here, so there is no delegation that can quietly go unrecorded. It reads the
    # status the envelope just stamped rather than deriving its own - there is exactly one
    # place that verdict is decided. record_delegation cannot raise, so this cannot be the
    # reason a caller loses a good answer.
    attempt_history = list(history or [])
    last_attempt = attempt_history[-1] if attempt_history and isinstance(attempt_history[-1], dict) else {}
    ledger.record_delegation(
        task=task,
        model=str(out.get("model") or ""),
        ok=bool(out.get("ok")),
        accepted=bool(last_attempt.get("accepted")),
        attempts=int(out.get("attempts") or 0),
        latency_ms=int((time.monotonic() - started_at) * 1000) if started_at is not None else 0,
        rejection_reason=_ledger_reason(out, attempt_history),
        status=str(out.get("status") or ""),
        validator_kinds=validator_kinds,
        entry_point=entry_point,
        glob_involved=glob_involved,
        validation_history=attempt_history,
    )
    return out


def _drain_signals(runtime: Any, into: list[dict[str, Any]]) -> None:
    """Move a runtime's recorded cap signals into this run's accumulator.

    getattr-with-default is load-bearing, not defensive noise: test doubles (and any future
    runtime implementation) have no recorder, and a delegation must not fail because the thing
    running it does not happen to observe caps.
    """
    recorder = getattr(runtime, "completeness", None)
    if recorder is None:
        return
    try:
        into.extend(recorder.drain())
    except Exception:
        pass


# status/coverage are part of the delegation contract, so they must be listed here or the
# fan-out paths (spawn_subagents_parallel, spawn_map_reduce's per-file entries) would silently
# drop them on the way to the MCP caller while single delegations - which to_json the whole
# dict - kept them. That asymmetry is exactly how a "working" field ends up doing nothing.
_COMPACT_KEYS = (
    "ok", "model", "attempts", "result", "result_truncated", "result_path",
    "result_bytes", "result_sha256", "session_id", "session_error", "error", "note",
    "status", "coverage",
)


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Shrink one task's envelope for a fan-out reply.

    validation_history is per-attempt bookkeeping that is pure noise once a task SUCCEEDED
    (N tasks x M attempts of "accepted: ok"), but is the only way to diagnose one that failed -
    so it is kept exclusively on failures.
    """
    out = {key: result[key] for key in _COMPACT_KEYS if key in result}
    if not result.get("ok"):
        out["validation_history"] = result.get("validation_history")
    return out


def _last_ai_text(result: dict[str, Any]) -> str:
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", None)
        if content and msg.__class__.__name__ == "AIMessage":
            return str(content)
    return ""


# =============================================================================
# Single-task runner with the validation-retry loop
# =============================================================================

def _run_subagent_task(
    task: str,
    model_name: str = "",
    max_history_chars: int = 20000,
    *,
    validator: Validator | None = None,
    max_attempts: int | None = None,
    escalation_models: list[str] | None = None,
    runtime_factory: Callable[[str], Any] | None = None,
    restore_history: list[dict[str, Any]] | None = None,
    session_id: str = "",
    session_name: str = "",
    spill_over_chars: int = 0,
    glob_involved: bool = False,
    validator_kinds: list[str] | None = None,
    entry_point: str = "",
) -> dict[str, Any]:
    """Run one isolated sub-agent task, looping until the result passes ``validator`` (or the
    attempt budget is exhausted). Returns a plain dict (both public tools wrap it in their own
    ``core.to_json`` envelope).

    Attempt 1 sends ``task`` fresh. Each rejected attempt re-invokes the SAME runtime with the
    validator's rejection ``reason`` as the next turn, so the model sees its own prior answer
    plus what was wrong ("needs more data" -> it continues; "wrong data" -> it corrects). When
    an escalation model is configured for the next attempt, the runtime is switched to it first
    (history is preserved across the switch, so the stronger model inherits the full context).

    ``glob_involved``, ``validator_kinds`` and ``entry_point`` carry no behaviour here: they are
    provenance the outcome ledger records, so "did this model work agentically for this shape of
    task" can distinguish a per-file sweep from a one-off, an unvalidated run from a strictly
    gated one, and a workflow step from a direct delegation.
    """
    started_at = time.monotonic()
    base_model = model_name or core.DEFAULT_MODEL
    # The attempt budget applies even with no validator. Without one, a successful attempt is
    # accepted immediately and returns on the first pass (unchanged behavior) - but an attempt
    # that RAISES (the proxy exhausted its own failover and surfaced a 429/5xx, which is the
    # common failure on free-tier model pools) now escalates to the next model instead of
    # failing the whole delegation on a single provider's rate limit.
    attempts = max(1, max_attempts or core.SUBAGENT_MAX_ATTEMPTS)
    sequence = _model_sequence(base_model, escalation_models, attempts)

    if runtime_factory is None:
        # Lazy import: agent_runtime imports tools_registry which imports this module.
        from .agent_runtime import AgentRuntime

        def runtime_factory(m: str) -> Any:  # type: ignore[misc]
            rt = AgentRuntime(m, pin_tools_to_task=True)
            # Restoring inside the FACTORY (rather than seeding a runtime before the loop) is
            # what makes resume compose with the existing retry logic untouched: the loop's
            # "runtime is None -> send the task fresh" rule still holds, and an attempt that
            # errors and rebuilds the runtime gets the prior history back too.
            if restore_history:
                rt.load_serialized_history(restore_history)
            return rt

    runtime: Any = None
    history: list[dict[str, Any]] = []
    # Cap signals accumulate ACROSS attempts. A rejected attempt that hit a capped grep still
    # under-covered the task, and the runtime is rebuilt from scratch after any attempt that
    # raised - so the signals have to be pulled out per attempt, not read once at the end.
    run_signals: list[dict[str, Any]] = []
    # A resume inherits its predecessor's caps. Seeded ONCE here rather than inside
    # runtime_factory, which is re-entered whenever an attempt raises and the runtime is rebuilt
    # - seeding there would re-add the same signals on every rebuild and inflate the count.
    if restore_history:
        run_signals.extend(completeness.signals_from_history(restore_history))
    final_text = ""
    feedback = ""

    for attempt in range(1, attempts + 1):
        model = sequence[attempt - 1]
        try:
            if runtime is None:
                runtime = runtime_factory(model)
                turn_input = task
            else:
                if getattr(runtime, "model_name", model) != model:
                    runtime.switch_model(model)
                # Rejection feedback drives the retry; the model already holds the task + its
                # previous answer in history, so we only need to say what to fix.
                turn_input = (
                    f"Your previous answer was not accepted: {feedback}\n\n"
                    "Revise or continue so the answer fully satisfies the original task, then "
                    "return the corrected complete answer."
                )
            result = runtime.invoke(turn_input)
            final_text = _last_ai_text(result)
            _drain_signals(runtime, run_signals)
        except Exception as exc:
            # A raised exception here means the transport-level loop already gave up (or a
            # local error). Record it and retry from scratch if attempts remain.
            _drain_signals(runtime, run_signals)  # before the reset, or the attempt's caps vanish
            history.append({"attempt": attempt, "model": model, "ok": False, "error": repr(exc)})
            feedback = f"the previous attempt failed with an error: {exc!r}"
            runtime = None  # rebuild fresh next attempt
            continue

        # Runs even with no validator, and before one: a degenerate answer is a failed
        # generation, not a candidate to be judged, and handing one to a judge_model pays for a
        # second model call to be told what a regex already knew.
        degenerate = _degenerate_reason(final_text)
        if degenerate:
            accepted, reason = False, degenerate
        else:
            accepted, reason = (True, "ok") if validator is None else validator(final_text)
        history.append({"attempt": attempt, "model": model, "ok": True, "accepted": accepted, "reason": reason})
        if accepted:
            saved_id, save_error = _persist_session(runtime, session_id, session_name, model)
            out = {
                "ok": True,
                "task": core.trim_text(task, 500),
                "model": model,
                "attempts": attempt,
                **_result_payload(final_text, max_history_chars, spill_over_chars),
                "validation_history": history,
            }
            if saved_id:
                out["session_id"] = saved_id
            if save_error:
                out["session_error"] = save_error
            return _finish_envelope(
                out, final_text, run_signals,
                task=task, history=history, started_at=started_at,
                validator_kinds=validator_kinds, glob_involved=glob_involved,
                entry_point=entry_point,
            )
        feedback = reason

    # With no validator, the only way to fall through is every attempt raising, so name that
    # cause rather than blaming a validation step that never ran.
    cause = "no attempt completed" if validator is None else "result did not pass validation"
    saved_id, save_error = _persist_session(runtime, session_id, session_name, sequence[-1])
    out = {
        "ok": False,
        "task": core.trim_text(task, 500),
        "model": sequence[-1],
        "attempts": attempts,
        "error": f"{cause} after {attempts} attempt(s): {feedback}",
        **_result_payload(final_text, max_history_chars, spill_over_chars),
        "validation_history": history,
    }
    if saved_id:
        out["session_id"] = saved_id
    if save_error:
        out["session_error"] = save_error
    return _finish_envelope(
        out, final_text, run_signals,
        task=task, history=history, started_at=started_at,
        validator_kinds=validator_kinds, glob_involved=glob_involved,
        entry_point=entry_point,
    )


def _validation_kwargs(
    min_length: int,
    require_substrings_json: str,
    require_regex: str,
    require_json: bool,
    require_keys_json: str,
    judge_model: str,
    task: str,
    max_attempts: int,
    escalate_models: str,
    require_citations: bool = False,
    citation_min: int = 1,
) -> dict[str, Any]:
    """Turn the flat tool arguments into the keyword args ``_run_subagent_task`` expects.

    Combines the deterministic validator (build_validator) with an optional LLM judge into a
    single validator that requires BOTH to pass (cheap checks first, judge only if they hold).
    Returns ``{}`` when nothing was requested, so callers fall straight back to the original
    single-shot behavior.

    The leading nine arguments are passed POSITIONALLY by every caller (map_reduce even passes
    "" fillers for criteria it does not expose), so any new criterion goes on the END and is
    passed by keyword - inserting one in the middle silently re-aims every later argument.
    """
    substrings = core.parse_json_arg(require_substrings_json, []) if require_substrings_json else []
    keys = core.parse_json_arg(require_keys_json, []) if require_keys_json else []
    if not isinstance(substrings, list):
        substrings = []
    if not isinstance(keys, list):
        keys = []

    deterministic = build_validator(
        min_length=max(0, int(min_length or 0)),
        require_substrings=[str(s) for s in substrings],
        require_regex=str(require_regex or ""),
        require_json=bool(require_json),
        require_keys=[str(k) for k in keys],
        require_citations=bool(require_citations),
        citation_min=max(1, int(citation_min or 1)),
    )
    judge = _judge_validator(task, judge_model.strip()) if judge_model and judge_model.strip() else None

    # Which criteria were actually asked for, by name. The ledger records this so a shape's
    # success rate can be read against how hard it was to pass: "this model succeeds 90% of the
    # time" means something different with no criteria than with require_json + a judge.
    kinds = [
        name for name, requested in (
            ("min_length", int(min_length or 0) > 0),
            ("require_substrings", bool(substrings)),
            ("require_regex", bool(require_regex)),
            ("require_json", bool(require_json)),
            ("require_keys", bool(keys)),
            ("require_citations", bool(require_citations)),
            ("judge_model", judge is not None),
        ) if requested
    ]

    if deterministic is None and judge is None:
        # Nothing requested: the caller gets {} and _run_subagent_task's validator_kinds default
        # (None) stands, which the ledger records as an empty string. A criteria-free delegation
        # is a real and useful data point, not a hole.
        return {}

    if deterministic is not None and judge is not None:
        def combined(text: str) -> tuple[bool, str]:
            ok, reason = deterministic(text)
            if not ok:
                return ok, reason
            return judge(text)

        validator: Validator = combined
    else:
        validator = deterministic or judge  # type: ignore[assignment]

    escalation = [m.strip() for m in escalate_models.split(",") if m.strip()] if escalate_models else None
    return {
        "validator": validator,
        "max_attempts": int(max_attempts) if max_attempts else None,
        "escalation_models": escalation,
        # Splatted into _run_subagent_task at every call site, so this reaches all four entry
        # points without any of them being edited.
        "validator_kinds": kinds,
    }


# =============================================================================
# Public tools
# =============================================================================

@tool
def spawn_subagent(
    task: str,
    model_name: str = "",
    max_history_chars: int = 20000,
    min_length: int = 0,
    require_substrings_json: str = "",
    require_regex: str = "",
    require_json: bool = False,
    require_keys_json: str = "",
    judge_model: str = "",
    max_attempts: int = 0,
    escalate_models: str = "",
    session_name: str = "",
    max_result_chars: int = 6000,
    require_citations: bool = False,
    citation_min: int = 1,
) -> str:
    """Delegate a self-contained sub-task to a fresh isolated agent sub-run that shares this
    workspace and all tools, but starts with empty history. Use for large, self-contained
    sub-tasks (a multi-file refactor, an isolated investigation) so intermediate tool calls don't
    flood the main conversation. The sub-agent does not see this conversation - give it a precise,
    self-contained task description. Returns the sub-agent's final answer text.

    Result-quality loop (optional): the LiteLLM proxy already retries on a non-200 transport
    error, but not on a 200 whose CONTENT is wrong or incomplete. Supply any acceptance
    criteria to make this tool re-run the sub-task (feeding the rejection reason back to the
    model, escalating models per attempt) until the answer passes or the attempt budget runs
    out:
      - min_length: reject an answer shorter than this many characters (catches truncated/empty).
      - require_substrings_json: JSON list of strings that MUST all appear in the answer.
      - require_regex: a regex the answer must match (re.I|re.S).
      - require_json / require_keys_json: require the answer to parse as JSON; require_keys_json
        is a JSON list of top-level keys that must be present (implies require_json).
      - require_citations: reject an answer that does not cite at least citation_min real
        workspace files as path:line (the file must exist and the line must be inside it).
        Checked against the filesystem here, so a fabricated or drifted reference cannot reach
        you; URLs, ports, timestamps and prose colons are ignored, not treated as citations.
      - citation_min: how many distinct verified citations require_citations demands (default 1).
      - judge_model: run a second model as a strict acceptance judge for semantic correctness
        (e.g. "verifier"). Applied only after the deterministic checks pass.
      - max_attempts: cap the retries (default core.SUBAGENT_MAX_ATTEMPTS).
      - escalate_models: comma-separated models to escalate through on successive attempts
        (e.g. "verifier,agent-orchestrator"); defaults to core.SUBAGENT_ESCALATION_MODELS.
    With no criteria, behaves exactly as before (single attempt). On final failure it returns
    ok=false with the best result and a per-attempt validation_history.

    Continuable delegation (optional): pass session_name to persist this sub-agent's full
    history (messages AND tool calls/results). The result then carries a session_id, and
    resume_subagent(session_id, message) continues THAT sub-agent with everything it already
    did still in context. Use it whenever the sub-task may need a decision only you can make:
    tell the sub-agent to reply "BLOCKED: <question>" instead of guessing, answer the question,
    and resume - instead of re-delegating from scratch and paying for the work twice. Without
    session_name nothing is persisted and the run is exactly as cheap as before.

    Result size: an answer longer than max_result_chars (default 6000) is written to a file and
    the reply carries only a head plus result_path/result_bytes/result_sha256 - nothing is lost,
    and you read the slice you need from that path. Set max_result_chars=0 to disable spilling.

    Completeness: every reply carries status (complete/partial/failed) plus coverage, both stamped
    by the harness from the raw tool results, never by the sub-agent. Read a clean status as NO
    TOOL REPORTED A CAP - not as proof the run was exhaustive; some tools truncate silently. A
    partial status means coverage.signals names what was capped, truncated or skipped, and you
    should treat the answer as drawn from incomplete evidence."""
    if core.SUBAGENT_DEPTH >= core.MAX_SUBAGENT_DEPTH:
        return core.to_json({
            "ok": False,
            "error": f"Sub-agent depth limit reached ({core.MAX_SUBAGENT_DEPTH}); refusing to spawn another sub-agent to avoid runaway recursion.",
        })

    core.SUBAGENT_DEPTH += 1
    try:
        kwargs = _validation_kwargs(
            min_length, require_substrings_json, require_regex, require_json,
            require_keys_json, judge_model, task, max_attempts, escalate_models,
            require_citations=require_citations, citation_min=citation_min,
        )
        result = _run_subagent_task(
            task, model_name, max_history_chars,
            session_name=session_name,
            spill_over_chars=max(0, int(max_result_chars or 0)),
            entry_point="spawn_subagent",
            **kwargs,
        )
        return core.to_json(result)
    finally:
        core.SUBAGENT_DEPTH -= 1


@tool
def resume_subagent(
    session_id: str,
    message: str,
    model_name: str = "",
    max_history_chars: int = 20000,
    min_length: int = 0,
    require_substrings_json: str = "",
    require_regex: str = "",
    require_json: bool = False,
    require_keys_json: str = "",
    judge_model: str = "",
    max_attempts: int = 0,
    escalate_models: str = "",
    require_citations: bool = False,
    citation_min: int = 1,
) -> str:
    """Continue a previously persisted sub-agent conversation, with its FULL prior history
    (its earlier reasoning, tool calls, and tool results) still in context. Requires a
    session_id returned by a spawn_subagent call that was given a session_name.

    This is the answer to a sub-agent that stopped and asked for input: it reported
    "BLOCKED: which collection?", you supply the answer here as message, and it picks up
    exactly where it stopped rather than redoing the work. Also use it to send follow-up
    work to a sub-agent that already paid to explore some code or a website - continuing is
    far cheaper than re-establishing that context in a fresh delegation.

    Note that only the CONVERSATION is restored. Live process state from the earlier run -
    open browser instances, background processes - is gone, so a resumed sub-agent must
    reopen anything it needs. It keeps the session's original model unless model_name
    overrides it, and it supports the same acceptance criteria as spawn_subagent. The
    session is re-saved after this turn, so you can resume repeatedly. The reply carries the same
    harness-stamped status/coverage pair as spawn_subagent."""
    if core.SUBAGENT_DEPTH >= core.MAX_SUBAGENT_DEPTH:
        return core.to_json({
            "ok": False,
            "error": f"Sub-agent depth limit reached ({core.MAX_SUBAGENT_DEPTH}); refusing to spawn another sub-agent to avoid runaway recursion.",
        })

    session_id = (session_id or "").strip()
    if not session_id:
        return core.to_json({"ok": False, "error": "session_id is required"})
    if not (message or "").strip():
        return core.to_json({"ok": False, "error": "message is required"})

    try:
        session, raw_history = sessions.SessionStore().load(session_id)
    except FileNotFoundError:
        return core.to_json({
            "ok": False,
            "error": f"No such sub-agent session: {session_id}. Use list_subagent_sessions to see live ones.",
        })
    except Exception as exc:
        return core.to_json({"ok": False, "error": f"Could not load session {session_id}: {exc!r}"})

    core.SUBAGENT_DEPTH += 1
    try:
        kwargs = _validation_kwargs(
            min_length, require_substrings_json, require_regex, require_json,
            require_keys_json, judge_model, message, max_attempts, escalate_models,
            require_citations=require_citations, citation_min=citation_min,
        )
        result = _run_subagent_task(
            message,
            model_name or session.model_name,
            max_history_chars,
            restore_history=raw_history,
            session_id=session_id,
            entry_point="resume_subagent",
            **kwargs,
        )
        result["resumed_session"] = {"id": session_id, "name": session.name, "turns_restored": len(raw_history)}
        return core.to_json(result)
    finally:
        core.SUBAGENT_DEPTH -= 1


@tool
def list_subagent_sessions(limit: int = 20) -> str:
    """List persisted sub-agent sessions (most recently updated first) that resume_subagent
    can continue: id, name, model, workspace, and timestamps. Use it when you know a
    sub-agent explored something earlier but you no longer have its session_id to hand."""
    try:
        found = sessions.SessionStore().list()
        limit = max(1, int(limit or 20))
        return core.to_json({
            "ok": True,
            "count": len(found),
            "sessions": [s.to_dict() for s in found[:limit]],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def spawn_map_reduce(
    glob_pattern: str,
    map_prompt: str,
    reduce_prompt: str = "",
    model_name: str = "",
    reduce_model: str = "",
    max_files: int = 200,
    max_result_chars: int = 1500,
    include_content: bool = False,
    max_content_chars: int = 12000,
    require_regex: str = "",
    require_json: bool = False,
    require_keys_json: str = "",
    min_length: int = 1,
    max_attempts: int = 0,
    offset: int = 0,
    require_citations: bool = False,
    citation_min: int = 1,
) -> str:
    """Run one sub-agent per file matching a glob, then optionally fold every answer into a
    single synthesis - without the file list ever entering your context.

    This is the fix for the usual fan-out leak: spawn_subagents_parallel needs you to BUILD the
    task array, so you must first list 400 bundle paths into your own window just to delegate
    over them. Here the harness expands the glob itself; you send a pattern and a template and
    get back one answer.

    map_prompt is a template run once per file. Placeholders: {path} (workspace-relative) and,
    when include_content=true, {content} (the file's text, truncated to max_content_chars). If
    you omit {content} the sub-agent simply reads the file itself, which is usually better - it
    can search rather than swallow the whole thing.

    reduce_prompt, if given, runs a final sub-agent over all per-file answers and its synthesis
    becomes the headline result (per-file answers are still returned, bounded). Without it you
    get the per-file answers only.

    Cost control: max_files caps the fan-out (default 200; files beyond it are reported as
    skipped, never silently dropped), and each map answer is capped at max_result_chars, so a
    200-file sweep cannot flood you. Ideal for bundle mining, mass classification, per-file
    audits.

    Quality: nobody reads 200 per-file answers, so bad ones must be caught mechanically. An
    empty answer is rejected by default (min_length=1) and retried on the next escalation model
    - some router models return a clean, empty, entirely useless answer. Add require_regex or
    require_keys_json when the answers have a known shape; each file is validated and retried
    independently. require_citations (with citation_min) additionally rejects a per-file answer
    that does not cite at least that many real workspace files as path:line - the path must
    exist and the line must be inside the file, checked here rather than taken on trust.

    Coverage and resume: the reply carries status/coverage computed by the harness from its OWN
    glob expansion (covered/total/gaps are counted here, never taken from a model's answer). If
    files were left over, resume_cursor tells you how to finish the sweep - call again with the
    same glob_pattern and max_files and the offset it hands back, and the next batch is picked
    up in the same sorted order. offset skips that many matched files from the start."""
    try:
        if not (glob_pattern or "").strip():
            return core.to_json({"ok": False, "error": "glob_pattern is required"})
        if not (map_prompt or "").strip():
            return core.to_json({"ok": False, "error": "map_prompt is required"})
        if core.SUBAGENT_DEPTH >= core.MAX_SUBAGENT_DEPTH:
            return core.to_json({
                "ok": False,
                "error": f"Sub-agent depth limit reached ({core.MAX_SUBAGENT_DEPTH}).",
            })

        max_files = max(1, min(int(max_files or 200), 1000))
        pattern = glob_pattern.strip().lstrip("/\\")

        matched: list[str] = []
        for p in core.WORKSPACE.glob(pattern):
            try:
                rp = p.resolve(strict=False)
                rp.relative_to(core.WORKSPACE)
            except Exception:
                continue
            if rp.is_file():
                matched.append(core.rel_path(rp))

        if not matched:
            return core.to_json({
                "ok": False, "error": f"No files matched {glob_pattern!r} under {core.WORKSPACE}",
            })

        # This sort is what makes the offset cursor meaningful: a resumed call re-globs from
        # scratch, so only a deterministic total order guarantees offset N lands on the file
        # after the last one processed. Change the ordering here and resume_cursor silently
        # starts re-mapping some files while skipping others.
        matched.sort()
        offset = max(0, int(offset or 0))
        selected = matched[offset:offset + max_files]
        gaps = matched[offset + len(selected):]
        # Files before the offset were covered by the earlier call in the cursor chain; only the
        # tail is genuinely unexamined by anyone, and that is what "skipped" must mean.
        skipped = len(gaps)

        if not selected:
            # Only reachable via an offset past the end - i.e. a cursor chain that is already
            # finished. Say so plainly instead of dying in ThreadPoolExecutor(max_workers=0).
            return core.to_json({
                "ok": False,
                "error": (
                    f"offset={offset} is past the {len(matched)} files matching {glob_pattern!r}; "
                    "the sweep is already complete."
                ),
                "files_matched": len(matched),
            })

        def _task_for(rel: str) -> str:
            body = map_prompt.replace("{path}", rel)
            if include_content and "{content}" in body:
                try:
                    text = (core.WORKSPACE / rel).read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    text = f"<could not read file: {exc!r}>"
                body = body.replace("{content}", core.trim_text(text, max_content_chars))
            return f"{body}\n\n(The file in question is: {rel})"

        core.SUBAGENT_DEPTH += 1
        try:
            worker_count = min(len(selected), max(1, (os.cpu_count() or 1) - 2))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = [
                    pool.submit(
                        _run_subagent_task, _task_for(rel), model_name,
                        max(200, int(max_result_chars or 1500)),
                        spill_over_chars=max(200, int(max_result_chars or 1500)),
                        glob_involved=True,
                        entry_point="map_reduce",
                        # Per-file answers are the least supervised output in the whole harness
                        # (nobody eyeballs 200 of them), so acceptance criteria matter MORE here
                        # than on a single delegation - a malformed answer is otherwise silently
                        # folded into the reduce step and reported as a clean result.
                        **_validation_kwargs(
                            min_length, "", require_regex, require_json,
                            require_keys_json, "", _task_for(rel), max_attempts, "",
                            require_citations=require_citations, citation_min=citation_min,
                        ),
                    )
                    for rel in selected
                ]
                map_results = [f.result() for f in futures]
        finally:
            core.SUBAGENT_DEPTH -= 1

        per_file = [
            {"path": rel, **_compact_result(r)} for rel, r in zip(selected, map_results)
        ]
        # A per-file run that came back partial (its sub-agent hit a capped tool) makes this sweep
        # partial too, otherwise a clean top-level status would launder every cap one level down.
        # These accumulate through the reduce phase and are stamped into the envelope at the end,
        # once every stage that can lose evidence has actually run.
        map_signals: list[dict[str, Any]] = []
        if skipped:
            map_signals.append({
                "tool": "spawn_map_reduce",
                "reason": "files_skipped_over_cap",
                "returned": len(selected),
                "total": len(matched),
            })
        partial_runs = sum(1 for r in map_results if r.get("status") == completeness.STATUS_PARTIAL)
        if partial_runs:
            map_signals.append({
                "tool": "map_file",
                "reason": "sub_run_partial",
                "returned": len(selected) - partial_runs,
                "total": len(selected),
            })
        # A per-file run that FAILED contributes no status signal of its own (its status is
        # "failed", not partial), so without this a sweep whose every sub-agent exhausted its
        # attempts still reported a clean verdict. Nobody eyeballs 200 per-file answers, which is
        # exactly why the count has to reach the top-level envelope.
        map_failed = sum(1 for r in map_results if not r.get("ok"))
        if map_failed:
            map_signals.append({
                "tool": "map_file",
                "reason": "sub_run_failed",
                "returned": len(selected) - map_failed,
                "total": len(selected),
            })

        reduced: dict[str, Any] | None = None
        if (reduce_prompt or "").strip():
            # The synthesis step must be told what it is NOT seeing, or it will happily write
            # "across all matching files..." over a partial sample.
            coverage_line = (
                f"({len(selected)} of {len(matched)} matching files; {skipped} were NOT examined "
                f"and are absent from this transcript)" if skipped
                else f"(all {len(matched)} matching files)"
            )
            transcript = "\n\n".join(
                f"### {r['path']}\n{r.get('result') or r.get('error') or '(no answer)'}"
                for r in per_file
            )
            reduced = _run_subagent_task(
                f"{reduce_prompt}\n\nPER-FILE RESULTS TO SYNTHESIZE "
                f"{coverage_line}:\n{transcript}",
                reduce_model or model_name,
                8000,
                spill_over_chars=8000,
                entry_point="map_reduce_reduce",
            )
            # The reduce answer IS the headline result - it is what workflows.py reads as the step
            # output and what map_reduce exists to produce - so its own verdict has to reach the
            # top-level envelope. A reduce that failed, or that answered off a capped grep, must
            # not ship under a clean sweep-level status just because the map phase went fine.
            map_signals.extend((reduced.get("coverage") or {}).get("signals") or [])
            if not reduced.get("ok"):
                map_signals.append({
                    "tool": "map_reduce_reduce", "reason": "reduce_failed",
                    "returned": 0, "total": 1,
                })

        # Key order is load-bearing: core.to_json trims the whole reply at MAX_OUTPUT_BYTES
        # (256 KB), and a 200-file sweep at the default 1500-char answers overshoots that - so
        # anything after "results" is deleted by the trim exactly when the sweep is big enough to
        # have skipped something. status/coverage/resume_cursor therefore go BEFORE the big list,
        # the same way spawn_subagents_parallel splices its envelope in below.
        # covered/total/gaps come from the harness's OWN glob expansion above, never from a
        # model's answer - that is the whole point of the field.
        out: dict[str, Any] = {
            "ok": True,
            "files_matched": len(matched),
            "files_mapped": len(selected),
            "files_skipped_over_cap": skipped,
            "map_failed": map_failed,
            **completeness.envelope(
                True, map_signals, covered=len(selected), total=len(matched),
                # A SAMPLE, not the list. The full set of unexamined files is not information the
                # caller has to hold: gaps_total says how many, and resume_cursor says how to get
                # them - re-globbing at that offset reproduces them exactly, in the same order.
                # At the 200 this used to inline, one corpus sweep spent ~10k characters of the
                # caller's context restating what a two-field cursor already encodes, which is
                # the leak this whole tool exists to avoid.
                gaps=gaps[:_MAP_GAPS_INLINE],
            ),
        }
        if gaps:
            out["gaps_total"] = len(gaps)
            out["resume_cursor"] = {
                "glob_pattern": glob_pattern,
                "offset": offset + len(selected),
                "max_files": max_files,
                "order": "path_asc",
            }
        if skipped:
            out["note"] = (
                f"{len(matched)} files matched but only {len(selected)} were processed "
                f"(max_files={max_files}, offset={offset}). {skipped} were NOT examined - "
                f"re-call with offset={offset + len(selected)} to continue."
            )
        if reduced is not None:
            out["reduced"] = _compact_result(reduced)
        # _compact_result/_COMPACT_KEYS filters the PER-FILE entries only; this top-level dict is
        # serialized whole, so the coverage keys above need no allowlist entry.
        out["results"] = per_file

        return core.to_json(out)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def spawn_subagents_parallel(
    tasks_json: str,
    model_name: str = "",
    min_length: int = 0,
    require_substrings_json: str = "",
    require_regex: str = "",
    require_json: bool = False,
    require_keys_json: str = "",
    judge_model: str = "",
    max_attempts: int = 0,
    escalate_models: str = "",
    max_result_chars: int = 4000,
    require_citations: bool = False,
    citation_min: int = 1,
) -> str:
    """Fan out several independent, self-contained sub-tasks to fresh isolated agent sub-runs
    concurrently (thread pool), each with its own empty history sharing this workspace and all
    tools. Use when you have multiple unrelated sub-tasks (e.g. investigate N separate files or
    endpoints) that don't depend on each other's results, instead of spawning them one at a time.
    tasks_json must be a JSON list of non-empty task description strings. Returns results in the
    same order as the input tasks, each paired with its original task text.

    The same optional result-quality criteria as spawn_subagent apply here (min_length,
    require_substrings_json, require_regex, require_json, require_keys_json, judge_model,
    require_citations, citation_min, max_attempts, escalate_models) - each task independently
    runs the validation-retry loop. The criteria are applied uniformly to every task in the batch.

    Result size is bounded BY CONSTRUCTION: each task's answer is capped at max_result_chars
    (default 4000) and anything longer is written to a file, with the reply carrying a head
    plus result_path for you to read selectively. Per-attempt validation detail is dropped for
    tasks that succeeded and kept only for those that failed. N tasks therefore cost roughly
    N*max_result_chars in the worst case, not N*20000.

    Completeness: each task carries its own status/coverage, and the batch-level status is only as
    good as its worst task - one task that ran on capped evidence, or failed, makes the batch
    partial. A clean status means no tool reported a cap, not that the search was exhaustive."""
    try:
        tasks = core.parse_json_arg(tasks_json, [])
        if not isinstance(tasks, list) or not tasks or not all(isinstance(t, str) and t.strip() for t in tasks):
            return core.to_json({
                "ok": False,
                "error": "tasks_json must be a non-empty JSON list of non-empty task strings",
            })

        if core.SUBAGENT_DEPTH >= core.MAX_SUBAGENT_DEPTH:
            return core.to_json({
                "ok": False,
                "error": f"Sub-agent depth limit reached ({core.MAX_SUBAGENT_DEPTH}); refusing to spawn another sub-agent to avoid runaway recursion.",
            })

        core.SUBAGENT_DEPTH += 1
        try:
            worker_count = min(len(tasks), max(1, (os.cpu_count() or 1) - 2))
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                # Each task gets its own validator (the judge closure captures that task's text).
                futures = [
                    pool.submit(
                        _run_subagent_task,
                        t,
                        model_name,
                        max(200, int(max_result_chars or 4000)),
                        spill_over_chars=max(200, int(max_result_chars or 4000)),
                        entry_point="spawn_subagents_parallel",
                        **_validation_kwargs(
                            min_length, require_substrings_json, require_regex, require_json,
                            require_keys_json, judge_model, t, max_attempts, escalate_models,
                            require_citations=require_citations, citation_min=citation_min,
                        ),
                    )
                    for t in tasks
                ]
                results = [f.result() for f in futures]
        finally:
            core.SUBAGENT_DEPTH -= 1

        # A batch is only as complete as its worst task: fold every sub-run's signals up so the
        # top-level verdict cannot read clean while one task ran on capped evidence. covered/
        # total stay None deliberately - every submitted task is always run, so a "N of N" ratio
        # here would be a denominator that means nothing.
        batch_signals: list[dict[str, Any]] = []
        for r in results:
            batch_signals.extend((r.get("coverage") or {}).get("signals") or [])
        failed = sum(1 for r in results if not r.get("ok"))
        if failed:
            batch_signals.append({
                "tool": "spawn_subagents_parallel",
                "reason": "task_failed",
                "returned": len(tasks) - failed,
                "total": len(tasks),
            })

        return core.to_json({
            "ok": True,
            "count": len(tasks),
            "failed": failed,
            **completeness.envelope(True, batch_signals),
            "results": [
                {"task": core.trim_text(t, 300), **_compact_result(r)}
                for t, r in zip(tasks, results)
            ],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
