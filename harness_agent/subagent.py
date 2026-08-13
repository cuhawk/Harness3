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
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from langchain_core.tools import tool

from . import core
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


def build_validator(
    *,
    min_length: int = 0,
    require_substrings: list[str] | None = None,
    require_regex: str = "",
    require_json: bool = False,
    require_keys: list[str] | None = None,
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

    if not (min_length or require_substrings or require_regex or want_json):
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


_COMPACT_KEYS = (
    "ok", "model", "attempts", "result", "result_truncated", "result_path",
    "result_bytes", "result_sha256", "session_id", "session_error", "error", "note",
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
) -> dict[str, Any]:
    """Run one isolated sub-agent task, looping until the result passes ``validator`` (or the
    attempt budget is exhausted). Returns a plain dict (both public tools wrap it in their own
    ``core.to_json`` envelope).

    Attempt 1 sends ``task`` fresh. Each rejected attempt re-invokes the SAME runtime with the
    validator's rejection ``reason`` as the next turn, so the model sees its own prior answer
    plus what was wrong ("needs more data" -> it continues; "wrong data" -> it corrects). When
    an escalation model is configured for the next attempt, the runtime is switched to it first
    (history is preserved across the switch, so the stronger model inherits the full context).
    """
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
        except Exception as exc:
            # A raised exception here means the transport-level loop already gave up (or a
            # local error). Record it and retry from scratch if attempts remain.
            history.append({"attempt": attempt, "model": model, "ok": False, "error": repr(exc)})
            feedback = f"the previous attempt failed with an error: {exc!r}"
            runtime = None  # rebuild fresh next attempt
            continue

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
            return out
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
    return out


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
) -> dict[str, Any]:
    """Turn the flat tool arguments into the keyword args ``_run_subagent_task`` expects.

    Combines the deterministic validator (build_validator) with an optional LLM judge into a
    single validator that requires BOTH to pass (cheap checks first, judge only if they hold).
    Returns ``{}`` when nothing was requested, so callers fall straight back to the original
    single-shot behavior.
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
    )
    judge = _judge_validator(task, judge_model.strip()) if judge_model and judge_model.strip() else None

    if deterministic is None and judge is None:
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
    and you read the slice you need from that path. Set max_result_chars=0 to disable spilling."""
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
        )
        result = _run_subagent_task(
            task, model_name, max_history_chars,
            session_name=session_name,
            spill_over_chars=max(0, int(max_result_chars or 0)),
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
    session is re-saved after this turn, so you can resume repeatedly."""
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
        )
        result = _run_subagent_task(
            message,
            model_name or session.model_name,
            max_history_chars,
            restore_history=raw_history,
            session_id=session_id,
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
    independently."""
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

        matched.sort()
        selected = matched[:max_files]
        skipped = len(matched) - len(selected)

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
                        # Per-file answers are the least supervised output in the whole harness
                        # (nobody eyeballs 200 of them), so acceptance criteria matter MORE here
                        # than on a single delegation - a malformed answer is otherwise silently
                        # folded into the reduce step and reported as a clean result.
                        **_validation_kwargs(
                            min_length, "", require_regex, require_json,
                            require_keys_json, "", _task_for(rel), max_attempts, "",
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
        out: dict[str, Any] = {
            "ok": True,
            "files_matched": len(matched),
            "files_mapped": len(selected),
            "files_skipped_over_cap": skipped,
            "map_failed": sum(1 for r in map_results if not r.get("ok")),
            "results": per_file,
        }
        if skipped:
            out["note"] = (
                f"{len(matched)} files matched but only {len(selected)} were processed "
                f"(max_files={max_files}). {skipped} were NOT examined."
            )

        if (reduce_prompt or "").strip():
            transcript = "\n\n".join(
                f"### {r['path']}\n{r.get('result') or r.get('error') or '(no answer)'}"
                for r in per_file
            )
            reduced = _run_subagent_task(
                f"{reduce_prompt}\n\nPER-FILE RESULTS TO SYNTHESIZE "
                f"({len(selected)} of {len(matched)} matching files):\n{transcript}",
                reduce_model or model_name,
                8000,
                spill_over_chars=8000,
            )
            out["reduced"] = _compact_result(reduced)

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
) -> str:
    """Fan out several independent, self-contained sub-tasks to fresh isolated agent sub-runs
    concurrently (thread pool), each with its own empty history sharing this workspace and all
    tools. Use when you have multiple unrelated sub-tasks (e.g. investigate N separate files or
    endpoints) that don't depend on each other's results, instead of spawning them one at a time.
    tasks_json must be a JSON list of non-empty task description strings. Returns results in the
    same order as the input tasks, each paired with its original task text.

    The same optional result-quality criteria as spawn_subagent apply here (min_length,
    require_substrings_json, require_regex, require_json, require_keys_json, judge_model,
    max_attempts, escalate_models) - each task independently runs the validation-retry loop.
    The criteria are applied uniformly to every task in the batch.

    Result size is bounded BY CONSTRUCTION: each task's answer is capped at max_result_chars
    (default 4000) and anything longer is written to a file, with the reply carrying a head
    plus result_path for you to read selectively. Per-attempt validation detail is dropped for
    tasks that succeeded and kept only for those that failed. N tasks therefore cost roughly
    N*max_result_chars in the worst case, not N*20000."""
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
                        **_validation_kwargs(
                            min_length, require_substrings_json, require_regex, require_json,
                            require_keys_json, judge_model, t, max_attempts, escalate_models,
                        ),
                    )
                    for t in tasks
                ]
                results = [f.result() for f in futures]
        finally:
            core.SUBAGENT_DEPTH -= 1

        return core.to_json({
            "ok": True,
            "count": len(tasks),
            "failed": sum(1 for r in results if not r.get("ok")),
            "results": [
                {"task": core.trim_text(t, 300), **_compact_result(r)}
                for t, r in zip(tasks, results)
            ],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
