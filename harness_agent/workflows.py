"""Saved/reusable workflows for the local agent.

A "workflow" is a named, persisted multi-step task description with
``{variable}`` placeholders (Python ``str.format`` syntax). The operator (or
the agent itself) saves one once with :func:`workflow_save`, then replays it
later by name with :func:`workflow_run` instead of re-deriving the same steps
every time.

Workflows are persisted as JSON at ``core.WORKFLOW_CONFIG_PATH`` - a dict
keyed by workflow name - using the same load/save-the-whole-file pattern as
the MCP server registry in main.py (``mcp_load_servers``/``mcp_save_servers``).

This module must NOT import ``agent_runtime`` directly: ``agent_runtime``
depends on the tools in this module (via the tools registry), and importing
it back here would create a cycle. Instead, ``cli.py`` (or whatever owns the
live ``AgentRuntime``) calls :func:`set_runtime_provider` once at startup
with a zero-arg callable that returns the live runtime, and
:func:`workflow_run` calls that callable lazily at execution time.

Every path that is derived from the active workspace is read off ``core`` at
call time (``core.WORKFLOW_CONFIG_PATH``, ``core.WORKSPACE``, ...) rather than
imported by name, so that ``core.set_workspace()`` is honored - see core.py's
module docstring for why.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any, Callable

from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from . import completeness
from . import core

# =============================================================================
# Runtime provider (dependency injection to avoid an import cycle)
# =============================================================================

# Zero-arg callable returning the live AgentRuntime instance, set once by
# cli.py at startup via set_runtime_provider(lambda: runtime).
_RUNTIME_PROVIDER: Any = None


def set_runtime_provider(provider: Callable[[], Any]) -> None:
    global _RUNTIME_PROVIDER
    _RUNTIME_PROVIDER = provider


# =============================================================================
# Registry load/save (same pattern as mcp_load_servers/mcp_save_servers)
# =============================================================================

def workflow_load_all() -> dict[str, dict[str, Any]]:
    core.ensure_dirs()
    if not core.WORKFLOW_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(core.WORKFLOW_CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            return {core.safe_name(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def workflow_save_all(workflows: dict[str, dict[str, Any]]) -> None:
    core.ensure_dirs()
    core.WORKFLOW_CONFIG_PATH.write_text(json.dumps(workflows, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Safe str.format_map wrapper
# =============================================================================

class _SafeFormatDict(dict):
    """dict subclass for str.format_map that raises a clear error only for keys
    actually referenced by the template and not supplied.

    Extra supplied variables that the template never references are simply
    ignored (this is a plain dict otherwise, so str.format_map looks up keys
    normally and only hits __missing__ for keys the template actually uses).
    """

    def __missing__(self, key: str) -> str:
        raise KeyError(key)


def format_template_safe(template: str, variables: dict[str, Any]) -> str:
    """Expand ``template`` (Python str.format syntax) against ``variables``.

    Raises KeyError(missing_key) if the template references a placeholder
    that is not present in ``variables``. Extra keys in ``variables`` that the
    template does not reference are ignored.
    """
    return str(template).format_map(_SafeFormatDict(variables))


# =============================================================================
# Tools
# =============================================================================

@tool
def workflow_save(name: str, prompt_template: str = "", description: str = "", steps_json: str = "") -> str:
    """Save (or update) a named reusable pipeline that can later be replayed with workflow_run.

    Two forms:

    1. Single-shot (prompt_template): one task description with {variable} placeholders.

    2. MULTI-STEP (steps_json) - preferred for anything real. A JSON list of steps, each run as
       its own validated sub-agent, with later steps consuming earlier ones:
         [{"id": "recon", "mode": "agent", "model": "mistral-large",
           "prompt": "Map the endpoints under {path}", "accept": {"min_length": 50}},
          {"id": "audit", "mode": "map", "glob": "src/**/*.js",
           "prompt": "Audit {file} for auth checks. Findings only.",
           "reduce_prompt": "Merge into one ranked list."},
          {"id": "final", "mode": "agent",
           "prompt": "Write a summary from: {steps.recon.result} and {steps.audit.result}"}]

    Step fields: id (required, unique), mode ("agent" default, or "map" for a per-file fan-out
    over glob), prompt (required), model, glob (map only), reduce_prompt (map only), and accept
    (an object of acceptance criteria: min_length, require_regex, require_json, require_keys,
    require_citations, citation_min, max_attempts). require_citations makes a step's answer
    prove itself by citing citation_min real workspace files as path:line, verified against the
    filesystem. Prompts may reference {variables} passed to workflow_run and {steps.<id>.result}
    from any earlier step; a map step's prompt may also use {file}.

    Multi-step workflows run each step through the full delegation machinery - validation, retry,
    and model escalation - and only the LAST step's answer is the headline result, so the
    intermediate stages never cost you context."""
    try:
        key = core.safe_name(name)
        prompt_template = str(prompt_template or "")
        steps_json = str(steps_json or "").strip()

        steps: list[dict[str, Any]] = []
        if steps_json:
            parsed = core.parse_json_arg(steps_json, [])
            if not isinstance(parsed, list) or not parsed:
                return core.to_json({"ok": False, "error": "steps_json must be a non-empty JSON list of step objects"})
            seen_ids: set[str] = set()
            for i, raw in enumerate(parsed):
                if not isinstance(raw, dict):
                    return core.to_json({"ok": False, "error": f"step {i} is not an object"})
                step_id = str(raw.get("id") or f"step{i + 1}").strip()
                if step_id in seen_ids:
                    return core.to_json({"ok": False, "error": f"duplicate step id: {step_id}"})
                seen_ids.add(step_id)
                mode = str(raw.get("mode") or "agent").strip().lower()
                if mode not in ("agent", "map"):
                    return core.to_json({"ok": False, "error": f"step {step_id}: mode must be 'agent' or 'map'"})
                if not str(raw.get("prompt") or "").strip():
                    return core.to_json({"ok": False, "error": f"step {step_id}: prompt is required"})
                if mode == "map" and not str(raw.get("glob") or "").strip():
                    return core.to_json({"ok": False, "error": f"step {step_id}: mode 'map' requires a glob"})
                accept = raw.get("accept") or {}
                if not isinstance(accept, dict):
                    return core.to_json({"ok": False, "error": f"step {step_id}: accept must be an object"})
                steps.append({
                    "id": step_id, "mode": mode, "prompt": str(raw["prompt"]),
                    "model": str(raw.get("model") or ""), "glob": str(raw.get("glob") or ""),
                    "reduce_prompt": str(raw.get("reduce_prompt") or ""),
                    "max_files": int(raw.get("max_files") or 100), "accept": accept,
                })
        elif not prompt_template.strip():
            return core.to_json({"ok": False, "error": "one of prompt_template or steps_json is required"})

        workflows = workflow_load_all()
        existing = workflows.get(key)
        now = core.now_str()
        record = {
            "name": key,
            "prompt_template": prompt_template,
            "steps": steps,
            "description": str(description or ""),
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "last_run_at": existing.get("last_run_at") if existing else None,
            "run_count": existing.get("run_count", 0) if existing else 0,
        }
        workflows[key] = record
        workflow_save_all(workflows)
        return core.to_json({"ok": True, "workflow": record})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def workflow_list() -> str:
    """List every saved workflow (name, description, template, and run stats)."""
    try:
        workflows = workflow_load_all()
        try:
            config_path = core.rel_path(core.WORKFLOW_CONFIG_PATH)
        except ValueError:
            config_path = str(core.WORKFLOW_CONFIG_PATH)
        items = sorted(workflows.values(), key=lambda w: str(w.get("name", "")))
        return core.to_json({"ok": True, "config_path": config_path, "count": len(items), "workflows": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def workflow_get(name: str) -> str:
    """Get a single saved workflow's full record by name."""
    try:
        key = core.safe_name(name)
        workflows = workflow_load_all()
        record = workflows.get(key)
        if record is None:
            return core.to_json({"ok": False, "error": f"Unknown workflow: {name}"})
        return core.to_json({"ok": True, "workflow": record})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def workflow_remove(name: str) -> str:
    """Delete a saved workflow by name."""
    try:
        key = core.safe_name(name)
        workflows = workflow_load_all()
        record = workflows.pop(key, None)
        if record is None:
            return core.to_json({"ok": False, "error": f"Unknown workflow: {name}"})
        workflow_save_all(workflows)
        return core.to_json({"ok": True, "removed": record})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


def _accept_kwargs(accept: dict[str, Any]) -> dict[str, Any]:
    """Translate a step's `accept` object into _run_subagent_task validation kwargs.

    Steps go through the same validator/retry/escalation machinery a direct delegation gets -
    which is precisely what the old single-template workflow bypassed, and why a saved workflow
    used to be strictly weaker than just calling spawn_subagent.
    """
    from . import subagent

    validator = subagent.build_validator(
        min_length=int(accept.get("min_length") or 0),
        require_substrings=list(accept.get("require_substrings") or []),
        require_regex=str(accept.get("require_regex") or ""),
        require_json=bool(accept.get("require_json")),
        require_keys=list(accept.get("require_keys") or []),
        require_citations=bool(accept.get("require_citations")),
        citation_min=max(1, int(accept.get("citation_min") or 1)),
    )
    if validator is None:
        return {}
    # Same criterion names _validation_kwargs reports, so a workflow step is not recorded in the
    # outcome ledger as criteria-free when it in fact ran under a validator.
    kinds = [
        name for name, requested in (
            ("min_length", int(accept.get("min_length") or 0) > 0),
            ("require_substrings", bool(accept.get("require_substrings"))),
            ("require_regex", bool(accept.get("require_regex"))),
            ("require_json", bool(accept.get("require_json"))),
            ("require_keys", bool(accept.get("require_keys"))),
            ("require_citations", bool(accept.get("require_citations"))),
        ) if requested
    ]
    return {
        "validator": validator,
        "max_attempts": int(accept.get("max_attempts") or 0) or None,
        "validator_kinds": kinds,
    }


def _run_steps(record: dict[str, Any], variables: dict[str, Any]) -> dict[str, Any]:
    """Execute a multi-step workflow, threading each step's output into later prompts."""
    from . import subagent

    steps = record.get("steps") or []
    # str.format parses "{steps.count.result}" as the key `steps` followed by ATTRIBUTE access,
    # never as a flat key of that name - so prior results are exposed as nested namespaces
    # rather than dotted dict keys, which silently fails to resolve.
    step_results: dict[str, str] = {}
    trace: list[dict[str, Any]] = []
    final_text = ""
    # A pipeline is only as complete as the evidence it ran on, and workflow_run is the tool whose
    # whole selling point is that the intermediate results never reach the caller - so the caller
    # cannot re-derive this from the trace even in principle. Folding every step's signals up is
    # what lets `outcome["status"]` be a field a caller can key on, the same as every other
    # delegation entry point.
    pipeline_signals: list[dict[str, Any]] = []

    def _scope() -> dict[str, Any]:
        scope: dict[str, Any] = dict(variables)
        scope["steps"] = SimpleNamespace(**{
            sid: SimpleNamespace(result=val) for sid, val in step_results.items()
        })
        return scope

    for step in steps:
        step_id = step["id"]
        # {steps.<id>.result} is exposed as a flat key so str.format never has to resolve
        # attribute access on a nested object.
        #
        # A map step's {file}/{content} are NOT workflow variables - they are filled per file by
        # spawn_map_reduce, one level down. Resolving them to their own literal here lets this
        # pass expand the real variables while leaving the per-file placeholders intact for the
        # next stage (map_reduce's own placeholder is {path}, so {file} maps onto it).
        step_scope = _scope()
        if step["mode"] == "map":
            step_scope.update({"file": "{path}", "path": "{path}", "content": "{content}"})
        try:
            prompt = format_template_safe(step["prompt"], step_scope)
        except (KeyError, AttributeError) as exc:
            missing = exc.args[0] if exc.args else str(exc)
            return {
                "ok": False,
                "error": (
                    f"step '{step_id}' references unknown placeholder '{missing}'. Variables: "
                    f"{sorted(k for k in variables)}; earlier steps available as "
                    f"{sorted('{steps.' + s + '.result}' for s in step_results)}"
                ),
                "steps_completed": len(trace),
                **completeness.envelope(False, pipeline_signals),
                "trace": trace,
            }

        accept = _accept_kwargs(step.get("accept") or {})
        if step["mode"] == "map":
            raw = subagent.spawn_map_reduce.invoke({
                "glob_pattern": step["glob"],
                "map_prompt": prompt,
                "reduce_prompt": format_template_safe(step.get("reduce_prompt") or "", _scope()),
                "model_name": step.get("model") or "",
                "max_files": int(step.get("max_files") or 100),
                "max_result_chars": 1200,
            })
            try:
                payload = json.loads(raw)
            except Exception:
                payload = {"ok": False, "error": "map step returned unparseable output"}
            reduced = (payload.get("reduced") or {}).get("result")
            text = reduced if reduced else json.dumps(
                [{"path": r.get("path"), "result": r.get("result")} for r in payload.get("results", [])],
                ensure_ascii=False,
            )
            ok = bool(payload.get("ok"))
            trace.append({
                "id": step_id, "mode": "map", "ok": ok,
                "files_mapped": payload.get("files_mapped"),
                "files_skipped_over_cap": payload.get("files_skipped_over_cap"),
                # Without this a map step that examined 20 of 400 files still reads as a clean
                # green step in the trace, and the pipeline's final answer inherits that claim.
                "status": payload.get("status"),
                "coverage": payload.get("coverage"),
                "result_preview": core.trim_text(str(text), 400),
            })
            pipeline_signals.extend((payload.get("coverage") or {}).get("signals") or [])
        else:
            result = subagent._run_subagent_task(
                prompt, step.get("model") or "", 8000, spill_over_chars=8000,
                entry_point="workflow", **accept
            )
            text = result.get("result") or ""
            ok = bool(result.get("ok"))
            trace.append({
                "id": step_id, "mode": "agent", "ok": ok, "model": result.get("model"),
                "attempts": result.get("attempts"),
                "status": result.get("status"),
                "result_preview": core.trim_text(str(text), 400),
            })
            pipeline_signals.extend((result.get("coverage") or {}).get("signals") or [])

        if not ok:
            return {
                "ok": False,
                "error": f"step '{step_id}' failed; pipeline stopped so later steps do not build on a bad result",
                "steps_completed": len(trace) - 1,
                **completeness.envelope(False, pipeline_signals),
                "trace": trace,
            }

        step_results[step_id] = str(text)
        final_text = str(text)

    # The envelope goes before "trace"/"result" for the same reason spawn_map_reduce splices its
    # own in early: core.to_json trims the reply at MAX_OUTPUT_BYTES, and a long trace must not be
    # able to push the verdict past the cut.
    return {
        "ok": True,
        "steps_run": len(steps),
        **completeness.envelope(True, pipeline_signals),
        "trace": trace,
        "result": final_text,
    }


@tool
def workflow_run(name: str, variables_json: str = "{}") -> str:
    """Replay a saved workflow by name with a JSON object of variables, e.g.
    {"target": "example.com"}.

    A multi-step workflow (saved with steps_json) runs its whole pipeline inside the harness -
    each step a validated sub-agent, later steps consuming {steps.<id>.result} from earlier ones -
    and returns the FINAL step's answer plus a compact per-step trace. Intermediate results stay
    in the harness, so a five-stage pipeline costs you one result instead of five. A step that
    fails stops the pipeline rather than letting later steps build on a bad result.

    A single-template workflow behaves as before: the expanded text is run as one task.

    Either form returns status (complete/partial/failed) and coverage for the pipeline as a whole,
    stamped by the harness from the raw tool results - a multi-step run is partial if ANY step
    ran on capped evidence, which you could not otherwise tell from the final answer alone."""
    try:
        key = core.safe_name(name)
        workflows = workflow_load_all()
        record = workflows.get(key)
        if record is None:
            return core.to_json({"ok": False, "error": f"Unknown workflow: {name}"})

        variables_pre = core.parse_json_arg(variables_json, {})
        if not isinstance(variables_pre, dict):
            return core.to_json({"ok": False, "error": "variables_json must be a JSON object"})

        if record.get("steps"):
            outcome = _run_steps(record, variables_pre)
            record["run_count"] = int(record.get("run_count", 0)) + 1
            record["last_run_at"] = core.now_str()
            workflows[key] = record
            workflow_save_all(workflows)
            return core.to_json({"workflow": key, **outcome})

        if _RUNTIME_PROVIDER is None:
            return core.to_json({"ok": False, "error": "No runtime provider is set; call set_runtime_provider() at startup before running workflows"})

        variables = core.parse_json_arg(variables_json, {})
        if not isinstance(variables, dict):
            return core.to_json({"ok": False, "error": "variables_json must be a JSON object"})

        prompt_template = str(record.get("prompt_template", ""))
        try:
            expanded_task = format_template_safe(prompt_template, variables)
        except KeyError as exc:
            missing_key = exc.args[0] if exc.args else str(exc)
            return core.to_json({"ok": False, "error": f"Missing required template variable: {missing_key}"})

        runtime = _RUNTIME_PROVIDER()
        if runtime is None:
            return core.to_json({"ok": False, "error": "Runtime provider returned no runtime instance"})

        # This path runs the LIVE shared runtime directly instead of going through
        # _run_subagent_task, so it gets none of that function's exit-point bookkeeping for free.
        # Draining the recorder here is doubly load-bearing: it is the only way this run reports a
        # cap, AND without it the shared runtime's signal list grows for the life of the server
        # and leaks this run's caps into whatever the operator does next.
        from . import ledger, subagent

        started = time.monotonic()
        subagent._drain_signals(runtime, [])  # discard anything left over from an earlier turn
        signals: list[dict[str, Any]] = []
        result = runtime.invoke(expanded_task)
        subagent._drain_signals(runtime, signals)

        answer = "[no final content]"
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                answer = msg.content
                break

        record["run_count"] = int(record.get("run_count", 0)) + 1
        record["last_run_at"] = core.now_str()
        workflows[key] = record
        workflow_save_all(workflows)

        envelope = completeness.envelope(True, signals)
        ledger.record_delegation(
            task=expanded_task, model=str(getattr(runtime, "model_name", "") or ""),
            ok=True, accepted=True, attempts=1,
            latency_ms=int((time.monotonic() - started) * 1000),
            # Read off the envelope just built, never re-derived: one verdict, one deriving site.
            status=str(envelope["status"]),
            entry_point="workflow_template",
        )
        return core.to_json({
            "ok": True,
            "name": key,
            "expanded_task": core.trim_text(expanded_task),
            **envelope,
            "result": answer,
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
