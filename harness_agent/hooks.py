"""Shell-command lifecycle hooks, mirroring Claude Code's hooks.json shape.

A hook is a shell command the operator registers against a lifecycle event
(SessionStart/UserPromptSubmit/PreToolUse/PostToolUse/Stop); when the event fires, the
JSON payload for that event is piped to the command's stdin, same contract as Claude
Code's own hooks. Only hook_list() is an @tool - hook_add/hook_remove/hook_set_enabled
stay plain functions so creating or editing a hook is human-only (via a slash command
wired elsewhere). Letting the agent register its own hooks would let it silently plant a
command that fires on every future tool call, which is a strictly worse version of the
depth-guard reasoning subagent.py already applies to sub-agent spawning.

Workspace-derived paths are read through ``core.CACHE_DIR`` at the point of use (never
imported by name at module top level) so a runtime ``core.set_workspace()`` switch is
honored - see core.py's module docstring for why.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import core

ALLOWED_EVENTS = ("SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop")
DEFAULT_TIMEOUT_SEC = 30
_OUTPUT_TRIM_CHARS = 4000


def _hooks_path() -> Path:
    return core.CACHE_DIR / "hooks.json"


def _log_path() -> Path:
    return core.CACHE_DIR / "hooks.log"


def load_hooks() -> list[dict[str, Any]]:
    """Read the persisted hook list, tolerating a missing or corrupt file."""
    try:
        data = json.loads(_hooks_path().read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [h for h in data if isinstance(h, dict)]


def save_hooks(hooks: list[dict[str, Any]]) -> None:
    """Persist the hook list, creating core.CACHE_DIR if this is the first write."""
    core.ensure_dirs()
    _hooks_path().write_text(json.dumps(hooks, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_hook_record(event: str, matcher_regex: str, command: str, timeout_sec: int) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex[:12],
        "event": event,
        "matcher_regex": matcher_regex,
        "command": command,
        "timeout_sec": int(timeout_sec) if timeout_sec else DEFAULT_TIMEOUT_SEC,
        "enabled": True,
        "created_at": core.now_str(),
    }


def hook_add(event: str, matcher_regex: str, command: str, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> dict[str, Any]:
    """Human-only: validate event, append a new hook record, persist, and return it."""
    if event not in ALLOWED_EVENTS:
        raise ValueError(f"event must be one of: {', '.join(ALLOWED_EVENTS)}")
    hooks = load_hooks()
    record = _new_hook_record(event, str(matcher_regex or ""), str(command or ""), int(timeout_sec or DEFAULT_TIMEOUT_SEC))
    hooks.append(record)
    save_hooks(hooks)
    return {"ok": True, "hook": record}


def hook_remove(hook_id: str) -> dict[str, Any]:
    """Human-only: remove a hook record by id and persist the remaining list."""
    hooks = load_hooks()
    remaining = [h for h in hooks if h.get("id") != hook_id]
    removed = len(remaining) != len(hooks)
    save_hooks(remaining)
    return {"ok": removed, "hook_id": hook_id, "removed": removed}


def hook_set_enabled(hook_id: str, enabled: bool) -> dict[str, Any]:
    """Human-only: toggle a hook record's enabled flag by id and persist it."""
    hooks = load_hooks()
    found: dict[str, Any] | None = None
    for h in hooks:
        if h.get("id") == hook_id:
            h["enabled"] = bool(enabled)
            found = h
            break
    save_hooks(hooks)
    return {"ok": found is not None, "hook_id": hook_id, "hook": found}


def _log_attempt(event: str, hook_id: str, exit_code: Any) -> None:
    try:
        core.ensure_dirs()
        with _log_path().open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"{core.now_str()} event={event} hook_id={hook_id} exit_code={exit_code}\n")
    except Exception:
        pass


def _run_one_hook(hook: dict[str, Any], event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run a single hook end to end; never lets one hook's failure/timeout escape."""
    hook_id = str(hook.get("id") or "")
    try:
        command = str(hook.get("command") or "")
        timeout_sec = int(hook.get("timeout_sec") or DEFAULT_TIMEOUT_SEC)
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(core.WORKSPACE),
            env=core.command_env(),
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
        exit_code = proc.returncode
        _log_attempt(event, hook_id, exit_code)
        return {
            "hook_id": hook_id,
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "stdout": core.trim_text(proc.stdout or "", _OUTPUT_TRIM_CHARS),
            "stderr": core.trim_text(proc.stderr or "", _OUTPUT_TRIM_CHARS),
        }
    except Exception as e:
        _log_attempt(event, hook_id, "error")
        return {
            "hook_id": hook_id,
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": core.trim_text(repr(e), _OUTPUT_TRIM_CHARS),
        }


def run_hooks(event: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run every enabled hook registered for `event`, in order, piping json.dumps(payload)
    to each command's stdin. For PreToolUse/PostToolUse, a non-empty matcher_regex is
    searched against payload['tool_name'] to decide whether that hook applies.

    A PreToolUse hook whose stdout parses as a JSON object with decision == "block" halts
    processing immediately - later hooks never run once the operator's own hook has vetoed
    the tool call - and the overall result is marked blocked with that object's reason.
    """
    tool_name = str(payload.get("tool_name") or "") if isinstance(payload, dict) else ""
    results: list[dict[str, Any]] = []
    blocked = False
    reason: str | None = None

    for hook in load_hooks():
        if not hook.get("enabled", True):
            continue
        if hook.get("event") != event:
            continue
        matcher = str(hook.get("matcher_regex") or "")
        if event in ("PreToolUse", "PostToolUse") and matcher:
            try:
                if not re.search(matcher, tool_name):
                    continue
            except re.error:
                continue

        outcome = _run_one_hook(hook, event, payload)
        results.append(outcome)

        if event == "PreToolUse":
            try:
                parsed = json.loads(outcome.get("stdout") or "")
            except Exception:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("decision") == "block":
                blocked = True
                reason = str(parsed.get("reason") or "")
                break

    result: dict[str, Any] = {"blocked": blocked, "results": results}
    if blocked:
        result["reason"] = reason
    return result


@tool
def hook_list() -> str:
    """List all configured lifecycle hooks (read-only). Hooks are created, edited, and
    removed only by a human via a slash command - never by the agent - since a hook runs
    an arbitrary shell command on future SessionStart/UserPromptSubmit/PreToolUse/
    PostToolUse/Stop events."""
    try:
        return core.to_json({"ok": True, "hooks": load_hooks()})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
