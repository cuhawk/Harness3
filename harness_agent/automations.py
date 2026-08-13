"""Scheduled automations: run a saved workflow on a repeating interval.

New module for harness_agent (not a port of anything in main.py - main.py has no
automations/workflow section to port from). Lets the operator arm a saved workflow
(see workflows.py) to run on a fixed interval in the background, without a human
re-prompting it each time (periodic recon refresh, nightly checks, etc).

Automation definitions persist as JSON at ``core.AUTOMATIONS_CONFIG_PATH`` (a dict
keyed by automation name), following the same load/save pattern as
mcp_load_servers/mcp_save_servers in main.py's MCP section. Live (in-process)
scheduling is driven by one background daemon thread per active automation; the
``_RUNNING`` registry and its guarding ``_LOCK`` are this module's in-memory view of
which automations currently have a live timer thread.

Every path that is derived from the active workspace (``core.AUTOMATIONS_CONFIG_PATH``)
is read through ``core.`` at the point of use rather than imported by name, so a
runtime workspace switch (``core.set_workspace``) is picked up correctly - see the
module docstring in core.py for why ``from .core import NAME`` would break that.
"""

from __future__ import annotations

import collections
import json
import threading
from typing import Any

from langchain_core.tools import tool

from . import core
from . import permissions
from . import workflows


# =============================================================================
# In-process scheduling state
# =============================================================================

# name -> {"thread": Thread, "stop_event": threading.Event, "logs": collections.deque}
_RUNNING: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


# =============================================================================
# Persisted registry helpers (same load/save pattern as mcp_load_servers /
# mcp_save_servers in main.py's MCP section)
# =============================================================================

def _load_automations() -> dict[str, dict[str, Any]]:
    core.ensure_dirs()
    if not core.AUTOMATIONS_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(core.AUTOMATIONS_CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            return {core.safe_name(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _save_automations(automations: dict[str, dict[str, Any]]) -> None:
    core.ensure_dirs()
    core.AUTOMATIONS_CONFIG_PATH.write_text(json.dumps(automations, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Workflow invocation helpers
# =============================================================================

def _workflow_get(workflow_name: str) -> dict[str, Any]:
    raw = workflows.workflow_get.invoke({"name": workflow_name})
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"ok": False, "error": "Could not parse workflow_get response", "raw": raw}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": "Unexpected workflow_get response shape", "raw": raw}


def _workflow_run(workflow_name: str, variables_json: str) -> tuple[bool, str, Any]:
    """Run a saved workflow once. Returns (ok, raw_result_str, parsed_result)."""
    raw = workflows.workflow_run.invoke({"name": workflow_name, "variables_json": variables_json})
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"raw": raw}
    run_ok = bool(parsed.get("ok", True)) if isinstance(parsed, dict) else True
    return run_ok, raw, parsed


# =============================================================================
# Background thread lifecycle
# =============================================================================

def _record_run_result(name: str, run_ok: bool, error_text: str | None) -> None:
    with _LOCK:
        registry = _load_automations()
        record = registry.get(name)
        if record is None:
            return
        record["last_run_at"] = core.now_str()
        record["last_result_ok"] = run_ok
        record["last_error"] = None if run_ok else error_text
        record["run_count"] = int(record.get("run_count", 0) or 0) + 1
        record["updated_at"] = core.now_str()
        registry[name] = record
        _save_automations(registry)


def _automation_loop(name: str, workflow_name: str, variables_json: str, interval_minutes: float, stop_event: threading.Event, logs: "collections.deque[dict[str, Any]]", run_first: bool) -> None:
    permissions.mark_background()
    interval_sec = max(1.0, float(interval_minutes) * 60.0)
    first_iteration = True
    while not stop_event.is_set():
        if first_iteration and not run_first:
            # Wait-then-run for the very first iteration only (operator asked for no
            # immediate run); every iteration after this is run-then-wait so a
            # cancel() interrupts the wait promptly instead of blocking a full interval.
            first_iteration = False
            if stop_event.wait(timeout=interval_sec):
                break
            continue
        first_iteration = False

        run_ok = True
        error_text: str | None = None
        try:
            run_ok, raw, parsed = _workflow_run(workflow_name, variables_json)
            summary = core.trim_text(raw, 2000)
            if not run_ok:
                error_text = core.trim_text(parsed.get("error", raw) if isinstance(parsed, dict) else raw, 2000)
        except Exception as e:
            run_ok = False
            error_text = repr(e)
            summary = error_text

        logs.append({"time": core.now_str(), "ok": run_ok, "summary": summary})
        _record_run_result(name, run_ok, error_text)

        if stop_event.wait(timeout=interval_sec):
            break


def _stop_running(name: str, join_timeout: float = 5.0) -> bool:
    """Signal and join a live automation thread, if any. Returns whether one was found."""
    with _LOCK:
        info = _RUNNING.pop(name, None)
    if info is None:
        return False
    info["stop_event"].set()
    try:
        info["thread"].join(timeout=join_timeout)
    except Exception:
        pass
    return True


def _start_automation_thread(name: str, workflow_name: str, variables_json: str, interval_minutes: float, run_first: bool) -> None:
    _stop_running(name)
    stop_event = threading.Event()
    logs: "collections.deque[dict[str, Any]]" = collections.deque(maxlen=200)
    thread = threading.Thread(
        target=_automation_loop,
        args=(name, workflow_name, variables_json, interval_minutes, stop_event, logs, run_first),
        name=f"automation-{name}",
        daemon=True,
    )
    with _LOCK:
        _RUNNING[name] = {"thread": thread, "stop_event": stop_event, "logs": logs}
    thread.start()


# =============================================================================
# Tools
# =============================================================================

@tool
def automation_schedule(name: str, workflow_name: str, interval_minutes: float, variables_json: str = "{}", autostart: bool = True, run_immediately: bool = True) -> str:
    """Schedule a saved workflow to run repeatedly on a fixed interval in the background (e.g. a periodic recon refresh) without a human re-prompting it each time."""
    try:
        name = core.safe_name(name)
        workflow_name = str(workflow_name).strip()

        check = _workflow_get(workflow_name)
        if not check.get("ok"):
            return core.to_json({"ok": False, "error": f"Unknown workflow: {workflow_name}", "detail": check})

        interval_minutes = float(interval_minutes)
        if interval_minutes <= 0:
            return core.to_json({"ok": False, "error": "interval_minutes must be > 0"})

        with _LOCK:
            registry = _load_automations()
            existing = registry.get(name, {})
            now = core.now_str()
            record = {
                "name": name,
                "workflow_name": workflow_name,
                "interval_minutes": interval_minutes,
                "variables_json": variables_json,
                "autostart": bool(autostart),
                "created_at": existing.get("created_at", now),
                "updated_at": now,
                "last_run_at": existing.get("last_run_at"),
                "last_result_ok": existing.get("last_result_ok"),
                "last_error": existing.get("last_error"),
                "run_count": int(existing.get("run_count", 0) or 0),
            }
            registry[name] = record
            _save_automations(registry)

        _start_automation_thread(name, workflow_name, variables_json, interval_minutes, run_first=bool(run_immediately))

        return core.to_json({"ok": True, "automation": record})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def automation_list() -> str:
    """List every persisted scheduled automation merged with its live running status."""
    try:
        with _LOCK:
            registry = _load_automations()
            running_snapshot = dict(_RUNNING)

        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for name, record in registry.items():
            info = running_snapshot.get(name)
            entry = dict(record)
            entry["running"] = bool(info is not None and info["thread"].is_alive())
            items.append(entry)
            seen.add(name)

        for name, info in running_snapshot.items():
            if name not in seen:
                items.append({"name": name, "running": info["thread"].is_alive(), "note": "running but not persisted"})

        return core.to_json({"ok": True, "count": len(items), "automations": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def automation_cancel(name: str) -> str:
    """Stop a scheduled automation's background timer and disable its autostart so it does not silently come back on next process start."""
    try:
        name = core.safe_name(name)

        with _LOCK:
            registry = _load_automations()
            persisted = name in registry
        running = name in _RUNNING

        if not persisted and not running:
            return core.to_json({"ok": False, "error": "Unknown automation"})

        stopped = _stop_running(name)

        with _LOCK:
            registry = _load_automations()
            if name in registry:
                registry[name]["autostart"] = False
                registry[name]["updated_at"] = core.now_str()
                _save_automations(registry)

        return core.to_json({"ok": True, "name": name, "stopped_thread": stopped, "autostart": False})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def automation_run_now(name: str) -> str:
    """Run a scheduled automation's workflow once immediately, outside its normal schedule, without disturbing its background timer."""
    try:
        name = core.safe_name(name)

        with _LOCK:
            registry = _load_automations()
            record = registry.get(name)
        if record is None:
            return core.to_json({"ok": False, "error": "Unknown automation"})

        run_ok, raw, parsed = _workflow_run(record["workflow_name"], record.get("variables_json", "{}"))
        error_text = None
        if not run_ok:
            error_text = core.trim_text(parsed.get("error", raw) if isinstance(parsed, dict) else raw, 2000)

        _record_run_result(name, run_ok, error_text)

        info = _RUNNING.get(name)
        if info is not None:
            info["logs"].append({"time": core.now_str(), "ok": run_ok, "summary": core.trim_text(raw, 2000)})

        return core.to_json({"ok": True, "name": name, "run_ok": run_ok, "result": parsed})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def automation_logs(name: str, max_lines: int = 50) -> str:
    """Return recent in-memory run-log entries for a currently-running automation (logs are in-memory only, not persisted to disk beyond last_run_at/last_error on the record)."""
    try:
        name = core.safe_name(name)
        max_lines = max(1, min(int(max_lines), 200))

        with _LOCK:
            info = _RUNNING.get(name)
            logs = list(info["logs"]) if info is not None else []
            is_alive = bool(info is not None and info["thread"].is_alive())

        if info is None:
            return core.to_json({
                "ok": True,
                "name": name,
                "running": False,
                "logs": [],
                "note": "Automation is not currently running; logs are in-memory only and not persisted to disk beyond last_run_at/last_error on the record.",
            })

        return core.to_json({"ok": True, "name": name, "running": is_alive, "logs": logs[-max_lines:]})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


# =============================================================================
# Startup re-arming (not a tool; called once by cli.py at process start)
# =============================================================================

def autostart_all() -> dict[str, Any]:
    """Re-arm the background timer for every persisted automation with autostart truthy that isn't already running."""
    started: list[str] = []
    skipped: list[str] = []

    with _LOCK:
        registry = _load_automations()
        already_running = set(_RUNNING.keys())

    for name, record in registry.items():
        if not record.get("autostart"):
            skipped.append(name)
            continue
        if name in already_running:
            skipped.append(name)
            continue
        _start_automation_thread(
            name,
            record.get("workflow_name", ""),
            record.get("variables_json", "{}"),
            float(record.get("interval_minutes", 60) or 60),
            run_first=False,
        )
        started.append(name)

    return {"started": started, "skipped": skipped}
