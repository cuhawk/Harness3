"""Tool-call permission and confirmation gate.

Kept deliberately separate from tools_registry.py/agent_runtime.py (which a later
integration pass wires this into) so this file has exactly one human-facing escape
hatch: policy mutation (permission_set/permission_add_rule/permission_remove_rule) is
plain-Python, never an @tool. Only permission_list is @tool-decorated - a compromised or
hallucinating model can inspect the current policy but can never grant itself broader
access, since granting always requires a human/CLI call into this module directly.

Uses contextvars rather than threading.local for the background/interactive marker
because a plain threading.Thread target can set a contextvar directly at the top of its
run function and have that value be visible only on that thread's call stack - the same
property threading.local gives - while also being the mechanism asyncio tasks and any
future async workers need (threading.local does not propagate correctly there). Since
nothing here currently uses asyncio, threading.local would also work today, but
contextvars is the forward-compatible choice and costs nothing extra.

Workspace-derived paths are read through ``core.CACHE_DIR`` at the point of use (never
imported by name at module top level) so a runtime ``core.set_workspace()`` switch is
honored - see core.py's module docstring for why.
"""

from __future__ import annotations

import contextvars
import json
import re
import threading
from typing import Any, Callable, Literal

from langchain_core.tools import tool

from . import core

PermissionDecision = Literal["allow", "ask", "deny"]

RULES_FILENAME = "permissions.json"

# Principle: ask by default for anything that executes shell/git/ssh, writes/deletes/
# moves/copies files, starts/stops managed processes or automations, mutates browser page
# state or runs arbitrary JS in a page, calls an MCP tool, sends stdin to a managed
# process, or deletes a memory/RAG record. Everything else (read/list/search/grep local
# state, or pure in-memory/local bookkeeping) defaults to allow.
_ASK_BY_DEFAULT_TOOLS: frozenset[str] = frozenset({
    "write_file",
    "replace_in_file",
    "delete_path",
    "move_path",
    "copy_path",
    "run_terminal",
    "git_command",
    "ssh_run",
    "ssh_upload",
    "ssh_download",
    "mcp_call_tool",
    "mcp_start_server",
    "mcp_stop_server",
    "mcp_add_server",
    "mcp_remove_server",
    "stop_instance",
    "send_instance_stdin",
    "start_process_instance",
    "start_caido_instance",
    "start_playwright_mcp_instance",
    "automation_schedule",
    "automation_cancel",
    "automation_run_now",
    "browser_open",
    "browser_close",
    "browser_goto",
    "browser_click",
    "browser_fill",
    "browser_press_key",
    "browser_evaluate",
    "memory_delete",
    "rag_delete_source",
    "graphql_request",
})


def default_policy_for(tool_name: str) -> PermissionDecision:
    """Hardcoded ask/allow table; see module-level _ASK_BY_DEFAULT_TOOLS for the set."""
    if tool_name in _ASK_BY_DEFAULT_TOOLS:
        return "ask"
    return "allow"


def auto_approved(tool_name: str) -> bool:
    """Whether ``HARNESS_AUTO_APPROVE`` pre-authorizes ``tool_name`` in a non-interactive run.

    An ask-policy tool in a background context has nobody to ask, so the gate auto-DENIES it
    (see PermissionGate.request). That is the right default for a scheduled automation, but it
    silently guts the MCP surface: a sub-agent delegated a real task there can read but never
    write, run a command, or drive a browser, and reports back that "the permission policy is
    preventing all write operations" instead of doing the work. This env var is the operator's
    explicit, opt-in standing consent for exactly that case.

      HARNESS_AUTO_APPROVE=all            every ask-policy tool
      HARNESS_AUTO_APPROVE=write_file,... only these tool names

    Read at call time, not import time, so a host can set it per server process. It only ever
    upgrades "ask" -> allowed: an explicit "deny" in the persisted rules is checked first in
    request() and still wins, so a deny rule cannot be overridden from the environment.
    """
    import os

    raw = (os.environ.get("HARNESS_AUTO_APPROVE") or "").strip()
    if not raw:
        return False
    if raw.lower() in ("all", "*", "1", "true", "yes"):
        return True
    return tool_name in {part.strip() for part in raw.split(",") if part.strip()}


def load_rules() -> dict[str, Any]:
    """Load persisted policy overrides/patterns, tolerating a missing or corrupt file."""
    path = core.CACHE_DIR / RULES_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def save_rules(rules: dict[str, Any]) -> None:
    core.ensure_dirs()
    path = core.CACHE_DIR / RULES_FILENAME
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def _primary_text_arg(tool_name: str, args: dict[str, Any]) -> str:
    if tool_name in ("run_terminal", "ssh_run"):
        value = args.get("command")
        if value is not None:
            return str(value)
    if tool_name == "git_command":
        value = args.get("args")
        if value is not None:
            return str(value)
    value = args.get("query")
    if value is not None:
        return str(value)
    return str(args)


def evaluate(tool_name: str, args: dict[str, Any]) -> PermissionDecision:
    """Persisted override (policy, else first matching pattern) then hardcoded default."""
    rules = load_rules()
    entry = rules.get(tool_name)
    if isinstance(entry, dict):
        policy = entry.get("policy")
        if policy in ("allow", "ask", "deny"):
            return policy  # type: ignore[return-value]

        patterns = entry.get("patterns")
        if isinstance(patterns, list):
            text = _primary_text_arg(tool_name, args)
            for pattern_entry in patterns:
                if not isinstance(pattern_entry, dict):
                    continue
                regex = pattern_entry.get("regex")
                pattern_policy = pattern_entry.get("policy")
                if not regex or pattern_policy not in ("allow", "ask", "deny"):
                    continue
                try:
                    if re.search(str(regex), text):
                        return pattern_policy  # type: ignore[return-value]
                except re.error:
                    continue

    return default_policy_for(tool_name)


# =============================================================================
# Background/interactive context marking
# =============================================================================

_permission_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "permission context", default="interactive"
)


def mark_background() -> None:
    """Call once at the very start of a background thread's target function."""
    _permission_context.set("background")


def is_background() -> bool:
    return _permission_context.get() == "background"


# =============================================================================
# Confirmation gate
# =============================================================================

ConfirmCallback = Callable[[str, dict[str, Any]], str]


class PermissionGate:
    """Serializes concurrent confirmation prompts behind a single lock so parallel
    sub-agents hitting ask-policy tools at the same time do not overlap prompts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.confirm_callback: ConfirmCallback | None = None

    def set_confirm_callback(self, fn: ConfirmCallback | None) -> None:
        self.confirm_callback = fn

    def request(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, str]:
        decision = evaluate(tool_name, args)

        if decision == "deny":
            return False, "denied by policy"
        if decision == "allow":
            return True, "allowed by policy"

        # decision == "ask"
        if is_background() or self.confirm_callback is None:
            if auto_approved(tool_name):
                return True, "auto-approved: pre-authorized via HARNESS_AUTO_APPROVE"
            return False, (
                "ask-policy tool has no interactive confirmation available in this "
                "non-interactive context and was auto-denied for safety"
            )

        with self._lock:
            result = self.confirm_callback(tool_name, args)

        if result == "always_allow":
            _persist_policy(tool_name, "allow")
            return True, "always_allow: policy persisted as allow"
        if result == "always_deny":
            _persist_policy(tool_name, "deny")
            return False, "always_deny: policy persisted as deny"
        if result == "allow":
            return True, "allowed once by confirmation"
        return False, "denied once by confirmation"


def _persist_policy(tool_name: str, policy: PermissionDecision) -> None:
    rules = load_rules()
    entry = rules.get(tool_name)
    if not isinstance(entry, dict):
        entry = {}
    entry["policy"] = policy
    entry.setdefault("patterns", entry.get("patterns", []))
    rules[tool_name] = entry
    save_rules(rules)


GATE = PermissionGate()


def set_confirm_callback(fn: ConfirmCallback | None) -> None:
    GATE.set_confirm_callback(fn)


# =============================================================================
# Agent-callable tool (read-only - policy mutation stays human-only, see module
# docstring)
# =============================================================================

@tool
def permission_list() -> str:
    """List the full persisted tool-call permission rules (policy overrides and
    regex patterns per tool name). Read-only - mutating policy requires a human/CLI
    call (permission_set/permission_add_rule/permission_remove_rule), never a tool."""
    try:
        return core.to_json({"ok": True, "rules": load_rules()})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


# =============================================================================
# Human/CLI-only policy mutation (plain functions, plain dict returns - never
# agent-callable)
# =============================================================================

def permission_set(tool_name: str, policy: str) -> dict[str, Any]:
    if policy not in ("allow", "ask", "deny"):
        return {"ok": False, "error": "policy must be one of: allow, ask, deny"}
    rules = load_rules()
    entry = rules.get(tool_name)
    if not isinstance(entry, dict):
        entry = {}
    entry["policy"] = policy
    entry.setdefault("patterns", entry.get("patterns", []))
    rules[tool_name] = entry
    save_rules(rules)
    return {"ok": True, "tool_name": tool_name, "policy": policy}


def permission_add_rule(tool_name: str, regex: str, policy: str) -> dict[str, Any]:
    if policy not in ("allow", "ask", "deny"):
        return {"ok": False, "error": "policy must be one of: allow, ask, deny"}
    rules = load_rules()
    entry = rules.get(tool_name)
    if not isinstance(entry, dict):
        entry = {}
    patterns = entry.get("patterns")
    if not isinstance(patterns, list):
        patterns = []
    patterns.append({"regex": regex, "policy": policy})
    entry["patterns"] = patterns
    rules[tool_name] = entry
    save_rules(rules)
    return {"ok": True, "tool_name": tool_name, "regex": regex, "policy": policy}


def permission_remove_rule(tool_name: str, regex: str) -> dict[str, Any]:
    rules = load_rules()
    entry = rules.get(tool_name)
    if not isinstance(entry, dict):
        return {"ok": False, "error": "Unknown tool_name", "tool_name": tool_name}
    patterns = entry.get("patterns")
    if not isinstance(patterns, list):
        return {"ok": False, "error": "No patterns for tool_name", "tool_name": tool_name}
    remaining = [p for p in patterns if not (isinstance(p, dict) and p.get("regex") == regex)]
    removed = len(remaining) != len(patterns)
    entry["patterns"] = remaining
    rules[tool_name] = entry
    save_rules(rules)
    return {"ok": True, "tool_name": tool_name, "regex": regex, "removed": removed}
