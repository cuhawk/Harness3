"""MCP server: exposes every tool in ``tools_registry.TOOLS`` to any MCP client
(Claude Code, another agent, anything speaking MCP) over stdio.

Hand-rolled rather than built on the installed ``mcp`` package's ``Server`` class: that
package here is a 2.0.0a3 pre-release with a callback-injection API that differs
substantially from the documented stable SDK and may still change under us. The wire
protocol itself (JSON-RPC 2.0 messages, one per line, newline-delimited stdio - the
framing the official MCP stdio transport spec actually uses, and what real clients like
Claude Code send) is simple enough that a small dependency-free implementation is more
robust here than pinning to alpha SDK internals for something this central.

Run directly: ``python mcp_server.py`` (thin launcher at the repo root), or
``python -m harness_agent.mcp_server``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import core
from . import permissions
from . import tools_registry
from . import workflows
from .tools_registry import TOOLS, TOOL_BY_NAME

PROTOCOL_VERSION = "2024-11-05"

# Concurrent tools/call executions. Each one may itself fan out sub-agents, so this is a
# ceiling on outstanding client calls, not on total worker threads.
_MAX_CONCURRENT_CALLS = max(2, int(os.environ.get("HARNESS_MCP_MAX_CONCURRENT_CALLS", "6")))


# =============================================================================
# Exposed tool surface
# =============================================================================
# The full registry is ~92 tools, and an MCP client pays for every one of them in its
# system prompt on every turn (~10.7k tokens here) whether it calls them or not. A host
# like Claude Code that already has its own filesystem/terminal/browser tools wants only
# the capabilities it does NOT have - delegation, cross-model verification, workflows, the
# RAG corpus - so let the host narrow the surface from the environment. Defaults to the
# complete registry, so an unconfigured server behaves exactly as before.
#
#   HARNESS_MCP_TOOL_GROUPS=delegate      a named preset (below), or comma-separated group
#                                         names from GROUPS, or "all"
#   HARNESS_MCP_TOOLS=spawn_subagent,...  explicit tool names; wins over the groups var
#
# Filtering applies to tools/list AND tools/call, so the advertised surface is the real
# one. It does not touch what a *sub-agent* can use internally - sub-agents go through
# tools_registry directly and always have the full catalog.

GROUPS: dict[str, list[Any]] = {
    "filesystem": tools_registry.FILESYSTEM_TOOLS,
    "memory": tools_registry.MEMORY_TOOLS,
    "rag": tools_registry.RAG_TOOLS,
    "terminal": tools_registry.TERMINAL_GIT_TOOLS,
    "web": tools_registry.WEB_TOOLS,
    "graphql": tools_registry.GRAPHQL_TOOLS,
    "process": tools_registry.PROCESS_TOOLS,
    "mcp": tools_registry.MCP_TOOLS,
    "delegation": tools_registry.DELEGATION_TOOLS,
    "workflow": tools_registry.WORKFLOW_TOOLS,
    "automation": tools_registry.AUTOMATION_TOOLS,
    "policy": tools_registry.POLICY_TOOLS,
    "task": tools_registry.TASK_TOOLS,
    "browser": tools_registry.BROWSER_TOOLS,
    "ssh": tools_registry.SSH_TOOLS,
    "verify": tools_registry.VERIFY_TOOLS,
    "extract": tools_registry.EXTRACT_TOOLS,
    "dossier": tools_registry.DOSSIER_TOOLS,
    "briefing": tools_registry.BRIEFING_TOOLS,
}

# Named presets. "delegate" is the one built for a coding-agent host: hand work off, verify
# it across models, replay saved workflows, park bulk text in RAG instead of the host's
# context - plus the two workspace tools, since every fs path a sub-agent touches is
# resolved against the harness workspace and the host needs to see and retarget it.
PRESETS: dict[str, list[str]] = {
    "delegate": [
        "spawn_subagent", "spawn_subagents_parallel", "spawn_map_reduce",
        "resume_subagent", "list_subagent_sessions",
        "compare_models", "verify_finding",
        "spawn_subagent_async", "spawn_subagents_parallel_async", "spawn_map_reduce_async",
        "check_job", "wait_for_jobs", "list_jobs",
        "set_briefing", "get_briefing",
        "extract_findings", "findings_search", "findings_stats",
        "dossier_note", "dossier_get", "dossier_set_active", "dossier_targets",
        "workflow_save", "workflow_list", "workflow_get", "workflow_run",
        "rag_ingest_path", "rag_search", "rag_read_chunk", "rag_collection_stats",
        "current_workspace", "switch_workspace",
    ],
    # For sessions that will not fan out: delegation and verification only, nothing else.
    "delegate-lite": [
        "spawn_subagent", "spawn_subagents_parallel", "spawn_map_reduce",
        "resume_subagent", "wait_for_jobs", "spawn_subagent_async", "check_job",
        "current_workspace",
    ],
}


def _selected_tools() -> list[Any]:
    names = [n.strip() for n in (os.environ.get("HARNESS_MCP_TOOLS") or "").split(",") if n.strip()]
    if not names:
        raw = (os.environ.get("HARNESS_MCP_TOOL_GROUPS") or "all").strip().lower()
        if raw in ("all", "*", ""):
            return list(TOOLS)
        requested = [g.strip() for g in raw.split(",") if g.strip()]
        for group in requested:
            if group in PRESETS:
                names.extend(PRESETS[group])
            elif group in GROUPS:
                names.extend(t.name for t in GROUPS[group])
            # An unknown group name contributes nothing rather than silently falling back
            # to the full catalog - a typo should be visible as a missing tool, not as a
            # 10k-token surprise.

    seen: set[str] = set()
    selected = []
    for name in names:
        tool = TOOL_BY_NAME.get(name)
        if tool is not None and name not in seen:
            seen.add(name)
            selected.append(tool)
    return selected or list(TOOLS)


EXPOSED: list[Any] = _selected_tools()
EXPOSED_BY_NAME: dict[str, Any] = {t.name: t for t in EXPOSED}


def _tool_schema(t: Any) -> dict[str, Any]:
    if hasattr(t.args_schema, "model_json_schema"):
        schema = t.args_schema.model_json_schema()
    elif t.args_schema is not None:
        schema = t.args_schema.schema()
    else:
        schema = {}
    schema.pop("description", None)
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {"name": t.name, "description": t.description or "", "inputSchema": schema}


def _list_tools_result() -> dict[str, Any]:
    return {"tools": [_tool_schema(t) for t in EXPOSED]}


def _call_tool_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    tool = EXPOSED_BY_NAME.get(name)
    if tool is None:
        hidden = " (present in the registry but not exposed by this server's tool filter)" if name in TOOL_BY_NAME else ""
        return {
            "content": [{"type": "text", "text": core.to_json({"ok": False, "error": f"Unknown tool: {name}{hidden}"})}],
            "isError": True,
        }
    try:
        result = tool.invoke(arguments)
        return {"content": [{"type": "text", "text": str(result)}], "isError": False}
    except Exception as exc:
        return {
            "content": [{"type": "text", "text": core.to_json({"ok": False, "error": repr(exc)})}],
            "isError": True,
        }


_SEND_LOCK = threading.Lock()


def _send(payload: dict[str, Any]) -> None:
    # Tool calls now run on worker threads (see serve_stdio), so several can finish at once.
    # One JSON-RPC message per line is only true if exactly one thread is writing at a time -
    # without this lock two concurrent replies interleave and corrupt the stream.
    raw = json.dumps(payload, ensure_ascii=False)
    with _SEND_LOCK:
        sys.stdout.buffer.write(raw.encode("utf-8") + b"\n")
        sys.stdout.buffer.flush()


def _read_message() -> dict[str, Any] | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    return json.loads(line.decode("utf-8"))


def _handle(msg: dict[str, Any]) -> None:
    method = msg.get("method")
    msg_id = msg.get("id")

    if method == "initialize":
        _send({
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "harness-agent", "version": "1.0"},
            },
        })
        return
    if method == "notifications/initialized":
        return
    if method == "tools/list":
        _send({"jsonrpc": "2.0", "id": msg_id, "result": _list_tools_result()})
        return
    if method == "tools/call":
        params = msg.get("params", {}) or {}
        # Dispatched to a worker thread by serve_stdio, so a 10-minute delegation no longer
        # blocks the read loop: further tool calls and the client's health-check pings keep
        # being served while it runs.
        result = _call_tool_result(str(params.get("name", "")), params.get("arguments", {}) or {})
        _send({"jsonrpc": "2.0", "id": msg_id, "result": result})
        return
    if method == "ping":
        _send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        return
    if msg_id is not None:
        _send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}})


def serve_stdio() -> None:
    permissions.mark_background()
    core.ensure_dirs()

    # workflow_run needs a live AgentRuntime to actually execute a saved workflow - wire a
    # provider up here too, same as cli.py's main()/run_one_shot_workflow, so workflow_run
    # works when called through this MCP surface, not only through the interactive REPL.
    # Built lazily (only on the first actual workflow_run call, not at server startup):
    # `from .agent_runtime import AgentRuntime` plus constructing one is ~2.7s alone
    # (langchain/langgraph imports + building the LLM-bound graph) - paying that on every
    # server start, before the client's very first `initialize`/`tools/list` can even be
    # answered, was slow enough to make Claude Code's own MCP connection health-check give
    # up and report "Failed to connect" even though the server was still starting up fine.
    _runtime_holder: dict[str, Any] = {}

    def _get_or_create_runtime() -> Any:
        if "runtime" not in _runtime_holder:
            from .agent_runtime import AgentRuntime

            _runtime_holder["runtime"] = AgentRuntime(core.DEFAULT_MODEL)
        return _runtime_holder["runtime"]

    workflows.set_runtime_provider(_get_or_create_runtime)

    # Tool calls run on worker threads; everything else (initialize / tools/list / ping) is
    # answered inline on the read loop since it is instant. Previously EVERY message was
    # handled to completion before the next was read, so one long delegation froze the whole
    # server - no second delegation, no polling, and the client's pings queued behind the job.
    # Threads are daemon so a client disconnect can still tear the process down.
    pool = ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_CALLS, thread_name_prefix="mcp-call")

    def _dispatch(message: dict[str, Any]) -> None:
        try:
            _handle(message)
        except Exception as exc:
            message_id = message.get("id") if isinstance(message, dict) else None
            if message_id is not None:
                _send({"jsonrpc": "2.0", "id": message_id, "error": {"code": -32000, "message": repr(exc)}})

    try:
        while True:
            try:
                msg = _read_message()
            except Exception:
                break
            if msg is None:
                break
            if isinstance(msg, dict) and msg.get("method") == "tools/call":
                # permissions context is a contextvar and does not propagate into pool threads,
                # so each worker re-marks itself as background before touching the gate.
                pool.submit(_run_call_in_thread, msg, _dispatch)
            else:
                _dispatch(msg)
    finally:
        # Don't block shutdown on an in-flight delegation the client will never read.
        pool.shutdown(wait=False, cancel_futures=True)


def _run_call_in_thread(msg: dict[str, Any], dispatch: Any) -> None:
    permissions.mark_background()
    dispatch(msg)


if __name__ == "__main__":
    serve_stdio()
