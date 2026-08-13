"""MCP server registry and tool bridge.

Ported verbatim from main.py's "# MCP server registry and tool bridge"
section. References workspace-derived paths via ``core.X`` (not top-level
imports) so that switching the workspace at runtime keeps picking up the
live value - see core.py's module docstring.
"""

from __future__ import annotations

import collections
import json
import os
import platform
import queue
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

import requests
from langchain_core.tools import tool

from . import core


# =============================================================================
# MCP server registry and tool bridge
# =============================================================================

@dataclass
class McpSession:
    name: str
    proc: Any
    started_at: float = field(default_factory=time.time)
    next_id: int = 1
    pending: dict[int, queue.Queue[dict[str, Any]]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    logs: Deque[tuple[float, str, str]] = field(default_factory=lambda: collections.deque(maxlen=5000))
    log_file: Path | None = None


MCP_SESSIONS: dict[str, McpSession] = {}
MCP_LOCK = threading.Lock()
MCP_PROTOCOL_VERSION = os.environ.get("MCP_PROTOCOL_VERSION", "2024-11-05")


def mcp_load_servers() -> dict[str, dict[str, Any]]:
    core.ensure_dirs()
    if not core.MCP_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(core.MCP_CONFIG_PATH.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, dict):
            return {core.safe_name(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def mcp_save_servers(servers: dict[str, dict[str, Any]]) -> None:
    core.ensure_dirs()
    core.MCP_CONFIG_PATH.write_text(json.dumps(servers, ensure_ascii=False, indent=2), encoding="utf-8")


def mcp_server_public(name: str, server: dict[str, Any]) -> dict[str, Any]:
    status = mcp_runtime_status(name, server)
    public = dict(server)
    if public.get("env"):
        public["env_keys"] = sorted(public.get("env", {}).keys())
    public.pop("env", None)
    public.update(status)
    return public


def mcp_runtime_status(name: str, server: dict[str, Any] | None = None) -> dict[str, Any]:
    name = core.safe_name(name)
    transport = (server or {}).get("transport", "stdio")
    with MCP_LOCK:
        session = MCP_SESSIONS.get(name)
        if session:
            poll = session.proc.poll()
            return {
                "running": poll is None,
                "exit_code": poll,
                "pid": getattr(session.proc, "pid", None),
                "uptime_sec": round(time.time() - session.started_at, 2),
                "log_file": core.rel_path(session.log_file) if session.log_file else None,
            }
    if transport in {"http", "streamable_http", "sse"}:
        return {"running": bool((server or {}).get("session_id")), "exit_code": None, "pid": None, "uptime_sec": None, "log_file": None}
    return {"running": False, "exit_code": None, "pid": None, "uptime_sec": None, "log_file": None}


def mcp_make_command(command: str, args: list[str]) -> list[str]:
    command = str(command or "").strip()
    if not command:
        return []
    if args:
        cmd = [command, *[str(a) for a in args]]
    else:
        cmd = shlex.split(command, posix=(platform.system().lower() != "windows"))
    if platform.system().lower() == "windows" and cmd:
        if cmd[0].lower() == "npx":
            cmd[0] = shutil.which("npx.cmd") or shutil.which("npx.exe") or cmd[0]
        elif cmd[0].lower() == "node":
            cmd[0] = shutil.which("node.exe") or cmd[0]
    return cmd


def mcp_log(session: McpSession, stream: str, text_value: str) -> None:
    ts = time.time()
    with MCP_LOCK:
        session.logs.append((ts, stream, text_value))
    if session.log_file:
        try:
            with session.log_file.open("a", encoding="utf-8", errors="replace") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))} {stream}] {text_value}\n")
        except Exception:
            pass


def mcp_send_raw(session: McpSession, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not session.proc.stdin:
        raise RuntimeError("MCP process has no stdin")
    session.proc.stdin.write(raw + b"\n")
    session.proc.stdin.flush()


def mcp_send_notification(session: McpSession, method: str, params: dict[str, Any] | None = None) -> None:
    payload = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        payload["params"] = params
    mcp_send_raw(session, payload)


def mcp_rpc(session: McpSession, method: str, params: dict[str, Any] | None = None, timeout_sec: int = 30) -> dict[str, Any]:
    timeout_sec = max(1, min(int(timeout_sec), core.MAX_TIMEOUT))
    with session.lock:
        request_id = session.next_id
        session.next_id += 1
        q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        session.pending[request_id] = q
        payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        mcp_send_raw(session, payload)
    try:
        response = q.get(timeout=timeout_sec)
    except queue.Empty:
        with session.lock:
            session.pending.pop(request_id, None)
        return {"ok": False, "error": f"MCP request timed out: {method}", "id": request_id}
    if "error" in response:
        return {"ok": False, "error": response.get("error"), "id": request_id}
    return {"ok": True, "id": request_id, "result": response.get("result")}


def mcp_stdout_reader(session: McpSession) -> None:
    stdout = session.proc.stdout
    try:
        while True:
            line = stdout.readline() if stdout else b""
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line.decode("utf-8"))
            except Exception as exc:
                mcp_log(session, "parse_error", f"{repr(exc)} body={core.trim_text(line.decode('utf-8', errors='replace'), 2000)}")
                continue
            msg_id = message.get("id")
            if msg_id is not None:
                with session.lock:
                    q = session.pending.pop(int(msg_id), None)
                if q:
                    q.put(message)
                else:
                    mcp_log(session, "orphan_response", json.dumps(message, ensure_ascii=False))
            else:
                mcp_log(session, "notification", json.dumps(message, ensure_ascii=False))
    except Exception as exc:
        mcp_log(session, "reader_error", repr(exc))


def mcp_stderr_reader(session: McpSession) -> None:
    stderr = session.proc.stderr
    try:
        while True:
            line = stderr.readline() if stderr else b""
            if not line:
                break
            if isinstance(line, bytes):
                line_text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            else:
                line_text = str(line).rstrip("\r\n")
            mcp_log(session, "stderr", line_text)
    except Exception as exc:
        mcp_log(session, "stderr_reader_error", repr(exc))


def mcp_http_extract_json(response: requests.Response) -> dict[str, Any]:
    content_type = response.headers.get("content-type", "").lower()
    if "text/event-stream" in content_type:
        data_lines = []
        for line in response.text.splitlines():
            if line.startswith("data:"):
                value = line.removeprefix("data:").strip()
                if value and value != "[DONE]":
                    data_lines.append(value)
        for value in reversed(data_lines):
            try:
                return json.loads(value)
            except Exception:
                continue
        return {"error": {"message": "No JSON data event found", "raw": core.trim_text(response.text, 4000)}}
    return response.json()


def mcp_http_rpc(server: dict[str, Any], method: str, params: dict[str, Any] | None = None, timeout_sec: int = 30, notification: bool = False) -> dict[str, Any]:
    url = str(server.get("url") or "").strip()
    if not url:
        return {"ok": False, "error": "HTTP MCP server requires url"}
    request_id = int(time.time() * 1000) % 1_000_000_000
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if not notification:
        payload["id"] = request_id
    if params is not None:
        payload["params"] = params
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": MCP_PROTOCOL_VERSION,
    }
    if server.get("session_id"):
        headers["Mcp-Session-Id"] = str(server["session_id"])
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=max(1, min(int(timeout_sec), core.MAX_TIMEOUT)))
        if r.status_code in {202, 204} and notification:
            return {"ok": True, "result": None}
        r.raise_for_status()
        session_id = r.headers.get("mcp-session-id") or r.headers.get("Mcp-Session-Id")
        if session_id:
            server["session_id"] = session_id
        data = mcp_http_extract_json(r)
        if "error" in data:
            return {"ok": False, "error": data.get("error"), "id": data.get("id", request_id)}
        return {"ok": True, "id": data.get("id", request_id), "result": data.get("result")}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def mcp_initialize_params() -> dict[str, Any]:
    return {
        "protocolVersion": MCP_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "litellm-langgraph-local-agent", "version": "1.0"},
    }


def mcp_get_session(name: str) -> McpSession | None:
    name = core.safe_name(name)
    with MCP_LOCK:
        session = MCP_SESSIONS.get(name)
        if session and session.proc.poll() is None:
            return session
    return None


def mcp_start_server_impl(name: str, timeout_sec: int = 30) -> dict[str, Any]:
    name = core.safe_name(name)
    servers = mcp_load_servers()
    server = servers.get(name)
    if not server:
        return {"ok": False, "error": "Unknown MCP server", "name": name}
    if server.get("disabled"):
        return {"ok": False, "error": "MCP server is disabled", "name": name}
    transport = str(server.get("transport") or "stdio").lower()
    if transport in {"http", "streamable_http", "sse"}:
        init = mcp_http_rpc(server, "initialize", mcp_initialize_params(), timeout_sec=timeout_sec)
        if init.get("ok"):
            mcp_http_rpc(server, "notifications/initialized", {}, timeout_sec=timeout_sec, notification=True)
            server["last_started_at"] = core.now_str()
            server["last_error"] = ""
            servers[name] = server
            mcp_save_servers(servers)
        else:
            server["last_error"] = init.get("error")
            servers[name] = server
            mcp_save_servers(servers)
        return {**init, "name": name, "transport": transport, "url": server.get("url")}

    with MCP_LOCK:
        existing = MCP_SESSIONS.get(name)
        if existing and existing.proc.poll() is None:
            return {"ok": True, "name": name, "message": "already running", **mcp_runtime_status(name, server)}

    args = server.get("args") if isinstance(server.get("args"), list) else []
    cmd = mcp_make_command(str(server.get("command") or ""), args)
    if not cmd:
        return {"ok": False, "error": "MCP server command is empty", "name": name}
    cwd = str(server.get("cwd") or ".")
    cwd_path = core.resolve_path(cwd)
    env = server.get("env") if isinstance(server.get("env"), dict) else {}
    core.ensure_dirs()
    log_file = core.MCP_LOG_DIR / f"{name}.log"
    try:
        with log_file.open("a", encoding="utf-8", errors="replace") as f:
            f.write(f"\n--- MCP started {core.now_str()} name={name} command={cmd} cwd={cwd_path} ---\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd_path),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            env=core.command_env(env),
            bufsize=0,
        )
        session = McpSession(name=name, proc=proc, log_file=log_file)
        threading.Thread(target=mcp_stdout_reader, args=(session,), daemon=True).start()
        threading.Thread(target=mcp_stderr_reader, args=(session,), daemon=True).start()
        with MCP_LOCK:
            MCP_SESSIONS[name] = session
        init = mcp_rpc(session, "initialize", mcp_initialize_params(), timeout_sec=timeout_sec)
        if not init.get("ok"):
            try:
                proc.terminate()
            except Exception:
                pass
            server["last_error"] = init.get("error")
            servers[name] = server
            mcp_save_servers(servers)
            return {**init, "name": name, "transport": transport}
        mcp_send_notification(session, "notifications/initialized", {})
        server["last_started_at"] = core.now_str()
        server["last_error"] = ""
        servers[name] = server
        mcp_save_servers(servers)
        return {"ok": True, "name": name, "transport": transport, "command": cmd, **mcp_runtime_status(name, server), "initialize": init.get("result")}
    except Exception as exc:
        server["last_error"] = repr(exc)
        servers[name] = server
        mcp_save_servers(servers)
        return {"ok": False, "error": repr(exc), "name": name}


def mcp_stop_server_impl(name: str, kill: bool = False) -> dict[str, Any]:
    name = core.safe_name(name)
    servers = mcp_load_servers()
    server = servers.get(name, {})
    transport = str(server.get("transport") or "stdio").lower()
    if transport in {"http", "streamable_http", "sse"}:
        if server.get("session_id"):
            server.pop("session_id", None)
            server["last_stopped_at"] = core.now_str()
            servers[name] = server
            mcp_save_servers(servers)
        return {"ok": True, "name": name, "transport": transport, "message": "cleared HTTP MCP session id"}
    with MCP_LOCK:
        session = MCP_SESSIONS.get(name)
    if not session:
        return {"ok": True, "name": name, "message": "not running"}
    if session.proc.poll() is not None:
        return {"ok": True, "name": name, "message": "already stopped", "exit_code": session.proc.returncode}
    try:
        exit_code = core.graceful_stop_process(session.proc, kill=kill)
        if name in servers:
            servers[name]["last_stopped_at"] = core.now_str()
            mcp_save_servers(servers)
        return {"ok": True, "name": name, "exit_code": exit_code}
    except Exception as exc:
        return {"ok": False, "name": name, "error": repr(exc)}


def mcp_refresh_tools_impl(name: str = "", timeout_sec: int = 30) -> dict[str, Any]:
    servers = mcp_load_servers()
    names = [core.safe_name(name)] if name.strip() else sorted(servers.keys())
    refreshed = []
    errors = []
    for server_name in names:
        server = servers.get(server_name)
        if not server:
            errors.append({"name": server_name, "error": "Unknown MCP server"})
            continue
        if server.get("disabled"):
            errors.append({"name": server_name, "error": "Server disabled"})
            continue
        transport = str(server.get("transport") or "stdio").lower()
        if transport in {"http", "streamable_http", "sse"}:
            if not server.get("session_id"):
                init = mcp_http_rpc(server, "initialize", mcp_initialize_params(), timeout_sec=timeout_sec)
                if init.get("ok"):
                    mcp_http_rpc(server, "notifications/initialized", {}, timeout_sec=timeout_sec, notification=True)
            result = mcp_http_rpc(server, "tools/list", {}, timeout_sec=timeout_sec)
        else:
            session = mcp_get_session(server_name)
            if not session:
                errors.append({"name": server_name, "error": "Server is not running"})
                continue
            result = mcp_rpc(session, "tools/list", {}, timeout_sec=timeout_sec)
        if not result.get("ok"):
            server["last_error"] = result.get("error")
            servers[server_name] = server
            errors.append({"name": server_name, "error": result.get("error")})
            continue
        tools = (result.get("result") or {}).get("tools", [])
        normalized = []
        for tool_item in tools:
            if isinstance(tool_item, dict):
                normalized.append({
                    "name": tool_item.get("name"),
                    "description": tool_item.get("description", ""),
                    "inputSchema": tool_item.get("inputSchema") or tool_item.get("input_schema") or {},
                })
        server["tools_cache"] = normalized
        server["tools_refreshed_at"] = core.now_str()
        server["last_error"] = ""
        servers[server_name] = server
        refreshed.append({"name": server_name, "count": len(normalized), "tools": normalized})
    mcp_save_servers(servers)
    return {"ok": len(errors) == 0, "refreshed": refreshed, "errors": errors}


@tool
def mcp_add_server(name: str, command: str = "", args_json: str = "", url: str = "", transport: str = "stdio", cwd: str = ".", env_json: str = "", disabled: bool = False, autostart: bool = False, description: str = "") -> str:
    """Add or update an MCP server definition. Use transport=stdio with command/args_json or transport=http with url."""
    try:
        name = core.safe_name(name)
        transport = str(transport or "stdio").lower().replace("-", "_")
        if transport not in {"stdio", "http", "streamable_http", "sse"}:
            return core.to_json({"ok": False, "error": "transport must be stdio, http, streamable_http, or sse"})
        args = core.parse_json_arg(args_json, []) if args_json else []
        if not isinstance(args, list):
            return core.to_json({"ok": False, "error": "args_json must be a JSON list"})
        env = core.parse_json_arg(env_json, {}) if env_json else {}
        if not isinstance(env, dict):
            return core.to_json({"ok": False, "error": "env_json must be a JSON object"})
        if transport == "stdio" and not str(command).strip():
            return core.to_json({"ok": False, "error": "stdio MCP server requires command"})
        if transport in {"http", "streamable_http", "sse"} and not str(url).strip():
            return core.to_json({"ok": False, "error": "HTTP MCP server requires url"})
        # Validate cwd stays inside workspace for stdio servers.
        cwd_value = core.rel_path(core.resolve_path(cwd)) if transport == "stdio" else str(cwd or ".")
        servers = mcp_load_servers()
        now = core.now_str()
        existing = servers.get(name, {})
        servers[name] = {
            "name": name,
            "transport": transport,
            "command": str(command or ""),
            "args": [str(a) for a in args],
            "url": str(url or ""),
            "cwd": cwd_value,
            "env": {str(k): str(v) for k, v in env.items()},
            "disabled": bool(disabled),
            "autostart": bool(autostart),
            "description": str(description or ""),
            "created_at": existing.get("created_at") or now,
            "updated_at": now,
            "tools_cache": existing.get("tools_cache", []),
            "tools_refreshed_at": existing.get("tools_refreshed_at", ""),
            "last_error": existing.get("last_error", ""),
        }
        mcp_save_servers(servers)
        return core.to_json({"ok": True, "server": mcp_server_public(name, servers[name])})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_remove_server(name: str, stop: bool = True) -> str:
    """Remove an MCP server definition. stop=true stops a running local stdio session first."""
    try:
        name = core.safe_name(name)
        stop_result = None
        if stop:
            stop_result = mcp_stop_server_impl(name)
        servers = mcp_load_servers()
        if name not in servers:
            return core.to_json({"ok": False, "error": "Unknown MCP server", "name": name, "stop_result": stop_result})
        removed = servers.pop(name)
        mcp_save_servers(servers)
        return core.to_json({"ok": True, "name": name, "removed": mcp_server_public(name, removed), "stop_result": stop_result})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_list_servers(show_disabled: bool = False) -> str:
    """List configured MCP servers and runtime status."""
    try:
        servers = mcp_load_servers()
        items = []
        for name, server in sorted(servers.items()):
            if server.get("disabled") and not show_disabled:
                continue
            items.append(mcp_server_public(name, server))
        return core.to_json({"ok": True, "config_path": core.rel_path(core.MCP_CONFIG_PATH), "count": len(items), "servers": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_start_server(name: str, timeout_sec: int = 30) -> str:
    """Start or initialize a configured MCP server by name."""
    try:
        return core.to_json(mcp_start_server_impl(name, timeout_sec=timeout_sec))
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_stop_server(name: str, kill: bool = False) -> str:
    """Stop a running MCP server by name, or clear HTTP MCP session state."""
    try:
        return core.to_json(mcp_stop_server_impl(name, kill=kill))
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_server_logs(name: str, max_lines: int = 200) -> str:
    """Read recent stderr/protocol logs for a running MCP stdio server."""
    try:
        name = core.safe_name(name)
        max_lines = max(1, min(int(max_lines), 2000))
        with MCP_LOCK:
            session = MCP_SESSIONS.get(name)
            if not session:
                return core.to_json({"ok": False, "error": "No running stdio MCP session", "name": name})
            logs = list(session.logs)[-max_lines:]
            poll = session.proc.poll()
        return core.to_json({
            "ok": True,
            "name": name,
            "running": poll is None,
            "exit_code": poll,
            "logs": [{"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), "stream": stream, "text": core.trim_text(line, 4000)} for ts, stream, line in logs],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_refresh_tools(name: str = "", timeout_sec: int = 30) -> str:
    """Refresh cached tools from one running MCP server, or all running/reachable MCP servers when name is empty."""
    try:
        return core.to_json(mcp_refresh_tools_impl(name=name, timeout_sec=timeout_sec))
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_search_tools(query: str, server_name: str = "", limit: int = 50) -> str:
    """Search cached MCP tool names/descriptions. Run mcp_refresh_tools first to populate caches."""
    try:
        q = str(query or "").lower().strip()
        limit = max(1, min(int(limit), 200))
        servers = mcp_load_servers()
        results = []
        for name, server in sorted(servers.items()):
            if server_name.strip() and core.safe_name(server_name) != name:
                continue
            for tool_item in server.get("tools_cache", []) or []:
                if not isinstance(tool_item, dict):
                    continue
                haystack = f"{tool_item.get('name', '')}\n{tool_item.get('description', '')}".lower()
                if not q or q in haystack or any(tok in haystack for tok in q.split()):
                    results.append({
                        "server": name,
                        "tool": tool_item.get("name"),
                        "description": tool_item.get("description", ""),
                        "inputSchema": tool_item.get("inputSchema", {}),
                        "tools_refreshed_at": server.get("tools_refreshed_at", ""),
                    })
                    if len(results) >= limit:
                        return core.to_json({"ok": True, "query": query, "count": len(results), "results": results})
        return core.to_json({"ok": True, "query": query, "count": len(results), "results": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def mcp_call_tool(server_name: str, tool_name: str, arguments_json: str = "{}", timeout_sec: int = 60) -> str:
    """Call a tool on a running MCP server. arguments_json must be a JSON object string."""
    try:
        server_name = core.safe_name(server_name)
        servers = mcp_load_servers()
        server = servers.get(server_name)
        if not server:
            return core.to_json({"ok": False, "error": "Unknown MCP server", "server_name": server_name})
        if server.get("disabled"):
            return core.to_json({"ok": False, "error": "MCP server is disabled", "server_name": server_name})
        arguments = core.parse_json_arg(arguments_json, {}) if arguments_json else {}
        if not isinstance(arguments, dict):
            return core.to_json({"ok": False, "error": "arguments_json must be a JSON object"})
        transport = str(server.get("transport") or "stdio").lower()
        if transport in {"http", "streamable_http", "sse"}:
            if not server.get("session_id") and server.get("autostart"):
                start = mcp_start_server_impl(server_name, timeout_sec=min(int(timeout_sec), 30))
                if not start.get("ok"):
                    return core.to_json({"ok": False, "error": "autostart failed", "start": start})
                servers = mcp_load_servers()
                server = servers.get(server_name, server)
            result = mcp_http_rpc(server, "tools/call", {"name": tool_name, "arguments": arguments}, timeout_sec=timeout_sec)
            servers[server_name] = server
            mcp_save_servers(servers)
        else:
            session = mcp_get_session(server_name)
            if not session and server.get("autostart"):
                start = mcp_start_server_impl(server_name, timeout_sec=min(int(timeout_sec), 30))
                if not start.get("ok"):
                    return core.to_json({"ok": False, "error": "autostart failed", "start": start})
                session = mcp_get_session(server_name)
            if not session:
                return core.to_json({"ok": False, "error": "MCP server is not running", "server_name": server_name})
            result = mcp_rpc(session, "tools/call", {"name": tool_name, "arguments": arguments}, timeout_sec=timeout_sec)
        return core.to_json({"ok": bool(result.get("ok")), "server": server_name, "tool": tool_name, "result": result.get("result"), "error": result.get("error")})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
