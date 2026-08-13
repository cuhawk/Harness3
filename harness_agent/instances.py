"""Background instance manager: generic, Caido, Playwright MCP.

Ported verbatim from the monolithic main.py "Background instance manager"
section. Workspace-derived paths (LOG_DIR, etc.) are read through ``core.X``
at call time so a runtime workspace switch (core.set_workspace()) is
correctly picked up - see core.py's module docstring.
"""

from __future__ import annotations

import collections
import platform
import shlex
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque

from langchain_core.tools import tool

from . import core


@dataclass
class ManagedInstance:
    name: str
    kind: str
    command: list[str]
    cwd: str | None
    proc: subprocess.Popen[str]
    started_at: float = field(default_factory=time.time)
    logs: Deque[tuple[float, str, str]] = field(default_factory=lambda: collections.deque(maxlen=5000))
    log_file: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


INSTANCE_REGISTRY: dict[str, ManagedInstance] = {}
INSTANCE_LOCK = threading.Lock()


def start_instance_process(name: str, kind: str, command: list[str], cwd: str | None, env: dict[str, str] | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    core.ensure_dirs()
    name = core.safe_name(name)
    kind = core.safe_name(kind)
    with INSTANCE_LOCK:
        existing = INSTANCE_REGISTRY.get(name)
        if existing and existing.proc.poll() is None:
            return {"ok": False, "error": "Instance already running", "name": name}

    log_file = core.LOG_DIR / f"{name}.log"
    lf = log_file.open("a", encoding="utf-8", errors="replace")
    lf.write(f"\n--- started {time.strftime('%Y-%m-%d %H:%M:%S')} kind={kind} command={command} ---\n")
    lf.flush()

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=core.command_env(env),
        bufsize=1,
    )

    inst = ManagedInstance(name=name, kind=kind, command=command, cwd=cwd, proc=proc, log_file=log_file, metadata=metadata or {})

    def reader(pipe: Any, stream_name: str) -> None:
        try:
            for line in iter(pipe.readline, ""):
                ts = time.time()
                with INSTANCE_LOCK:
                    inst.logs.append((ts, stream_name, line))
                try:
                    lf.write(f"[{stream_name}] {line}")
                    lf.flush()
                except Exception:
                    pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    threading.Thread(target=reader, args=(proc.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=reader, args=(proc.stderr, "stderr"), daemon=True).start()

    with INSTANCE_LOCK:
        INSTANCE_REGISTRY[name] = inst

    return {"ok": True, "name": name, "kind": kind, "pid": proc.pid, "command": command, "cwd": cwd, "log_file": core.rel_path(log_file), "metadata": inst.metadata}


@tool
def start_process_instance(name: str, command: str, kind: str = "generic", cwd: str = ".", env_json: str = "") -> str:
    """Start a named long-running background process instance. Use for multiple Caido/Playwright/custom instances."""
    try:
        cwd_path = core.resolve_path(cwd)
        env = core.parse_json_arg(env_json, {})
        split_cmd = shlex.split(command, posix=(platform.system().lower() != "windows"))
        if not split_cmd:
            return core.to_json({"ok": False, "error": "Empty command"})
        result = start_instance_process(name=core.safe_name(name), kind=kind, command=split_cmd, cwd=str(cwd_path), env=env, metadata={"raw_command": command})
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def start_caido_instance(name: str, command: str, cwd: str = ".", env_json: str = "") -> str:
    """Start a named Caido instance using the exact command you provide. Example command may include custom port/profile flags."""
    try:
        cwd_path = core.resolve_path(cwd)
        env = core.parse_json_arg(env_json, {})
        split_cmd = shlex.split(command, posix=(platform.system().lower() != "windows"))
        if not split_cmd:
            return core.to_json({"ok": False, "error": "Empty command"})
        result = start_instance_process(name=core.safe_name(name), kind="caido", command=split_cmd, cwd=str(cwd_path), env=env, metadata={"raw_command": command})
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def start_playwright_mcp_instance(name: str, port: int, extra_args: str = "", cwd: str = ".", env_json: str = "") -> str:
    """Start a named Playwright MCP HTTP server: npx -y @playwright/mcp@latest --port PORT plus optional extra_args."""
    try:
        cwd_path = core.resolve_path(cwd)
        env = core.parse_json_arg(env_json, {})
        port = int(port)
        if port < 1 or port > 65535:
            return core.to_json({"ok": False, "error": "port must be 1-65535"})
        extra = shlex.split(extra_args, posix=(platform.system().lower() != "windows")) if extra_args.strip() else []
        cmd = ["npx", "-y", "@playwright/mcp@latest", "--port", str(port), *extra]
        if platform.system().lower() == "windows":
            npx = shutil.which("npx.cmd") or shutil.which("npx.exe") or shutil.which("npx")
            if npx:
                cmd[0] = npx
        result = start_instance_process(
            name=core.safe_name(name),
            kind="playwright-mcp",
            command=cmd,
            cwd=str(cwd_path),
            env=env,
            metadata={"port": port, "mcp_url": f"http://localhost:{port}/mcp", "extra_args": extra_args},
        )
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def list_instances() -> str:
    """List running/stopped managed process instances."""
    try:
        with INSTANCE_LOCK:
            items = []
            for name, inst in INSTANCE_REGISTRY.items():
                poll = inst.proc.poll()
                items.append({
                    "name": name,
                    "kind": inst.kind,
                    "pid": inst.proc.pid,
                    "running": poll is None,
                    "exit_code": poll,
                    "uptime_sec": round(time.time() - inst.started_at, 2),
                    "cwd": inst.cwd,
                    "command": inst.command,
                    "log_file": core.rel_path(inst.log_file) if inst.log_file else None,
                    "metadata": inst.metadata,
                })
        return core.to_json({"ok": True, "count": len(items), "instances": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def read_instance_logs(name: str, max_lines: int = 200) -> str:
    """Read recent logs from a managed process instance."""
    try:
        name = core.safe_name(name)
        max_lines = max(1, min(int(max_lines), 2000))
        with INSTANCE_LOCK:
            inst = INSTANCE_REGISTRY.get(name)
            if not inst:
                return core.to_json({"ok": False, "error": "Unknown instance", "name": name})
            logs = list(inst.logs)[-max_lines:]
            poll = inst.proc.poll()
        return core.to_json({
            "ok": True,
            "name": name,
            "running": poll is None,
            "exit_code": poll,
            "logs": [{"time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)), "stream": s, "text": core.trim_text(line, 4000)} for ts, s, line in logs],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def stop_instance(name: str, kill: bool = False) -> str:
    """Stop a managed process instance by name. Closes stdin to request graceful exit first, then terminate, then kill=true escalates straight to a hard kill."""
    try:
        name = core.safe_name(name)
        with INSTANCE_LOCK:
            inst = INSTANCE_REGISTRY.get(name)
            if not inst:
                return core.to_json({"ok": False, "error": "Unknown instance", "name": name})
            if inst.proc.poll() is not None:
                return core.to_json({"ok": True, "name": name, "message": "already stopped", "exit_code": inst.proc.returncode})
        exit_code = core.graceful_stop_process(inst.proc, kill=kill)
        return core.to_json({"ok": True, "name": name, "exit_code": exit_code})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def send_instance_stdin(name: str, text: str) -> str:
    """Send text to a managed process instance stdin."""
    try:
        name = core.safe_name(name)
        with INSTANCE_LOCK:
            inst = INSTANCE_REGISTRY.get(name)
            if not inst:
                return core.to_json({"ok": False, "error": "Unknown instance", "name": name})
            if inst.proc.poll() is not None:
                return core.to_json({"ok": False, "error": "Instance is not running", "exit_code": inst.proc.returncode})
            if not inst.proc.stdin:
                return core.to_json({"ok": False, "error": "Instance has no stdin"})
            inst.proc.stdin.write(text)
            inst.proc.stdin.flush()
        return core.to_json({"ok": True, "name": name, "bytes_sent": len(text.encode('utf-8', errors='replace'))})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
