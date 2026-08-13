"""Streaming process helpers, shell resolution, and terminal/git tools.

Ported verbatim from main.py's "# Streaming process helpers", "# Shell
resolution", and "# Terminal and git tools" sections. References
``core.WORKSPACE``-derived/config names (not top-level imports) so that
switching the workspace at runtime keeps picking up the live value, and
delegates live output streaming to ``terminal_ui.ui_write`` and large-output
RAG auto-ingest to ``rag.rag_store_document``.
"""

from __future__ import annotations

import hashlib
import os
import platform
import queue
import re
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import core
from . import rag
from . import terminal_ui


# =============================================================================
# Streaming process helpers
# =============================================================================

def run_process_stream(
    args: list[str],
    cwd: str | None,
    timeout_sec: int,
    label: str,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    timeout_sec = max(1, min(int(timeout_sec), core.MAX_TIMEOUT))
    started = time.time()
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    output_queue: queue.Queue[tuple[str, str]] = queue.Queue()

    proc = subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env or core.command_env(),
        bufsize=1,
    )

    def reader(pipe: Any, stream_name: str) -> None:
        try:
            for line in iter(pipe.readline, ""):
                output_queue.put((stream_name, line))
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    threads = [
        threading.Thread(target=reader, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=reader, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()

    timed_out = False

    while True:
        if proc.poll() is not None and output_queue.empty():
            break
        if time.time() - started > timeout_sec:
            timed_out = True
            try:
                proc.kill()
            except Exception:
                pass
            break
        try:
            stream_name, line = output_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        terminal_ui.ui_write(f"[{label}:{stream_name}] {line}", flush=True)
        if stream_name == "stdout":
            stdout_parts.append(line)
        else:
            stderr_parts.append(line)

    while not output_queue.empty():
        stream_name, line = output_queue.get_nowait()
        terminal_ui.ui_write(f"[{label}:{stream_name}] {line}", flush=True)
        if stream_name == "stdout":
            stdout_parts.append(line)
        else:
            stderr_parts.append(line)

    try:
        exit_code = proc.wait(timeout=3)
    except Exception:
        exit_code = proc.returncode

    full_stdout = "".join(stdout_parts)
    full_stderr = "".join(stderr_parts)
    rag_auto_ingest: dict[str, Any] | None = None
    if core.RAG_AUTO_INGEST_OUTPUT and len((full_stdout + full_stderr).encode("utf-8", errors="replace")) >= core.RAG_AUTO_INGEST_OUTPUT_BYTES:
        try:
            source_name = f"{label}:{time.strftime('%Y%m%d-%H%M%S')}:{hashlib.sha256((' '.join(args) + str(started)).encode()).hexdigest()[:8]}"
            rag_auto_ingest = rag.rag_store_document(
                collection="terminal",
                source_type="terminal",
                path_or_name=source_name,
                title=source_name,
                text_value=f"COMMAND_ARGS:\n{args}\n\nCWD:\n{cwd}\n\nEXIT_CODE:\n{exit_code}\n\nSTDOUT:\n{full_stdout}\n\nSTDERR:\n{full_stderr}",
                metadata={"args": args, "cwd": cwd, "label": label, "exit_code": exit_code, "timeout": timed_out},
            )
        except Exception as exc:
            rag_auto_ingest = {"ok": False, "error": repr(exc)}

    return {
        "ok": exit_code == 0 and not timed_out,
        "exit_code": exit_code,
        "timeout": timed_out,
        "elapsed_sec": round(time.time() - started, 3),
        "stdout": core.trim_text(full_stdout),
        "stderr": core.trim_text(full_stderr),
        "rag_auto_ingest": rag_auto_ingest,
        "args": args,
        "cwd": cwd,
    }


# =============================================================================
# Shell resolution
# =============================================================================

def powershell_executable() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def cmd_executable() -> str | None:
    if platform.system().lower() == "windows":
        return shutil.which("cmd.exe") or "cmd.exe"
    return shutil.which("cmd")


def bash_executable() -> str | None:
    return shutil.which("bash") or shutil.which("bash.exe")


def wsl_executable() -> str | None:
    if platform.system().lower() == "windows":
        return shutil.which("wsl.exe") or shutil.which("wsl")
    return None


def windows_path_to_wsl(win_path: Path) -> str | None:
    wsl = wsl_executable()
    if not wsl:
        return None
    try:
        r = subprocess.run([wsl, "wslpath", "-a", str(win_path)], capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        return None
    return None


def build_shell_args(shell: str, command: str, cwd_path: Path) -> tuple[list[str], str | None]:
    shell = shell.lower().strip()
    use_cwd: str | None = str(cwd_path)
    if shell == "powershell":
        exe = powershell_executable()
        if not exe:
            raise RuntimeError("PowerShell not found")
        return [exe, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command], use_cwd
    if shell == "cmd":
        exe = cmd_executable()
        if not exe:
            raise RuntimeError("CMD not found")
        return [exe, "/d", "/s", "/c", command], use_cwd
    if shell == "bash":
        bash = bash_executable()
        if bash:
            return [bash, "-lc", command], use_cwd
        wsl = wsl_executable()
        if not wsl:
            raise RuntimeError("Bash not found")
        wsl_cwd = windows_path_to_wsl(cwd_path)
        if not wsl_cwd:
            raise RuntimeError("Could not convert Windows cwd to WSL path")
        return [wsl, "bash", "-lc", f"cd {shlex.quote(wsl_cwd)} && {command}"], None
    raise ValueError("shell must be powershell, cmd, or bash")


# =============================================================================
# Terminal and git tools
# =============================================================================

@tool
def run_terminal(shell: str, command: str, cwd: str = ".", timeout_sec: int = core.DEFAULT_TIMEOUT) -> str:
    """Run a terminal command in powershell, cmd, or bash from inside the workspace. Streams live output."""
    try:
        shell = shell.lower().strip()
        cwd_path = core.resolve_path(cwd)
        if not cwd_path.exists() or not cwd_path.is_dir():
            return core.to_json({"ok": False, "error": "cwd must exist and be a directory", "cwd": str(cwd_path)})
        reason = core.blocked_command_reason(command)
        if reason:
            return core.to_json({"ok": False, "blocked": True, "error": reason, "hint": "Set ALLOW_UNSAFE_COMMANDS=1 only if you understand the risk."})
        args, use_cwd = build_shell_args(shell, command, cwd_path)
        result = run_process_stream(args=args, cwd=use_cwd, timeout_sec=timeout_sec, label=shell)
        result.update({"shell": shell, "command": command, "workspace_cwd": core.rel_path(cwd_path)})
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


def block_git_args(args_text: str) -> str | None:
    if core.ALLOW_GIT_DESTRUCTIVE:
        return None
    bad = [
        r"(?i)^reset\s+--hard\b",
        r"(?i)^clean\b.*(-f|--force)",
        r"(?i)^push\b.*--force",
        r"(?i)^checkout\b.*(-f|--force)",
        r"(?i)^restore\b.*\s--source\b.*\s/",
        r"(?i)^filter-branch\b",
        r"(?i)^rebase\b",
    ]
    for pattern in bad:
        if re.search(pattern, args_text.strip()):
            return f"Blocked potentially destructive git command: {pattern}. Set ALLOW_GIT_DESTRUCTIVE=1 to allow."
    return None


@tool
def git_command(args: str, cwd: str = ".", timeout_sec: int = core.DEFAULT_TIMEOUT) -> str:
    """Run a git command inside the workspace. Example args: 'status --short' or 'diff -- path'. Streams output."""
    try:
        git = shutil.which("git")
        if not git:
            return core.to_json({"ok": False, "error": "git not found"})
        cwd_path = core.resolve_path(cwd)
        if not cwd_path.is_dir():
            return core.to_json({"ok": False, "error": "cwd is not a directory"})
        reason = block_git_args(args)
        if reason:
            return core.to_json({"ok": False, "blocked": True, "error": reason})
        split_args = shlex.split(args, posix=(platform.system().lower() != "windows"))
        result = run_process_stream([git, *split_args], cwd=str(cwd_path), timeout_sec=timeout_sec, label="git")
        result.update({"git_args": args, "workspace_cwd": core.rel_path(cwd_path)})
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
