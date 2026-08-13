"""Remote-host SSH/SCP tools for the local agent.

Shells out to the system OpenSSH client rather than a Python SSH library
(paramiko/fabric) because the machine already has a real ``ssh``/``scp`` on
PATH and shelling out lets us reuse ``shell_git.run_process_stream`` verbatim
- one streaming/timeout/RAG-auto-ingest code path for every subprocess-backed
tool in this package instead of a second bespoke one just for SSH.

Batch mode plus accept-new host-key checking is baked into every invocation
so a first-time connection to a new host doesn't hang forever on an
interactive "yes/no" prompt that has nowhere to go (stdin is DEVNULL in
``run_process_stream``) - this is the standard non-interactive posture for
scripted SSH, not a security bypass (it only auto-trusts host keys never seen
before; it still fails closed on a *changed* key like normal StrictHostKeyChecking).

The remote command in ``ssh_run`` is passed as a single trailing argv element
to the local ``ssh`` process, exactly like typing ``ssh host 'cmd'`` by hand -
ssh itself forwards that one string to the remote shell for interpretation.
That is standard remote command execution over a real network hop, not a
local shell-injection surface, since nothing on this end tokenizes it.
"""

from __future__ import annotations

import shlex
import shutil

from langchain_core.tools import tool

from . import core
from . import shell_git

DEFAULT_SSH_PORT = 22


def _build_ssh_base_args(
    host: str,
    user: str = "",
    port: int = DEFAULT_SSH_PORT,
    identity_file: str = "",
    extra_args: str = "",
) -> list[str]:
    port = int(port)
    if not (1 <= port <= 65535):
        raise ValueError(f"port must be between 1 and 65535, got {port}")

    ssh = shutil.which("ssh")
    if not ssh:
        raise RuntimeError("ssh executable not found on PATH")

    args = [ssh, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if port != DEFAULT_SSH_PORT:
        args += ["-p", str(port)]
    if identity_file.strip():
        args += ["-i", identity_file.strip()]
    if extra_args.strip():
        args += shlex.split(extra_args.strip())
    args.append(f"{user}@{host}" if user.strip() else host)
    return args


def _build_scp_base_args(
    host: str,
    user: str = "",
    port: int = DEFAULT_SSH_PORT,
    identity_file: str = "",
    extra_args: str = "",
) -> list[str]:
    port = int(port)
    if not (1 <= port <= 65535):
        raise ValueError(f"port must be between 1 and 65535, got {port}")

    scp = shutil.which("scp")
    if not scp:
        raise RuntimeError("scp executable not found on PATH")

    args = [scp, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if port != DEFAULT_SSH_PORT:
        args += ["-P", str(port)]
    if identity_file.strip():
        args += ["-i", identity_file.strip()]
    if extra_args.strip():
        args += shlex.split(extra_args.strip())
    return args


def _scp_destination(host: str, user: str, remote_path: str) -> str:
    return f"{user}@{host}:{remote_path}" if user.strip() else f"{host}:{remote_path}"


@tool
def ssh_run(
    host: str,
    command: str,
    user: str = "",
    port: int = DEFAULT_SSH_PORT,
    identity_file: str = "",
    extra_args: str = "",
    timeout_sec: int = core.DEFAULT_TIMEOUT,
) -> str:
    """Run a command on a remote host over SSH (non-interactive, auto-accepts new host keys). Streams live output."""
    try:
        args = _build_ssh_base_args(host, user=user, port=port, identity_file=identity_file, extra_args=extra_args)
        args.append(command)
        result = shell_git.run_process_stream(args=args, cwd=str(core.WORKSPACE), timeout_sec=timeout_sec, label="ssh")
        result.update({"host": host, "user": user, "port": int(port), "command": command})
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def ssh_upload(
    host: str,
    local_path: str,
    remote_path: str,
    user: str = "",
    port: int = DEFAULT_SSH_PORT,
    identity_file: str = "",
    timeout_sec: int = core.DEFAULT_TIMEOUT,
) -> str:
    """Upload a file from inside the workspace to a remote host over SCP."""
    try:
        local = core.resolve_path(local_path)
        if not local.exists():
            return core.to_json({"ok": False, "error": "local_path does not exist", "local_path": str(local)})
        args = _build_scp_base_args(host, user=user, port=port, identity_file=identity_file)
        args.append(str(local))
        args.append(_scp_destination(host, user, remote_path))
        result = shell_git.run_process_stream(args=args, cwd=str(core.WORKSPACE), timeout_sec=timeout_sec, label="scp")
        result.update({
            "host": host,
            "user": user,
            "port": int(port),
            "local_path": core.rel_path(local),
            "remote_path": remote_path,
        })
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def ssh_download(
    host: str,
    remote_path: str,
    local_path: str,
    user: str = "",
    port: int = DEFAULT_SSH_PORT,
    identity_file: str = "",
    timeout_sec: int = core.DEFAULT_TIMEOUT,
) -> str:
    """Download a file from a remote host to a path inside the workspace over SCP."""
    try:
        local = core.resolve_path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)
        args = _build_scp_base_args(host, user=user, port=port, identity_file=identity_file)
        args.append(_scp_destination(host, user, remote_path))
        args.append(str(local))
        result = shell_git.run_process_stream(args=args, cwd=str(core.WORKSPACE), timeout_sec=timeout_sec, label="scp")
        result.update({
            "host": host,
            "user": user,
            "port": int(port),
            "remote_path": remote_path,
            "local_path": core.rel_path(local),
        })
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
