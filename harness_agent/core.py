"""Shared config, paths, and generic helpers for the local agent.

Every other module reaches workspace-derived paths through this module's
attributes (``core.WORKSPACE``, ``core.RAG_DB_PATH``, ...) rather than
importing the names directly. That indirection is what makes the workspace
switchable at runtime: ``set_workspace()`` rebinds these module globals, and
anything that does ``core.WORKSPACE`` at call time picks up the new value.
Anything that did ``from .core import WORKSPACE`` would keep the stale path.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# =============================================================================
# Static config (does not change with workspace)
# =============================================================================

LITELLM_BASE_URL = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000/v1")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-anything")
DEFAULT_MODEL = os.environ.get("MODEL", "cerebras-glm-4.7")

SEARXNG_URL = os.environ.get("SEARXNG_URL", "http://localhost:8080")

MAX_READ_BYTES = int(os.environ.get("MAX_READ_BYTES", str(1024 * 1024)))
MAX_WRITE_BYTES = int(os.environ.get("MAX_WRITE_BYTES", str(10 * 1024 * 1024)))
MAX_OUTPUT_BYTES = int(os.environ.get("MAX_OUTPUT_BYTES", str(256 * 1024)))
MAX_TREE_ENTRIES = int(os.environ.get("MAX_TREE_ENTRIES", "3000"))
MAX_LOG_BYTES = int(os.environ.get("MAX_LOG_BYTES", str(1024 * 1024)))
MAX_RAG_FILE_BYTES = int(os.environ.get("MAX_RAG_FILE_BYTES", str(100 * 1024 * 1024)))
RAG_CHUNK_CHARS = int(os.environ.get("RAG_CHUNK_CHARS", "1800"))
RAG_CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "250"))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "8"))
RAG_EMBED_MODEL = os.environ.get("RAG_EMBED_MODEL", os.environ.get("EMBEDDING_MODEL", "")).strip()
RAG_EMBED_DIM = int(os.environ.get("RAG_EMBED_DIM", "384"))
RAG_EMBED_BATCH = int(os.environ.get("RAG_EMBED_BATCH", "32"))
RAG_HASH_FALLBACK = os.environ.get("RAG_HASH_FALLBACK", "1") == "1"
RAG_AUTO_INGEST_OUTPUT = os.environ.get("RAG_AUTO_INGEST_OUTPUT", "1") == "1"
RAG_AUTO_INGEST_OUTPUT_BYTES = int(os.environ.get("RAG_AUTO_INGEST_OUTPUT_BYTES", str(128 * 1024)))

MODEL_TOOL_RESULT_BYTES = int(os.environ.get("MODEL_TOOL_RESULT_BYTES", "12000"))

DEFAULT_TIMEOUT = int(os.environ.get("DEFAULT_TIMEOUT", "60"))
MAX_TIMEOUT = int(os.environ.get("MAX_TIMEOUT", "900"))
RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", "100"))

ALLOW_UNSAFE_COMMANDS = os.environ.get("ALLOW_UNSAFE_COMMANDS", "0") == "1"
ALLOW_GIT_DESTRUCTIVE = os.environ.get("ALLOW_GIT_DESTRUCTIVE", "0") == "1"
ALLOW_DELETE = os.environ.get("ALLOW_DELETE", "1") == "1"

# Runtime-toggleable via /thinking and /stream (cli.py mutates these directly as
# core.SHOW_THINKING / core.STREAM_RESPONSES - same "read through core.X" pattern as the
# workspace-derived paths, so a toggle takes effect on the very next turn everywhere that
# checks it, with no plumbing needed through intermediate call sites).
SHOW_THINKING = os.environ.get("SHOW_THINKING", "1") != "0"
STREAM_RESPONSES = os.environ.get("STREAM_RESPONSES", "0") != "0"

# Sub-agent delegation depth guard: a spawned sub-agent can itself spawn
# sub-agents up to this many levels deep before the tool refuses.
MAX_SUBAGENT_DEPTH = int(os.environ.get("MAX_SUBAGENT_DEPTH", "2"))
SUBAGENT_DEPTH = 0  # mutated via contextvar-style push/pop in subagent.py

# Result-quality retry loop (subagent.py). The LiteLLM proxy already retries/fails-over
# on a NON-200 transport error (allowed_fails); these govern the complementary loop for a
# 200 response whose *content* is wrong or incomplete - re-run the sub-task, feeding the
# rejection reason back to the model, up to this many attempts before giving up.
SUBAGENT_MAX_ATTEMPTS = max(1, int(os.environ.get("SUBAGENT_MAX_ATTEMPTS", "3")))
# Comma-separated model names to escalate through on each failed attempt (attempt 1 uses the
# base model, attempt 2 the first entry here, and so on). Blank => keep retrying the same
# model with feedback (still useful for the "needs more data" case, which benefits from
# continuation even without a stronger model). Escape hatch: SUBAGENT_ESCALATION_MODELS=verifier,agent-orchestrator
SUBAGENT_ESCALATION_MODELS = [m.strip() for m in os.environ.get("SUBAGENT_ESCALATION_MODELS", "").split(",") if m.strip()]


# =============================================================================
# Workspace-derived paths (mutable; see module docstring)
# =============================================================================

def _derive_workspace_paths(workspace: Path) -> dict[str, Path]:
    cache_dir = workspace / ".agent_cache"
    rag_dir = cache_dir / "rag"
    return {
        "WORKSPACE": workspace,
        "CACHE_DIR": cache_dir,
        "GRAPHQL_CACHE_DIR": cache_dir / "graphql",
        "LOG_DIR": cache_dir / "instance_logs",
        "RAG_DIR": rag_dir,
        "RAG_SOURCES_DIR": rag_dir / "sources",
        "RAG_DB_PATH": rag_dir / "rag.sqlite3",
        "MEMORY_DB_PATH": cache_dir / "memory.sqlite3",
        "MCP_CONFIG_PATH": cache_dir / "mcp_servers.json",
        "MCP_LOG_DIR": cache_dir / "mcp_logs",
        "WORKFLOW_CONFIG_PATH": cache_dir / "workflows.json",
        "AUTOMATIONS_CONFIG_PATH": cache_dir / "automations.json",
        "PROMPT_HISTORY_PATH": cache_dir / "prompt_history.txt",
    }


_initial = _derive_workspace_paths(Path(os.environ.get("AGENT_WORKSPACE", os.getcwd())).expanduser().resolve())
WORKSPACE: Path = _initial["WORKSPACE"]
CACHE_DIR: Path = _initial["CACHE_DIR"]
GRAPHQL_CACHE_DIR: Path = _initial["GRAPHQL_CACHE_DIR"]
LOG_DIR: Path = _initial["LOG_DIR"]
RAG_DIR: Path = _initial["RAG_DIR"]
RAG_SOURCES_DIR: Path = _initial["RAG_SOURCES_DIR"]
RAG_DB_PATH: Path = _initial["RAG_DB_PATH"]
MEMORY_DB_PATH: Path = _initial["MEMORY_DB_PATH"]
MCP_CONFIG_PATH: Path = _initial["MCP_CONFIG_PATH"]
MCP_LOG_DIR: Path = _initial["MCP_LOG_DIR"]
WORKFLOW_CONFIG_PATH: Path = _initial["WORKFLOW_CONFIG_PATH"]
AUTOMATIONS_CONFIG_PATH: Path = _initial["AUTOMATIONS_CONFIG_PATH"]
PROMPT_HISTORY_PATH: Path = _initial["PROMPT_HISTORY_PATH"]
WORKSPACE_HISTORY: list[str] = []


def ensure_dirs() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHQL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    RAG_SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    MCP_LOG_DIR.mkdir(parents=True, exist_ok=True)


def set_workspace(new_workspace: str | Path) -> dict[str, Any]:
    """Switch the active workspace at runtime.

    Rebinds every workspace-derived global in this module. Every other module
    must read these through ``core.NAME`` (not ``from .core import NAME``) for
    the switch to actually propagate - see the module docstring.
    """
    global WORKSPACE, CACHE_DIR, GRAPHQL_CACHE_DIR, LOG_DIR, RAG_DIR, RAG_SOURCES_DIR
    global RAG_DB_PATH, MEMORY_DB_PATH, MCP_CONFIG_PATH, MCP_LOG_DIR
    global WORKFLOW_CONFIG_PATH, AUTOMATIONS_CONFIG_PATH, PROMPT_HISTORY_PATH

    path = Path(str(new_workspace)).expanduser()
    if not path.is_absolute():
        path = WORKSPACE / path
    resolved = path.resolve(strict=False)
    resolved.mkdir(parents=True, exist_ok=True)

    previous = str(WORKSPACE)
    derived = _derive_workspace_paths(resolved)
    WORKSPACE = derived["WORKSPACE"]
    CACHE_DIR = derived["CACHE_DIR"]
    GRAPHQL_CACHE_DIR = derived["GRAPHQL_CACHE_DIR"]
    LOG_DIR = derived["LOG_DIR"]
    RAG_DIR = derived["RAG_DIR"]
    RAG_SOURCES_DIR = derived["RAG_SOURCES_DIR"]
    RAG_DB_PATH = derived["RAG_DB_PATH"]
    MEMORY_DB_PATH = derived["MEMORY_DB_PATH"]
    MCP_CONFIG_PATH = derived["MCP_CONFIG_PATH"]
    MCP_LOG_DIR = derived["MCP_LOG_DIR"]
    WORKFLOW_CONFIG_PATH = derived["WORKFLOW_CONFIG_PATH"]
    AUTOMATIONS_CONFIG_PATH = derived["AUTOMATIONS_CONFIG_PATH"]
    PROMPT_HISTORY_PATH = derived["PROMPT_HISTORY_PATH"]
    ensure_dirs()

    if previous != str(WORKSPACE):
        WORKSPACE_HISTORY.append(previous)

    return {"ok": True, "previous_workspace": previous, "workspace": str(WORKSPACE)}


# =============================================================================
# System prompt
# =============================================================================

BRIEFING_FILENAME = "briefing.txt"


def load_briefing() -> str:
    """The operator's standing context for this workspace, prepended to every agent's system
    prompt. Set once via the briefing tools instead of re-pasting the same target/scope/paths
    preamble into every delegated task description. Missing file => empty string."""
    try:
        return (CACHE_DIR / BRIEFING_FILENAME).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return ""


def build_system_prompt() -> str:
    """Build the system prompt against the *current* workspace and capabilities.

    A function rather than a frozen string so a workspace switch or a model
    switch is reflected immediately, and so it can describe exactly which
    capability tiers (memory, RAG, sub-agents, workflows, automations) exist
    without the model having to infer it from tool names alone. Read at call
    time (not import time) for the same reason the briefing is.
    """
    briefing = load_briefing()
    briefing_block = f"""
## Standing briefing from the operator (applies to every task in this workspace)
{briefing}
""" if briefing else ""

    # Accumulated target knowledge from earlier runs. Imported lazily and defensively: core is
    # the bottom of the dependency graph and must not hard-depend on a leaf module, and a
    # missing/corrupt dossier DB must never stop an agent from starting.
    dossier_block = ""
    try:
        from . import dossier as _dossier

        rendered = _dossier.render_for_prompt()
        if rendered:
            dossier_block = f"\n{rendered}\n"
    except Exception:
        dossier_block = ""

    return briefing_block + dossier_block + f"""
You are a local coding, filesystem, git, web-search, browser-instance, GraphQL, and
automation agent running with real tool access on this machine. You are not a sandbox:
treat destructive or irreversible actions (deletes, force-pushes, unsafe commands,
external network calls) with the same caution a careful human operator would.

Workspace root (current):
{WORKSPACE}

## Core operating loop
- Inspect before you act: list/read/search before you write, edit, or run commands.
- Chain tools deliberately: inspect -> search -> read -> edit -> test -> summarize.
  Do not stop after one tool call if the task is not actually verified yet.
- Never claim success unless a tool result shows ok=true or exit_code=0. If a tool
  returns ok=false, fix the underlying issue or report the failure - do not paper over it.
- Use git status/diff before modifying tracked files when relevant.
- Terminal commands stream live to the console; you receive the final summarized result.
- For long-running commands, set a generous timeout rather than letting them get killed.
- Prefer small, reversible, scoped operations over broad destructive ones. If a command
  looks destructive enough that it might be blocked, say what you are about to do and why.

## Three-tier memory model - choose the right tier deliberately
1. SESSION memory (kind="session", the default): short-lived working notes scoped to the
   current task/run - intermediate findings, decisions made this session, things you might
   need again in the next few turns but that are not generally true facts. Cleared on /reset.
2. PROCEDURAL memory (kind="procedural"): durable, reusable "how to do X" knowledge - a fix
   that worked, a runbook, a gotcha worth remembering across sessions and across workspaces.
   Use memory_save_debug_fix for fixes specifically. This is what should survive long-term.
3. RAG memory (kind="rag" via memory_save, or directly via rag_ingest_text/rag_ingest_path):
   large or bulky reference material meant for semantic/lexical search rather than recall by
   ID - documents, big files, terminal/browser output, cached schemas. Saving a memory with
   kind="rag" also indexes it into rag_search automatically.
   Prefer rag_search before read_file when a file is likely too large or the question is
   semantic. For huge files, browser output, long terminal output, large logs, and big
   schemas, ingest into RAG then search/read focused chunks instead of pasting everything
   into context.
   For GraphQL schemas: introspect once, cache, then use graphql_schema_search/graphql_get_type
   instead of re-pasting the full schema.

## Going beyond a single tool-call loop
- Sub-agents: for a large, self-contained sub-task (a multi-file refactor, an isolated
  investigation, anything that would otherwise flood this conversation with intermediate
  tool calls), delegate it with spawn_subagent instead of doing it inline. Give the
  sub-agent a precise, self-contained task description - it does not see this conversation.
- Workflows: if you find yourself about to repeat the same multi-step tool sequence (or the
  operator describes a recurring task), save it with workflow_save so it can be replayed by
  name later via workflow_run, instead of re-deriving the steps every time.
- Automations: if a workflow should run on a schedule without a human re-prompting it
  (periodic recon refresh, nightly checks), use automation_schedule rather than asking the
  operator to remember to re-run it.
- Multiple Caido/Playwright/MCP instances: start named process instances and inspect their
  logs rather than running one at a time when the task needs several concurrently.

## Task tracking
- For any multi-step task - roughly more than two sequential actions, or anything where a
  step could plausibly be forgotten or silently dropped - call todo_write up front with the
  full plan as a list of items, each with an id, content, and status.
- Mark an item in_progress right before you start it and completed immediately after you
  finish it. Call todo_write again whenever the plan itself changes (steps added, removed,
  or reordered) rather than letting the recorded plan drift out of sync with what you are
  actually doing.
- This is what lets the operator and the live todo panel in the interface see real-time
  progress as it happens, instead of only trusting a final summary at the end.

## Hard rules
- Operate inside the workspace for file tools; do not try to escape it.
- Read files before editing them.
- For multiple Caido/Playwright jobs, start named process instances and inspect logs.
- Keep paths compatible with Windows and Linux.
- Use run_terminal only when command output is needed or the user explicitly asks to
  execute something; keep its cwd inside allowed roots.
- Prefer bash/sh on Linux/macOS and powershell/cmd on Windows when the user asks for a
  specific shell.
- SSH: ssh_run executes exactly one remote command over SSH and streams its output the same
  way run_terminal does, then returns - it does not keep the connection open. For a
  persistent interactive remote session that needs to stay open across multiple turns,
  start a named process instance (start_process_instance, kind="generic") running an
  interactive ssh session, then drive it with send_instance_stdin and read_instance_logs
  instead of reconnecting fresh with ssh_run for every single command.

## Uncertainty and cross-verification
- Never assert an uncertain, unverified, or internally contradicted claim as settled fact -
  hedge explicitly instead (say what you are confident about, what you are not, and why).
- This is a deliberate, situational judgment call, not a forced extra pass on every trivial
  turn: reaching for a second opinion on things that were never actually in doubt just
  doubles cost and latency for no benefit.
- Reach for one when it is actually warranted - confidence is genuinely low, tool output is
  ambiguous or contradicts itself, or the task is high stakes (security findings, destructive
  operations, or anything the operator will act on directly without re-checking it
  themselves). In those cases, before finalizing a response:
  - Call compare_models to get an independent second model's take on the same question, or
  - Use spawn_subagents_parallel with a deliberately skeptical/adversarial framing - ask it
    to try to find reasons the current answer is wrong - rather than a framing that just
    re-confirms what you already believe.
- Surface any disagreement between models or checks to the operator rather than silently
  picking one side.
""".strip()


# =============================================================================
# Generic helpers
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def trim_text(text: Any, limit: int = MAX_OUTPUT_BYTES) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= limit:
        return text
    return raw[:limit].decode("utf-8", errors="replace") + "\n...[truncated]..."


def to_json(obj: Any) -> str:
    import json

    return trim_text(json.dumps(obj, ensure_ascii=False, indent=2))


def parse_json_arg(value: str | None, default: Any) -> Any:
    import json

    if value is None or str(value).strip() == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    return json.loads(str(value))


def safe_name(value: str) -> str:
    value = value.strip() or "default"
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value[:100]


def resolve_path(path: str | None = ".") -> Path:
    if path is None or str(path).strip() == "":
        path = "."
    raw = str(path)
    if "\x00" in raw:
        raise ValueError("NUL byte is not allowed in paths")
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = WORKSPACE / p
    resolved = p.resolve(strict=False)
    try:
        resolved.relative_to(WORKSPACE)
    except ValueError as exc:
        raise PermissionError(f"Path escapes workspace: {raw}") from exc
    return resolved


def rel_path(p: Path) -> str:
    return str(p.resolve(strict=False).relative_to(WORKSPACE))


def path_metadata(p: Path) -> dict[str, Any]:
    rp = p.resolve(strict=False)
    try:
        st = rp.stat()
        return {
            "path": rel_path(rp),
            "absolute_path": str(rp),
            "exists": True,
            "type": "directory" if rp.is_dir() else "file" if rp.is_file() else "other",
            "size_bytes": st.st_size,
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        }
    except FileNotFoundError:
        return {
            "path": rel_path(rp),
            "absolute_path": str(rp),
            "exists": False,
        }


def command_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    return env


def blocked_command_reason(command: str) -> str | None:
    if ALLOW_UNSAFE_COMMANDS:
        return None
    dangerous_patterns = [
        r"(?i)\brm\s+-rf\s+/(?:\s|$)",
        r"(?i)\brm\s+-rf\s+/\*",
        r"(?i)\bsudo\s+rm\s+-rf\b",
        r"(?i)\bmkfs(?:\.[a-z0-9]+)?\b",
        r"(?i)\bdd\s+.*\bof=/dev/",
        r"(?i):\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;",
        r"(?i)\bshutdown\b",
        r"(?i)\breboot\b",
        r"(?i)\bpoweroff\b",
        r"(?i)\bhalt\b",
        r"(?i)\bformat\s+[a-z]:",
        r"(?i)\bdiskpart\b",
        r"(?i)\bbcdedit\b",
        r"(?i)\breg\s+delete\b",
        r"(?i)\bdel\s+/[fsq]+\s+[a-z]:\\",
        r"(?i)\brmdir\s+/s\s+/q\s+[a-z]:\\",
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            return f"Blocked by dangerous-command pattern: {pattern}"
    return None


def graceful_stop_process(proc: subprocess.Popen, kill: bool = False, stdin_grace_sec: float = 3.0, term_grace_sec: float = 7.0, kill_grace_sec: float = 5.0) -> int:
    """Stop a child process, preferring a graceful exit over an abrupt kill.

    subprocess.terminate() on Windows is TerminateProcess(): an unconditional kill with
    no chance for the child to clean up. Many stdio servers (MCP servers, REPLs) treat
    stdin EOF as their shutdown signal, so closing stdin first gives well-behaved
    children a real chance to exit cleanly before we escalate to terminate/kill.
    """
    if kill:
        proc.kill()
        return proc.wait(timeout=kill_grace_sec)

    if proc.stdin:
        try:
            proc.stdin.close()
        except Exception:
            pass
    try:
        return proc.wait(timeout=stdin_grace_sec)
    except subprocess.TimeoutExpired:
        pass

    proc.terminate()
    try:
        return proc.wait(timeout=term_grace_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        return proc.wait(timeout=kill_grace_sec)
