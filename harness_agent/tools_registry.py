"""Tool registry plus token-budget-aware tool selection.

``ALL_TOOLS`` is the complete capability set. ``select_tools()`` chooses the
smaller per-turn set sent to the model, so simple filesystem/git prompts do not
pay to describe browser, MCP, SSH, GraphQL, automation, and delegation tools.
Set ``HARNESS_TOOL_PROFILE=full`` to bind the complete catalog on every call.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import BaseTool

from . import briefing
from . import dossier
from . import extract_tools
from . import fs_tools
from . import jobs
from . import memory
from . import rag
from . import shell_git
from . import web_search
from . import graphql_tools
from . import instances
from . import mcp_bridge
from . import subagent
from . import workflows
from . import automations
from . import permissions
from . import hooks
from . import tasks
from . import browser_tools
from . import ssh_tools
from . import verify_tools


def _dedupe_tools(*groups: list[BaseTool]) -> list[BaseTool]:
    seen: set[str] = set()
    out: list[BaseTool] = []
    for group in groups:
        for tool in group:
            if tool.name in seen:
                continue
            seen.add(tool.name)
            out.append(tool)
    return out


FILESYSTEM_TOOLS: list[BaseTool] = [
    fs_tools.list_dir, fs_tools.list_tree, fs_tools.stat_path, fs_tools.read_file,
    fs_tools.write_file, fs_tools.touch_file, fs_tools.make_dir, fs_tools.copy_path,
    fs_tools.move_path, fs_tools.delete_path, fs_tools.replace_in_file, fs_tools.find_files,
    fs_tools.grep_files, fs_tools.file_hash, fs_tools.switch_workspace, fs_tools.current_workspace,
]

MEMORY_TOOLS: list[BaseTool] = [
    memory.memory_save, memory.memory_save_debug_fix, memory.memory_search,
    memory.memory_search_debug_fixes, memory.memory_get, memory.memory_update, memory.memory_delete,
    memory.memory_list, memory.memory_promote_from_session, memory.memory_clear_session,
]

RAG_TOOLS: list[BaseTool] = [
    rag.rag_ingest_text, rag.rag_ingest_path, rag.rag_ingest_browser_snapshot,
    rag.rag_ingest_terminal_output, rag.rag_search, rag.rag_read_chunk, rag.rag_delete_source,
    rag.rag_list_collections, rag.rag_collection_stats,
]

TERMINAL_GIT_TOOLS: list[BaseTool] = [
    shell_git.run_terminal, shell_git.git_command,
]

WEB_TOOLS: list[BaseTool] = [
    web_search.searxng_search,
]

GRAPHQL_TOOLS: list[BaseTool] = [
    graphql_tools.graphql_request, graphql_tools.graphql_introspect,
    graphql_tools.graphql_schema_search, graphql_tools.graphql_get_type,
    graphql_tools.rag_ingest_graphql_cache,
]

PROCESS_TOOLS: list[BaseTool] = [
    instances.start_process_instance, instances.start_caido_instance,
    instances.start_playwright_mcp_instance, instances.list_instances, instances.read_instance_logs,
    instances.stop_instance, instances.send_instance_stdin,
]

MCP_TOOLS: list[BaseTool] = [
    mcp_bridge.mcp_add_server, mcp_bridge.mcp_remove_server, mcp_bridge.mcp_list_servers,
    mcp_bridge.mcp_start_server, mcp_bridge.mcp_stop_server, mcp_bridge.mcp_server_logs,
    mcp_bridge.mcp_refresh_tools, mcp_bridge.mcp_search_tools, mcp_bridge.mcp_call_tool,
]

DELEGATION_TOOLS: list[BaseTool] = [
    subagent.spawn_subagent, subagent.spawn_subagents_parallel,
    subagent.spawn_map_reduce,
    subagent.resume_subagent, subagent.list_subagent_sessions,
    jobs.spawn_subagent_async, jobs.spawn_subagents_parallel_async,
    jobs.spawn_map_reduce_async,
    jobs.check_job, jobs.wait_for_jobs, jobs.list_jobs,
]

BRIEFING_TOOLS: list[BaseTool] = [
    briefing.set_briefing, briefing.get_briefing,
]

EXTRACT_TOOLS: list[BaseTool] = [
    extract_tools.extract_findings, extract_tools.findings_search, extract_tools.findings_stats,
]

DOSSIER_TOOLS: list[BaseTool] = [
    dossier.dossier_note, dossier.dossier_get, dossier.dossier_set_active,
    dossier.dossier_targets, dossier.dossier_prune,
]

WORKFLOW_TOOLS: list[BaseTool] = [
    workflows.workflow_save, workflows.workflow_list, workflows.workflow_get,
    workflows.workflow_remove, workflows.workflow_run,
]

AUTOMATION_TOOLS: list[BaseTool] = [
    automations.automation_schedule, automations.automation_list, automations.automation_cancel,
    automations.automation_run_now, automations.automation_logs,
]

POLICY_TOOLS: list[BaseTool] = [
    permissions.permission_list, hooks.hook_list,
]

TASK_TOOLS: list[BaseTool] = [
    tasks.todo_write, tasks.todo_read, tasks.todo_build,
]

BROWSER_TOOLS: list[BaseTool] = [
    browser_tools.browser_open, browser_tools.browser_goto, browser_tools.browser_click,
    browser_tools.browser_fill, browser_tools.browser_press_key, browser_tools.browser_get_text,
    browser_tools.browser_snapshot, browser_tools.browser_screenshot, browser_tools.browser_evaluate,
    browser_tools.browser_console_logs, browser_tools.browser_list, browser_tools.browser_close,
]

SSH_TOOLS: list[BaseTool] = [
    ssh_tools.ssh_run, ssh_tools.ssh_upload, ssh_tools.ssh_download,
]

VERIFY_TOOLS: list[BaseTool] = [
    verify_tools.compare_models, verify_tools.verify_finding,
]

CORE_TOOLS: list[BaseTool] = _dedupe_tools(
    FILESYSTEM_TOOLS,
    TERMINAL_GIT_TOOLS,
    MEMORY_TOOLS,
    TASK_TOOLS,
    BRIEFING_TOOLS,
    # Dossier write-back is core on purpose: a sub-agent must be able to record what it learns
    # in ANY profile, or the knowledge dies with the run and the whole point is lost.
    DOSSIER_TOOLS,
)

ALL_TOOLS: list[BaseTool] = _dedupe_tools(
    FILESYSTEM_TOOLS,
    BRIEFING_TOOLS,
    DOSSIER_TOOLS,
    EXTRACT_TOOLS,
    MEMORY_TOOLS,
    RAG_TOOLS,
    TERMINAL_GIT_TOOLS,
    WEB_TOOLS,
    GRAPHQL_TOOLS,
    PROCESS_TOOLS,
    MCP_TOOLS,
    DELEGATION_TOOLS,
    WORKFLOW_TOOLS,
    AUTOMATION_TOOLS,
    POLICY_TOOLS,
    TASK_TOOLS,
    BROWSER_TOOLS,
    SSH_TOOLS,
    VERIFY_TOOLS,
)

# Backward-compatible name for callers that import the flat registry directly.
TOOLS: list[BaseTool] = ALL_TOOLS
TOOL_BY_NAME: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


TOKEN_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "with", "you", "your",
}


def _tokenize(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower().replace("_", " "))
        if token not in TOKEN_STOPWORDS
    }


def _tool_text(tool: BaseTool) -> str:
    args_text = ""
    try:
        args = getattr(tool, "args", {}) or {}
        if isinstance(args, dict):
            args_text = " ".join(str(k) for k in args.keys())
    except Exception:
        args_text = ""
    return f"{tool.name} {getattr(tool, 'description', '') or ''} {args_text}"


@dataclass(frozen=True)
class ToolGroupProfile:
    name: str
    tools: list[BaseTool]
    aliases: tuple[str, ...]
    dependencies: tuple[str, ...] = ()


SPECIALIST_GROUPS: dict[str, ToolGroupProfile] = {
    "web": ToolGroupProfile(
        name="web",
        tools=WEB_TOOLS,
        aliases=("web search", "internet", "online", "search online", "searx", "latest news", "current"),
    ),
    "browser": ToolGroupProfile(
        name="browser",
        tools=BROWSER_TOOLS,
        aliases=(
            "browser", "playwright", "site", "webpage", "web page", "url", "http", "https",
            "visitor sees", "visually", "visual", "open website", "click", "screenshot", "dom",
            "console log", "form", "button", "page text",
        ),
        dependencies=("process",),
    ),
    "graphql": ToolGroupProfile(
        name="graphql",
        tools=GRAPHQL_TOOLS,
        aliases=(
            "graphql", "graph ql", "introspection", "schema", "api schema", "query schema",
            "type map", "api type", "type fields", "mutation", "subscription", "endpoint fields",
        ),
        dependencies=("rag",),
    ),
    "rag": ToolGroupProfile(
        name="rag",
        tools=RAG_TOOLS,
        aliases=(
            "rag", "semantic search", "semantic", "ingest", "index", "chunks", "collection",
            "large file", "large log", "search document", "knowledge base", "vector",
        ),
    ),
    "process": ToolGroupProfile(
        name="process",
        tools=PROCESS_TOOLS,
        aliases=("process instance", "instance", "instances", "caido", "logs", "start process", "background process"),
    ),
    "mcp": ToolGroupProfile(
        name="mcp",
        tools=MCP_TOOLS,
        aliases=(
            "mcp", "tool cache", "tools cache", "server registry", "callable capabilities",
            "registered server", "external server", "external tools", "capabilities", "tools list",
        ),
        dependencies=("process",),
    ),
    "delegation": ToolGroupProfile(
        name="delegation",
        tools=DELEGATION_TOOLS,
        aliases=("subagent", "sub agent", "delegate", "parallel agents", "parallel work", "spawn agent"),
    ),
    "workflow": ToolGroupProfile(
        name="workflow",
        tools=WORKFLOW_TOOLS,
        aliases=("workflow", "runbook", "reusable command", "command sequence", "reusable steps", "save steps", "replay"),
    ),
    "automation": ToolGroupProfile(
        name="automation",
        tools=AUTOMATION_TOOLS,
        aliases=(
            "automation", "schedule", "recurring", "repeat", "each morning", "every morning",
            "daily", "hourly", "nightly", "run every", "monitor", "reminder",
        ),
        dependencies=("workflow",),
    ),
    "policy": ToolGroupProfile(
        name="policy",
        tools=POLICY_TOOLS,
        aliases=("permission", "policy", "hook", "pretooluse", "posttooluse", "approval rule"),
    ),
    "ssh": ToolGroupProfile(
        name="ssh",
        tools=SSH_TOOLS,
        aliases=(
            "ssh", "remote host", "bastion", "remote server", "copy to host", "upload to host",
            "download from host", "scp", "remote shell",
        ),
    ),
    "verify": ToolGroupProfile(
        name="verify",
        tools=VERIFY_TOOLS,
        aliases=("compare models", "second model", "cross model", "verify with another model",
                 "independent model", "false positive", "verify finding", "triage", "exploitable"),
    ),
    "extract": ToolGroupProfile(
        name="extract",
        tools=EXTRACT_TOOLS,
        aliases=("secret", "secrets", "endpoint", "endpoints", "hardcoded", "api key", "credential",
                 "bundle", "sourcemap", "findings", "extract", "scan for", "graphql operation"),
    ),
    "dossier": ToolGroupProfile(
        name="dossier",
        tools=DOSSIER_TOOLS,
        aliases=("dossier", "target knowledge", "what do we know", "known endpoints", "target intel",
                 "record fact", "remember about target"),
    ),
}


def _alias_score(text: str, prompt_tokens: set[str], alias: str) -> float:
    alias_l = alias.lower()
    alias_tokens = _tokenize(alias_l)
    if not alias_tokens:
        return 0.0
    if len(alias_tokens) > 1 and alias_l in text:
        return 4.0 + min(2.0, len(alias_tokens) * 0.25)
    overlap = len(prompt_tokens & alias_tokens)
    if len(alias_tokens) > 1 and overlap == len(alias_tokens):
        return 3.0
    if overlap:
        return min(1.5, overlap / len(alias_tokens))
    return 0.0


def _score_group(profile: ToolGroupProfile, text: str, prompt_tokens: set[str]) -> float:
    score = sum(_alias_score(text, prompt_tokens, alias) for alias in profile.aliases)
    for tool in profile.tools:
        overlap = prompt_tokens & _tokenize(_tool_text(tool))
        if overlap:
            score += min(2.0, len(overlap) * 0.35)

    if profile.name == "browser" and re.search(r"https?://|www\.|[a-z0-9-]+\.[a-z]{2,}", text):
        score += 4.0
    if profile.name == "ssh" and {"copy", "archive", "host"} <= prompt_tokens:
        score += 4.0
    if profile.name == "automation" and prompt_tokens & {"repeat", "daily", "hourly", "morning", "nightly", "recurring"}:
        score += 3.0
    if profile.name == "graphql" and {"api", "type", "fields"} <= prompt_tokens:
        score += 3.5
    if profile.name == "mcp" and {"callable", "capabilities", "server"} <= prompt_tokens:
        score += 3.5
    return score


def _expand_group_dependencies(group_names: set[str]) -> set[str]:
    expanded = set(group_names)
    changed = True
    while changed:
        changed = False
        for name in list(expanded):
            profile = SPECIALIST_GROUPS.get(name)
            if not profile:
                continue
            for dep in profile.dependencies:
                if dep not in expanded:
                    expanded.add(dep)
                    changed = True
    return expanded


def select_tools(prompt: str = "", profile: str | None = None) -> list[BaseTool]:
    """Return the tool set to bind for one model call.

    Profiles:
    - ``auto``/``smart`` (default): core tools plus locally scored specialist groups.
    - ``core``/``lite``/``lean``: filesystem, terminal/git, memory, and tasks.
    - ``full``/``all``: every tool, matching the original behavior.
    """
    requested = (profile or os.environ.get("HARNESS_TOOL_PROFILE") or "auto").strip().lower()
    if requested in {"full", "all"}:
        return ALL_TOOLS
    if requested in {"core", "lite", "lean"}:
        return CORE_TOOLS

    text = str(prompt or "").lower()
    if _contains_any(text, ("all tools", "full tool", "full tools", "entire tool catalog")):
        return ALL_TOOLS

    prompt_tokens = _tokenize(text)
    threshold = float(os.environ.get("HARNESS_TOOL_SCORE_THRESHOLD", "3.0"))
    max_groups = max(1, int(os.environ.get("HARNESS_TOOL_MAX_GROUPS", "6")))
    scored = [
        (name, _score_group(profile, text, prompt_tokens))
        for name, profile in SPECIALIST_GROUPS.items()
    ]
    selected_names = {
        name
        for name, score in sorted(scored, key=lambda item: item[1], reverse=True)[:max_groups]
        if score >= threshold
    }
    selected_names = _expand_group_dependencies(selected_names)
    selected = [CORE_TOOLS]
    selected.extend(SPECIALIST_GROUPS[name].tools for name in SPECIALIST_GROUPS if name in selected_names)
    return _dedupe_tools(*selected)


def _human_texts(messages: list[Any]) -> list[str]:
    out = []
    for message in messages:
        message_type = getattr(message, "type", "")
        if message_type == "human" or message.__class__.__name__ == "HumanMessage":
            out.append(str(getattr(message, "content", "") or ""))
    return out


def select_tools_for_messages(
    messages: list[Any], profile: str | None = None, pin_to_first: bool = False
) -> list[BaseTool]:
    """Select tools from a LangChain message list.

    By default this scores off the most recent human message, which is right for an
    interactive session: the user changes topic and the tool set follows.

    ``pin_to_first`` is for a DELEGATED run, where the first human message is the task
    and every later one is machinery - a retry's rejection-feedback string, or a resume's
    follow-up. Scoring those alone silently drops the task's own tools mid-run (a browser
    task retried with "your previous answer was too short" scores nothing browser-ish and
    comes back with no browser tools bound). Scoring the task and the latest turn together
    keeps the task's tools while still reacting to what the new turn asks for.
    """
    humans = _human_texts(messages)
    if not humans:
        prompt = ""
    elif pin_to_first and len(humans) > 1:
        prompt = humans[0] + "\n\n" + humans[-1]
    else:
        prompt = humans[-1]
    return select_tools(prompt, profile=profile)
