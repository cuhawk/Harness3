"""CLI entry point: REPL loop, slash-command dispatch, and one-shot workflow runner.

Ported from the "# CLI" section of the original monolithic ``main.py``, split so every
tool call goes through its owning domain module (``memory.``, ``mcp_bridge.``,
``instances.``, ``rag.``, ``terminal_ui.``, ``shell_git.``, ``fs_tools.``, ``workflows.``,
``automations.``) instead of calling a bare module-level function.

KNOWN DEVIATION: ``harness_agent/memory.py`` does not exist yet in this split (see
``tools_registry.py``'s module docstring, which omits importing it for the same reason).
``handle_memory_cli`` below still does ``from . import memory`` and calls
``memory.memory_list`` / ``memory.memory_search`` / ``memory.memory_search_debug_fixes`` /
``memory.memory_get`` / ``memory.memory_save`` / ``memory.memory_delete`` /
``memory.memory_promote_from_session`` exactly as instructed, mirroring the original
main.py's memory CLI subcommands and signatures (with memory_save's new "kind" argument
left unset from this CLI shortcut so it keeps using memory.memory_save's own default kind).
This import will raise ModuleNotFoundError at call time (not at parse time - ast/py_compile
syntax checks still pass) until memory.py is written by another task in this split.
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import shlex
import sys
from typing import Any

from . import agent_runtime
from . import automations
from . import core
from . import fs_tools
from . import hooks
from . import instances
from . import mcp_bridge
from . import model_diagnostics
from . import permissions
from . import rag
from . import sessions
from . import tasks
from . import terminal_ui
from . import workflows
from .agent_runtime import AgentRuntime
from .terminal_ui import ui_print

# Process-wide session manager: every session activity (new/switch/resume/close) and every
# chat turn flows through this single instance so history is never silently unsaved - see
# handle_turn's invoke/invoke_streaming call sites and main()'s first-session bootstrap.
SESSION_MANAGER = sessions.SessionManager()

# =============================================================================
# CLI command handlers
# =============================================================================


def handle_memory_cli(command_input: str) -> str:
    from . import memory

    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    if len(parts) == 1:
        return memory.memory_list.invoke({})
    sub = parts[1].lower()
    if sub == "search":
        return memory.memory_search.invoke({"query": " ".join(parts[2:])})
    if sub == "debug":
        return memory.memory_search_debug_fixes.invoke({"query": " ".join(parts[2:])})
    if sub == "get" and len(parts) >= 3:
        return memory.memory_get.invoke({"memory_id": parts[2]})
    if sub == "save" and len(parts) >= 3:
        return memory.memory_save.invoke({"content": " ".join(parts[2:])})
    if sub == "delete" and len(parts) >= 3:
        return memory.memory_delete.invoke({"memory_id": parts[2]})
    if sub == "promote" and len(parts) >= 3:
        return memory.memory_promote_from_session.invoke({"memory_id": parts[2]})
    return core.to_json({
        "ok": False,
        "usage": [
            "/memory",
            "/memory search QUERY",
            "/memory debug QUERY",
            "/memory get MEMORY_ID",
            "/memory save TEXT",
            "/memory delete MEMORY_ID",
            "/memory promote MEMORY_ID",
        ],
    })


def handle_mcp_cli(command_input: str) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    cmd = parts[0].lower() if parts else "/mcp"
    if cmd == "/mcp":
        return core.to_json({
            "ok": True,
            "commands": {
                "/mcp-list": "List configured MCP servers, including disabled servers.",
                "/mcp-add-playwright [name] [extra args...]": "Add Playwright MCP stdio server using npx -y @playwright/mcp@latest.",
                "/mcp-start NAME": "Start configured stdio server or initialize HTTP server.",
                "/mcp-stop NAME": "Stop stdio server or clear HTTP session.",
                "/mcp-refresh [NAME]": "Refresh cached tools from running/reachable servers.",
                "/mcp-search QUERY": "Search cached tool names/descriptions.",
                "/mcp-call SERVER TOOL [JSON]": "Call an MCP tool with optional JSON arguments.",
                "/mcp-logs NAME": "Show recent stdio MCP logs.",
                "/mcp-remove NAME": "Remove server config and stop it first.",
                "/mcp-autostart NAME true|false": "Toggle autostart before mcp_call_tool.",
            },
        })
    if cmd == "/mcp-list":
        return mcp_bridge.mcp_list_servers.invoke({"show_disabled": True})
    if cmd == "/mcp-add-playwright":
        name = parts[1] if len(parts) >= 2 else "playwright"
        extra = parts[2:] if len(parts) > 2 else []
        args = ["-y", "@playwright/mcp@latest", *extra]
        return mcp_bridge.mcp_add_server.invoke({"name": name, "command": "npx", "args_json": json.dumps(args), "transport": "stdio", "cwd": ".", "description": "Playwright MCP server"})
    if cmd == "/mcp-start" and len(parts) >= 2:
        return mcp_bridge.mcp_start_server.invoke({"name": parts[1]})
    if cmd == "/mcp-stop" and len(parts) >= 2:
        return mcp_bridge.mcp_stop_server.invoke({"name": parts[1]})
    if cmd == "/mcp-refresh":
        return mcp_bridge.mcp_refresh_tools.invoke({"name": parts[1] if len(parts) >= 2 else ""})
    if cmd == "/mcp-search" and len(parts) >= 2:
        return mcp_bridge.mcp_search_tools.invoke({"query": " ".join(parts[1:])})
    if cmd == "/mcp-call" and len(parts) >= 3:
        arguments_json = parts[3] if len(parts) >= 4 else "{}"
        return mcp_bridge.mcp_call_tool.invoke({"server_name": parts[1], "tool_name": parts[2], "arguments_json": arguments_json})
    if cmd == "/mcp-logs" and len(parts) >= 2:
        return mcp_bridge.mcp_server_logs.invoke({"name": parts[1]})
    if cmd == "/mcp-remove" and len(parts) >= 2:
        return mcp_bridge.mcp_remove_server.invoke({"name": parts[1]})
    if cmd == "/mcp-autostart" and len(parts) >= 3:
        name = core.safe_name(parts[1])
        value = parts[2].lower() in {"1", "true", "yes", "on", "enable", "enabled"}
        servers = mcp_bridge.mcp_load_servers()
        if name not in servers:
            return core.to_json({"ok": False, "error": "Unknown MCP server", "name": name})
        servers[name]["autostart"] = value
        servers[name]["updated_at"] = core.now_str()
        mcp_bridge.mcp_save_servers(servers)
        return core.to_json({"ok": True, "name": name, "autostart": value})
    return core.to_json({"ok": False, "error": "Invalid MCP command. Use /mcp for help."})


def handle_workflow_cli(command_input: str) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    cmd = parts[0].lower() if parts else "/workflow"
    if cmd == "/workflow-list":
        return workflows.workflow_list.invoke({})
    if cmd == "/workflow-save" and len(parts) >= 3:
        name = parts[1]
        # shlex-split (same as handle_mcp_cli) so quoted segments in the template survive.
        prompt_template = " ".join(parts[2:])
        return workflows.workflow_save.invoke({"name": name, "prompt_template": prompt_template})
    if cmd == "/workflow-run" and len(parts) >= 2:
        name = parts[1]
        variables: dict[str, str] = {}
        for token in parts[2:]:
            if "=" in token:
                key, _, value = token.partition("=")
                variables[key] = value
        return workflows.workflow_run.invoke({"name": name, "variables_json": json.dumps(variables)})
    if cmd == "/workflow-remove" and len(parts) >= 2:
        return workflows.workflow_remove.invoke({"name": parts[1]})
    return core.to_json({
        "ok": False,
        "usage": [
            "/workflow-list",
            "/workflow-save NAME PROMPT_TEMPLATE...",
            "/workflow-run NAME [k=v k2=v2 ...]",
            "/workflow-remove NAME",
        ],
    })


def handle_automation_cli(command_input: str) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    cmd = parts[0].lower() if parts else "/automation"
    if cmd == "/automation-list":
        return automations.automation_list.invoke({})
    if cmd == "/automation-schedule" and len(parts) >= 4:
        name = parts[1]
        workflow_name = parts[2]
        minutes = parts[3]
        variables: dict[str, str] = {}
        for token in parts[4:]:
            if "=" in token:
                key, _, value = token.partition("=")
                variables[key] = value
        try:
            interval_minutes = float(minutes)
        except ValueError:
            return core.to_json({"ok": False, "error": f"Invalid MINUTES value: {minutes}"})
        return automations.automation_schedule.invoke({
            "name": name,
            "workflow_name": workflow_name,
            "interval_minutes": interval_minutes,
            "variables_json": json.dumps(variables),
        })
    if cmd == "/automation-cancel" and len(parts) >= 2:
        return automations.automation_cancel.invoke({"name": parts[1]})
    if cmd == "/automation-logs" and len(parts) >= 2:
        return automations.automation_logs.invoke({"name": parts[1]})
    return core.to_json({
        "ok": False,
        "usage": [
            "/automation-list",
            "/automation-schedule NAME WORKFLOW_NAME MINUTES [k=v ...]",
            "/automation-cancel NAME",
            "/automation-logs NAME",
        ],
    })


def handle_permissions_cli(command_input: str) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    sub = parts[1].lower() if len(parts) >= 2 else ""

    if len(parts) == 1 or sub == "list":
        return permissions.permission_list.invoke({})
    if sub == "set" and len(parts) >= 4:
        return core.to_json(permissions.permission_set(parts[2], parts[3]))
    if sub == "add-rule" and len(parts) >= 5:
        return core.to_json(permissions.permission_add_rule(parts[2], parts[3], parts[4]))
    if sub == "remove-rule" and len(parts) >= 4:
        return core.to_json(permissions.permission_remove_rule(parts[2], parts[3]))
    return core.to_json({
        "ok": False,
        "usage": [
            "/permission",
            "/permission list",
            "/permission set TOOL_NAME allow|ask|deny",
            "/permission add-rule TOOL_NAME REGEX allow|ask|deny",
            "/permission remove-rule TOOL_NAME REGEX",
        ],
    })


def handle_hooks_cli(command_input: str) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    sub = parts[1].lower() if len(parts) >= 2 else ""

    if len(parts) == 1 or sub == "list":
        return hooks.hook_list.invoke({})
    if sub == "add" and len(parts) >= 6:
        event = parts[2]
        matcher_regex = parts[3]
        try:
            timeout_sec = int(parts[4])
        except ValueError:
            return core.to_json({"ok": False, "error": f"Invalid TIMEOUT_SEC value: {parts[4]}"})
        # Rejoin the remaining tokens (rather than taking a single token) since the shell
        # command being registered can itself contain spaces and quoting.
        command = " ".join(parts[5:])
        try:
            return core.to_json(hooks.hook_add(event, matcher_regex, command, timeout_sec))
        except ValueError as e:
            return core.to_json({"ok": False, "error": str(e)})
    if sub == "remove" and len(parts) >= 3:
        return core.to_json(hooks.hook_remove(parts[2]))
    if sub in {"enable", "disable"} and len(parts) >= 3:
        return core.to_json(hooks.hook_set_enabled(parts[2], sub == "enable"))
    return core.to_json({
        "ok": False,
        "usage": [
            "/hooks",
            "/hooks list",
            "/hooks add EVENT MATCHER_REGEX TIMEOUT_SEC COMMAND...",
            "/hooks remove HOOK_ID",
            "/hooks enable HOOK_ID",
            "/hooks disable HOOK_ID",
        ],
    })


def handle_model_status_cli(command_input: str, runtime: AgentRuntime) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    raw_json = "--json" in parts
    args = [part for part in parts[1:] if part != "--json"]
    target = args[0] if args else runtime.model_name
    if target.lower() == "all":
        result = model_diagnostics.model_status_report(runtime.model_name, runtime.history, all_models=True)
    else:
        result = model_diagnostics.model_status_report(target, runtime.history, all_models=False)
    return result if raw_json else model_diagnostics.format_model_status_table(result)


def handle_model_test_cli(command_input: str, runtime: AgentRuntime) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    target = parts[1] if len(parts) >= 2 else runtime.model_name
    if target.lower() == "all":
        return model_diagnostics.model_test_report(runtime.model_name, all_models=True)
    return model_diagnostics.model_test_report(target, all_models=False)


def _resolve_session_target(target_text: str) -> sessions.Session | None:
    """Exact id match, else exact name match, among sessions the manager already knows
    about (its persisted store listing) - used by /session switch and /session resume."""
    target_text = target_text.strip()
    if not target_text:
        return None
    for s in SESSION_MANAGER.list_sessions():
        if s.id == target_text:
            return s
    for s in SESSION_MANAGER.list_sessions():
        if s.name == target_text:
            return s
    return None


def handle_session_cli(command_input: str) -> str:
    parts = shlex.split(command_input, posix=(platform.system().lower() != "windows"))
    sub = parts[1].lower() if len(parts) >= 2 else ""

    if sub == "new":
        name = " ".join(parts[2:]) if len(parts) >= 3 else ""
        session, _runtime = SESSION_MANAGER.new_session(name, core.DEFAULT_MODEL)
        tasks.set_active_scope(session.id)
        return core.to_json({"ok": True, "session": session.to_dict()})

    if sub == "list":
        return core.to_json({"ok": True, "sessions": [s.to_dict() for s in SESSION_MANAGER.list_sessions()]})

    if sub in {"switch", "resume"} and len(parts) >= 3:
        target_text = " ".join(parts[2:])
        target = _resolve_session_target(target_text)
        if target is None:
            return core.to_json({"ok": False, "error": f"No session matches id or name: {target_text}"})
        session, _runtime = SESSION_MANAGER.resume_session(target.id)
        tasks.set_active_scope(session.id)
        return core.to_json({"ok": True, "session": session.to_dict()})

    if sub == "rename" and len(parts) >= 3:
        new_name = " ".join(parts[2:])
        renamed = SESSION_MANAGER.rename_active(new_name)
        if renamed is None:
            return core.to_json({"ok": False, "error": "No active session"})
        return core.to_json({"ok": True, "session": renamed.to_dict()})

    if sub == "close":
        active = SESSION_MANAGER.active_session()
        if active is None:
            return core.to_json({"ok": False, "error": "No active session"})
        closing_session, _runtime = active
        SESSION_MANAGER.close_session(closing_session.id)

        # Never leave the app with nothing active: resume the most-recently-updated other
        # persisted session, or create a brand new one if none exist.
        if SESSION_MANAGER.active_session() is None:
            candidates = [s for s in SESSION_MANAGER.list_sessions() if s.id != closing_session.id]
            if candidates:
                new_session, _rt = SESSION_MANAGER.resume_session(candidates[0].id)
            else:
                new_session, _rt = SESSION_MANAGER.new_session("", core.DEFAULT_MODEL)
            tasks.set_active_scope(new_session.id)
            return core.to_json({"ok": True, "closed": closing_session.id, "now_active": new_session.to_dict()})
        return core.to_json({"ok": True, "closed": closing_session.id})

    return core.to_json({
        "ok": False,
        "usage": [
            "/session new [NAME]",
            "/session list",
            "/session switch ID_OR_NAME",
            "/session resume ID_OR_NAME",
            "/session rename NEW_NAME",
            "/session close",
        ],
    })


# =============================================================================
# Non-interactive one-shot mode (for external schedulers: Task Scheduler / cron)
# =============================================================================

def _parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-workflow", dest="run_workflow", default=None)
    parser.add_argument("--vars", dest="vars", default=None)
    args, _unknown = parser.parse_known_args(argv)
    return args


def _parse_vars_arg(vars_arg: str | None) -> dict[str, str]:
    variables: dict[str, str] = {}
    if not vars_arg:
        return variables
    for token in vars_arg.split(","):
        token = token.strip()
        if not token or "=" not in token:
            continue
        key, _, value = token.partition("=")
        variables[key.strip()] = value.strip()
    return variables


def run_one_shot_workflow(run_workflow: str, vars_arg: str | None) -> None:
    """Run a single saved workflow to completion, print the JSON result, and return.

    Used by external schedulers (Windows Task Scheduler / cron) that need a workflow to
    execute once per process invocation, since in-process automations (automation_schedule)
    only keep running while this REPL stays open. Does NOT start the interactive loop and
    does NOT call automations.autostart_all() - a one-shot process should not arm background
    timers it cannot outlive.
    """
    permissions.mark_background()
    core.ensure_dirs()
    runtime = AgentRuntime(core.DEFAULT_MODEL)
    workflows.set_runtime_provider(lambda: runtime)
    variables = _parse_vars_arg(vars_arg)
    result = workflows.workflow_run.invoke({"name": run_workflow, "variables_json": json.dumps(variables)})
    ui_print(result)


# =============================================================================
# Main REPL
# =============================================================================

def handle_turn(runtime: AgentRuntime, user_input: str) -> bool:
    """Dispatch one submitted line: a slash command, or a real chat turn.

    Returns True if the session should stop (an /exit-equivalent was entered) - the plain
    input()-loop fallback in main() breaks on True; ChatApp.on_chat_input_submitted's worker
    calls self.exit() on True. Assumes user_input may still need stripping/emptiness-checking
    (both callers also skip blank input themselves before invoking this, but the check is
    kept here too since it's cheap and this function's contract shouldn't depend on callers
    getting that pre-check right).
    """
    command_input = user_input.strip()

    hook_result = hooks.run_hooks("UserPromptSubmit", {"text": command_input})
    if hook_result.get("blocked"):
        ui_print(f"blocked by hook: {hook_result.get('reason', '')}")
        return False

    if not command_input:
        return False

    if command_input.lower() in {"/exit", "exit", "quit"}:
        ui_print("bye")
        return True

    if command_input == "/workspace":
        ui_print(str(core.WORKSPACE))
        return False

    if command_input.startswith("/workspace "):
        new_path = command_input.removeprefix("/workspace ").strip()
        if not new_path:
            ui_print("usage: /workspace PATH")
            return False
        ui_print(fs_tools.switch_workspace.invoke({"path": new_path}))
        return False

    if command_input == "/tools":
        for t in agent_runtime.TOOLS:
            ui_print(f"- {t.name}: {t.description}")
        return False

    if command_input == "/instances":
        ui_print(instances.list_instances.invoke({}))
        return False

    if command_input == "/rag":
        ui_print(rag.rag_list_collections.invoke({}))
        return False

    if command_input == "/ui":
        ui_print(terminal_ui.terminal_ui_info())
        return False

    if command_input == "/clear":
        terminal_ui.clear_terminal_screen()
        return False

    if command_input.startswith("/memory"):
        ui_print(handle_memory_cli(command_input))
        return False

    if command_input == "/mcp" or command_input.startswith("/mcp-"):
        ui_print(handle_mcp_cli(command_input))
        return False

    if command_input.startswith("/workflow"):
        ui_print(handle_workflow_cli(command_input))
        return False

    if command_input.startswith("/automation"):
        ui_print(handle_automation_cli(command_input))
        return False

    if command_input == "/permission" or command_input.startswith("/permission "):
        ui_print(handle_permissions_cli(command_input))
        return False

    if command_input == "/hooks" or command_input.startswith("/hooks "):
        ui_print(handle_hooks_cli(command_input))
        return False

    if command_input == "/todo":
        ui_print(tasks.todo_read.invoke({}))
        return False

    if command_input == "/session" or command_input.startswith("/session "):
        ui_print(handle_session_cli(command_input))
        return False

    if command_input == "/reset":
        runtime.reset()
        from . import memory

        clear_result = memory.memory_clear_session.invoke({})
        ui_print(f"history reset (session memory cleared: {clear_result})")
        return False

    if command_input == "/model":
        ui_print(f"active model: {runtime.model_name}")
        return False

    if command_input.startswith("/model "):
        new_model = command_input.removeprefix("/model ").strip()
        if not new_model:
            ui_print("usage: /model MODEL_NAME")
            return False
        runtime.switch_model(new_model)
        ui_print(f"switched model to: {runtime.model_name}")
        return False

    if command_input == "/models":
        try:
            models = agent_runtime.list_proxy_models()
            if not models:
                ui_print("No models returned by proxy.")
            for model in models:
                marker = "*" if model == runtime.model_name else " "
                ui_print(f" {marker} {model}")
        except Exception as e:
            ui_print(f"failed to list models: {repr(e)}")
        return False

    if command_input == "/model-status" or command_input.startswith("/model-status "):
        ui_print(handle_model_status_cli(command_input, runtime))
        return False

    if command_input == "/model-test" or command_input.startswith("/model-test "):
        ui_print(handle_model_test_cli(command_input, runtime))
        return False

    if command_input in {"/thinking", "/stream"}:
        flag_name = "SHOW_THINKING" if command_input == "/thinking" else "STREAM_RESPONSES"
        ui_print(f"{command_input.lstrip('/')}: {'on' if getattr(core, flag_name) else 'off'}")
        return False

    if command_input.startswith("/thinking ") or command_input.startswith("/stream "):
        name, _, value = command_input.partition(" ")
        flag_name = "SHOW_THINKING" if name == "/thinking" else "STREAM_RESPONSES"
        value = value.strip().lower()
        if value not in {"on", "off", "true", "false", "1", "0"}:
            ui_print(f"usage: {name} on|off")
            return False
        setattr(core, flag_name, value in {"on", "true", "1"})
        ui_print(f"{name.lstrip('/')}: {'on' if getattr(core, flag_name) else 'off'}")
        return False

    try:
        active = SESSION_MANAGER.active_session()
        if active is None:
            raise RuntimeError("no active session")
        session, session_runtime = active

        # session.id threaded through every render/output call below so a turn started in
        # one tab always writes into THAT tab's scrollback, even if the user switches to a
        # different tab while it's still running - see terminal_ui._history_container's
        # session_id parameter and tui_app.py's module docstring for why this matters for
        # multi-session tabs. render_thinking/render_tool_use get the same treatment via
        # AgentRuntime.session_id (see agent_runtime.py), not from here.
        terminal_ui.render_user_message(user_input, session_id=session.id)

        if core.STREAM_RESPONSES:
            terminal_ui.stream_start(session_id=session.id)
            on_token = functools.partial(terminal_ui.stream_token, session_id=session.id)
            result = sessions.invoke_and_persist(SESSION_MANAGER.store, session, session_runtime, user_input, on_token)
            terminal_ui.stream_end(session_id=session.id)
        else:
            with terminal_ui.agent_thinking_status("agent running"):
                result = sessions.invoke_and_persist(SESSION_MANAGER.store, session, session_runtime, user_input)
            messages = result.get("messages", [])
            answer_text = "[no final content]"
            from langchain_core.messages import AIMessage

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    answer_text = msg.content
                    break
            terminal_ui.render_assistant_message(answer_text, session_id=session.id)
    except Exception as e:
        ui_print(f"\nerror> {repr(e)}\n")

    return False


def main() -> None:
    cli_args = _parse_cli_args(sys.argv[1:])
    if cli_args.run_workflow:
        run_one_shot_workflow(cli_args.run_workflow, cli_args.vars)
        return

    core.ensure_dirs()
    hooks.run_hooks("SessionStart", {"workspace": str(core.WORKSPACE)})

    session, runtime = SESSION_MANAGER.new_session("", core.DEFAULT_MODEL)
    tasks.set_active_scope(session.id)
    workflows.set_runtime_provider(lambda: runtime)
    autostart_result = automations.autostart_all()
    if autostart_result.get("started"):
        ui_print(f"Autostarted automations: {', '.join(autostart_result['started'])}")

    shells = {
        "PowerShell": None,
        "CMD": None,
        "Bash": None,
    }
    try:
        from . import shell_git

        shells = {
            "PowerShell": shell_git.powershell_executable(),
            "CMD": shell_git.cmd_executable(),
            "Bash": shell_git.bash_executable() or shell_git.wsl_executable(),
        }
    except Exception:
        pass

    if terminal_ui.use_textual_ui():
        from .tui_app import ChatApp

        ChatApp(SESSION_MANAGER, session, shells).run()
        return

    terminal_ui.print_startup(runtime.model_name, shells)
    # Plain (non-Textual) fallback's own confirm-callback registration - mirrors the Textual
    # path's permissions.set_confirm_callback wiring in tui_app.ChatApp.on_mount, right
    # alongside that path's own print_startup call. This path is already single-threaded, so
    # a direct blocking input() prompt (terminal_ui.request_permission_confirmation_plain) is
    # enough - no cross-thread dispatch needed here.
    permissions.set_confirm_callback(terminal_ui.request_permission_confirmation_plain)

    while True:
        try:
            user_input = terminal_ui.read_user_input_plain()
        except (KeyboardInterrupt, EOFError):
            hooks.run_hooks("Stop", {"reason": "interrupt"})
            ui_print("\nbye")
            return

        # Re-derive the active runtime every turn (rather than trusting the local variable
        # to still be current) so a /session switch|resume|close during the previous turn
        # is reflected immediately - see handle_session_cli.
        active = SESSION_MANAGER.active_session()
        if active is not None:
            _, runtime = active

        if handle_turn(runtime, user_input):
            hooks.run_hooks("Stop", {"reason": "exit_command"})
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        ui_print("\nbye")
