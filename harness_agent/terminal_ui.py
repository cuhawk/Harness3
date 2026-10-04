"""Terminal UI: Textual-backed interactive REPL, with a plain input()/print() fallback.

Ported from the "# Terminal UI helpers" and "# CLI" sections of the original monolithic
main.py, then rewritten off prompt_toolkit onto Textual (see harness_agent/tui_app.py).

Why Textual, not prompt_toolkit: prompt_toolkit's PromptSession renders by incrementally
diffing against how many terminal rows it drew last time, then erasing/redrawing relative to
that remembered count. Resizing the terminal mid-edit invalidates that memory with no recovery
path - a long-standing, unresolved upstream bug (python-prompt-toolkit#1933, #932), reproducing
across OSes/terminals. Textual repaints its whole screen from its own widget/layout model every
frame, sized against the CURRENT terminal dimensions - structurally immune to that bug class.

Threading model: three independent classes of thread call into this module's render/output
functions - the current turn's worker (agent_runtime.py's LangGraph nodes call render_thinking/
render_tool_use directly), automations.py's own daemon threads (which run whole turns via
workflows.workflow_run -> runtime.invoke -> the same render functions), and shell_git.py's
subprocess-output-reader threads (which call ui_write directly). None of these have any fixed
relationship to "the app's own thread." So every render/output function routes through
`_dispatch_to_ui`, a single generic bridge built on Textual's `App.call_from_thread` (confirmed:
safe from any thread other than the app's own, raises RuntimeError otherwise, which
`_dispatch_to_ui` treats as "not applicable, fall back"). The startup banner sidesteps this
entirely rather than being an exception to it: build_startup_widget() just builds and returns a
widget, mounted directly by tui_app.ChatApp._build_pane as part of composing each session's
tab - there's no separate post-mount call into "whichever tab happens to be active" the way an
on_mount-time print_startup call would need to be.

History is a container of individually mounted widgets (Static/Markdown/Collapsible) per
session tab, not a RichLog - tool-call/thinking blocks need to be real Collapsible widgets to
be foldable, and assistant replies are rendered through Textual's Markdown widget so headings/
code-fences/lists actually parse instead of showing raw '#'/'```' characters. See
_history_container/_mount_history below.

Anything that depends on the active workspace (e.g. the prompt history file) is read through
``core.NAME`` at call time rather than imported as a frozen value, so a runtime workspace switch
(``core.set_workspace()``) is reflected immediately - see core.py's module docstring for why.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
from typing import Any, TYPE_CHECKING

from . import core
from . import rag
from . import voice

if TYPE_CHECKING:
    from .tui_app import ChatApp

# =============================================================================
# Terminal UI helpers (used by both the Textual path and the plain fallback)
# =============================================================================

_ORIGINAL_PRINT = print
_TERMINAL_LOCK = threading.RLock()

ENABLE_TUI = os.environ.get("AGENT_TUI", "1") != "0"
PROMPT_INPUT_LABEL = os.environ.get("AGENT_PROMPT_LABEL", "❯ ")

USE_RICH = os.environ.get("AGENT_RICH_UI", "1") != "0"

try:
    from rich.console import Console
    from rich.syntax import Syntax

    _RICH_AVAILABLE = True
    # _environ={} is load-bearing, not cosmetic: Rich's Console.size property calls the
    # correct live os.get_terminal_size() first, then UNCONDITIONALLY overrides that with
    # os.environ["COLUMNS"]/["LINES"] if either is set. PowerShell/VS Code's shell
    # integration sets those once when the terminal is created and never updates them on
    # resize, so the default Console() silently keeps using that stale width forever after
    # the first resize. Passing an empty _environ removes that override. Only used by the
    # plain (non-Textual) fallback path now - Textual's RichLog manages its own width.
    _console = Console(_environ={})
except ImportError:
    _RICH_AVAILABLE = False
    _console = None


def ui_print(*values: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = True, session_id: str | None = None) -> None:
    """Serialize terminal writes so tool output cannot corrupt concurrent output.

    Routes into the live Textual history log first (when one is running and the caller
    isn't redirecting to an explicit file) - covers every plain ui_print(...) call scattered
    through cli.py's slash-command handlers, not just the render_* functions below.

    session_id: which session's tab to write into. Defaults to None (whichever tab is
    currently active) - most call sites are slash commands that run near-instantly on the
    same worker that already made their own session the active one, so the race window for
    "user switched tabs before this line was written" is negligible there. Pass it explicitly
    from any call site that can run for a while (see handle_turn's chat-turn path) so output
    still lands in the right tab if the user switches away mid-turn.
    """
    if file is None and _dispatch_to_ui(_write_plain_history, sep.join(str(v) for v in values), session_id):
        return
    target = sys.stdout if file is None else file
    with _TERMINAL_LOCK:
        try:
            _ORIGINAL_PRINT(*values, sep=sep, end=end, file=target, flush=flush)
        except UnicodeEncodeError:
            # Some terminals/consoles (legacy codepages, certain non-UTF-8 locales) can't
            # encode characters we print. Degrade to '?' for the unencodable characters
            # instead of crashing the app.
            encoding = getattr(target, "encoding", None) or "ascii"
            safe_values = [
                v.encode(encoding, errors="replace").decode(encoding, errors="replace") if isinstance(v, str) else v
                for v in values
            ]
            _ORIGINAL_PRINT(*safe_values, sep=sep, end=end, file=target, flush=flush)


def ui_write(text_value: Any, end: str = "", flush: bool = True) -> None:
    """Write raw text through the same terminal lock used by ui_print.

    Used for already-line-shaped content (shell_git's per-line subprocess output) - each
    call becomes one line in the Textual history log. NOT used for token-by-token streaming
    (see stream_token below), since RichLog.write() always starts a new line per call - it
    has no "continue the previous line" primitive, so incremental tokens need their own
    dedicated in-progress widget instead.
    """
    text = str(text_value) + (end or "")
    if _dispatch_to_ui(_write_plain_history, text):
        return
    with _TERMINAL_LOCK:
        sys.stdout.write(text)
        if flush:
            sys.stdout.flush()


def ui_status(text_value: str) -> None:
    """Small append-only status line. Keep this out of the prompt render area."""
    ui_print(f"\n[status] {text_value}")


def normalize_prompt_input(text: str) -> str:
    """Normalize terminal newlines without stripping pasted prompt content."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


# =============================================================================
# Textual bridge
# =============================================================================

_APP: "ChatApp | None" = None  # set in ChatApp.on_mount, cleared in ChatApp.on_unmount


def log_tui_error(context: str, exc: BaseException) -> None:
    """Append a timestamped traceback to a plain file, not the (possibly not-yet-rendering)
    Textual UI itself.

    Several of this module's Textual writer functions deliberately swallow exceptions
    (`except Exception: pass`) so a rendering hiccup can't crash a live turn - but that same
    swallowing hides the real cause when something is wrong from the very start (e.g. nothing
    ever appears, not even the startup banner). Call this from those except clauses instead
    of a bare `pass` when diagnosing that class of problem - it's cheap, append-only, and
    never raises itself (a logging failure must never mask the original error).
    """
    import traceback

    try:
        core.ensure_dirs()
        path = core.CACHE_DIR / "tui_errors.log"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- {core.now_str()} [{context}] ---\n")
            f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


# =============================================================================
# Per-session widget id scheme
#
# Single source of truth for how session ids map to widget ids, shared by tui_app.py (which
# builds the widgets) and this module (which looks them up to render into). Session ids are
# raw hex uuids (see sessions.SessionStore.create) and can start with a digit, which Textual
# rejects as a widget id on its own (identifiers must start with a letter/underscore) - hence
# the fixed prefixes below rather than using the bare session id directly.
# =============================================================================

def pane_widget_id(session_id: str) -> str:
    return f"session-{session_id}"


def log_widget_id(session_id: str) -> str:
    return f"log-{session_id}"


def input_widget_id(session_id: str) -> str:
    return f"input-{session_id}"


def session_id_from_pane_id(pane_id: str) -> str:
    return pane_id.removeprefix("session-")


def _dispatch_to_ui(fn, *args: Any, **kwargs: Any) -> bool:
    """Route a UI update to the live Textual app from any thread.

    Returns True if it was routed there (caller should do nothing else), False if the
    caller should fall back to the plain/Rich implementation - which happens when there's
    no live app (Textual not in use, not yet mounted, or already exited), or when this is
    itself called from the app's own thread (call_from_thread raises RuntimeError in that
    case - print_startup handles that one case by calling its Textual writer directly
    instead of going through this function at all; see print_startup below).
    """
    app = _APP
    if app is None:
        return False
    try:
        app.call_from_thread(fn, *args, **kwargs)
        return True
    except RuntimeError:
        return False


def _history_container(session_id: str | None = None):
    """The VerticalScroll belonging to the given session id, or - if no session id is given -
    the one belonging to whichever tab is currently active. Returns None if there's no live
    app, or the target widget can't be resolved (not mounted yet, no active tab, etc).

    History is a container of individually-mounted widgets (Static/Markdown/Collapsible),
    not a RichLog - see this module's docstring for why: collapsible tool-call/thinking
    blocks and a real Markdown-rendered assistant reply both need actual live widgets, which
    a RichLog (a flat scrolling text buffer) can't hold.
    """
    app = _APP
    if app is None:
        return None
    try:
        from textual.containers import VerticalScroll

        target_session_id = session_id if session_id is not None else app.active_session_id()
        if target_session_id is None:
            log_tui_error("_history_container: no resolvable session id (active_session_id() returned None)", RuntimeError(f"session_id arg={session_id!r}"))
            return None
        return app.query_one(f"#{log_widget_id(target_session_id)}", VerticalScroll)
    except Exception as exc:
        log_tui_error(f"_history_container(session_id={session_id!r})", exc)
        return None


async def _mount_history(widget, session_id: str | None = None) -> None:
    """Mount one widget at the bottom of a session's history container and scroll it into
    view. Awaited (not fire-and-forget): Widget.mount() only schedules the mount - awaiting
    it is what guarantees the container's virtual_size already reflects the new widget
    before scroll_end() runs, so the freshly-added content is reliably visible without
    depending on a second, later refresh to catch up. call_from_thread (see _dispatch_to_ui)
    awaits an async callable like this one correctly, blocking the calling thread until it's
    actually done - confirmed against Textual's own call_from_thread implementation.
    """
    container = _history_container(session_id)
    if container is None:
        return
    try:
        await container.mount(widget)
        container.scroll_end(animate=False)
    except Exception as exc:
        log_tui_error("_mount_history", exc)


# =============================================================================
# Slash-command catalogue - single source of truth for autocomplete (tui_app.ChatInput)
# and for anyone wanting a quick reference of what's dispatched in cli.handle_turn.
# =============================================================================

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/workspace", "show or switch the active workspace"),
    ("/tools", "list available tools"),
    ("/instances", "list background instances"),
    ("/rag", "list RAG collections"),
    ("/ui", "show terminal UI info"),
    ("/clear", "clear this tab's scrollback"),
    ("/memory", "memory list/search/save/get/delete"),
    ("/mcp", "MCP server management"),
    ("/mcp-list", "list configured MCP servers"),
    ("/mcp-add-playwright", "add the Playwright MCP server"),
    ("/mcp-start", "start an MCP server"),
    ("/mcp-stop", "stop an MCP server"),
    ("/mcp-refresh", "refresh cached MCP tools"),
    ("/mcp-search", "search cached MCP tools"),
    ("/mcp-call", "call an MCP tool"),
    ("/mcp-logs", "show MCP server logs"),
    ("/mcp-remove", "remove an MCP server"),
    ("/mcp-autostart", "toggle MCP autostart"),
    ("/workflow-list", "list saved workflows"),
    ("/workflow-save", "save a workflow"),
    ("/workflow-run", "run a saved workflow"),
    ("/workflow-remove", "remove a saved workflow"),
    ("/automation-list", "list scheduled automations"),
    ("/automation-schedule", "schedule a workflow to run on an interval"),
    ("/automation-cancel", "cancel a scheduled automation"),
    ("/automation-logs", "show automation run logs"),
    ("/permission", "manage tool permission policy"),
    ("/hooks", "manage lifecycle hooks"),
    ("/todo", "show the current todo list"),
    ("/session", "session new/list/switch/resume/rename/close"),
    ("/reset", "reset chat history + session memory"),
    ("/model", "show or switch the active model"),
    ("/models", "list models available from the LiteLLM proxy"),
    ("/model-status", "show context estimates and proxy/key status"),
    ("/model-test", "send a tiny health-check prompt to one or all models"),
    ("/thinking", "show or toggle tool-call/reasoning visibility"),
    ("/stream", "show or toggle live token streaming"),
    ("/usage", "session token totals; toggle the per-reply tokens/speed footer"),
    ("/exit", "quit"),
]


TODOS_PANEL_ID = "todos-panel"
PERMISSION_OVERLAY_ID = "permission-overlay"


async def _write_plain_history(text: str, session_id: str | None = None) -> None:
    """Mount arbitrary/untrusted text as a literal line - built as a plain rich.text.Text
    (never Text.from_markup) so content that happens to contain '[...]' (JSON arrays,
    user-typed brackets, model output) can never be misparsed as Rich markup tags."""
    from textual.widgets import Static
    from rich.text import Text

    await _mount_history(Static(Text(text)), session_id)


async def _write_markup_history(markup_text: str, session_id: str | None = None) -> None:
    """Mount a hand-crafted, trusted markup string (e.g. "[cyan]-> tool[/]") - parsed via
    Text.from_markup explicitly, since only this module's own hardcoded strings go through
    this path (never raw user/model/tool content - see _write_plain_history for that)."""
    from textual.widgets import Static
    from rich.text import Text

    try:
        await _mount_history(Static(Text.from_markup(markup_text)), session_id)
    except Exception as exc:
        log_tui_error(f"_write_markup_history({markup_text!r})", exc)


# =============================================================================
# Prompt history (shared by the Textual input widget and the plain fallback loop)
# =============================================================================

def append_prompt_history(text: str) -> None:
    if not text or not text.strip():
        return
    try:
        core.ensure_dirs()
        with open(core.PROMPT_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_prompt_history() -> list[str]:
    try:
        path = core.PROMPT_HISTORY_PATH
        if not path.exists():
            return []
        entries: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except (ValueError, TypeError):
                    continue
        return entries
    except Exception:
        return []


# =============================================================================
# Todo/plan panel
#
# Small toggle-visibility strip (same pattern as #streaming/#status: composed directly on
# ChatApp, hidden via CSS display:none, shown/hidden on demand - see tui_app.ChatApp.compose)
# showing the CURRENTLY ACTIVE session's todo list, sourced from tasks.current_todos() - a
# cheap plain-function poll, not a tool call. Re-rendered by ChatApp right after every turn
# finishes and on every tab switch (see ChatApp._turn_finished / on_tabbed_content_tab_
# activated), so it always reflects whichever session's tab is currently active rather than
# a stale list left over from a previously-active session.
# =============================================================================

_TODO_STATUS_GLYPH = {
    "completed": "[x]",
    "in_progress": "[~]",
    "pending": "[ ]",
}


def format_todos(todos: list[dict]) -> str:
    """Pure formatting: one simple checkbox-style line per todo item."""
    if not todos:
        return ""
    lines = [
        f"{_TODO_STATUS_GLYPH.get(item.get('status'), '[ ]')} {item.get('content', '')}"
        for item in todos
    ]
    return "\n".join(lines)


def refresh_todos_panel() -> None:
    """Re-read the active session's todos (tasks.current_todos()) and re-render the todo
    panel widget.

    Only ever called from the app's own thread - both call sites
    (ChatApp._turn_finished and ChatApp.on_tabbed_content_tab_activated) already are - so
    unlike the render_*/stream_* functions above this doesn't need to route through
    _dispatch_to_ui.
    """
    app = _APP
    if app is None:
        return
    from . import tasks
    from textual.widgets import Static

    try:
        panel = app.query_one(f"#{TODOS_PANEL_ID}", Static)
    except Exception:
        return
    text = format_todos(tasks.current_todos())
    panel.update(text)
    panel.display = bool(text)


# =============================================================================
# Permission confirmation overlay (Textual)
#
# Toggle-visibility container - same pattern as #streaming/#status above, deliberately NOT
# a pushed/modal Screen (see tui_app.py's module docstring for why this app never pushes a
# second Screen). request_permission_confirmation_tui is registered as the permissions
# module's confirm callback at app startup (ChatApp.on_mount, right alongside the existing
# terminal_ui._APP / print_startup wiring).
# =============================================================================

_PERMISSION_KEY_DECISIONS = {
    "y": "allow",
    "a": "always_allow",
    "n": "deny",
    "d": "always_deny",
}

# Single shared pending-confirmation slot: permissions.PermissionGate.request already
# serializes concurrent callers behind its own lock (see permissions.py), so only one
# request_permission_confirmation_tui call is ever in flight app-wide - a single module-level
# slot is safe rather than needing something keyed per-call.
_pending_permission_event: "threading.Event | None" = None
_pending_permission_result: list[str | None] = [None]


def format_permission_prompt(tool_name: str, args: dict):
    """Built as a rich.text.Text object (not a markup string) so the args - arbitrary tool
    call arguments, not this module's own trusted text - are appended as a literal segment
    rather than parsed for '[...]'-shaped Rich markup tags. Each key is shown in a colored
    "key cap" (bold text on a solid background) right next to what it does, so the four
    choices read as a legend instead of a plain options line."""
    from rich.text import Text

    args_json = json.dumps(args, ensure_ascii=False)
    trimmed = args_json if len(args_json) <= 300 else args_json[:297] + "..."

    text = Text()
    text.append("Permission requested", style="bold yellow")
    text.append("  ·  tool: ")
    text.append(tool_name, style="bold")
    text.append("\n")
    text.append(trimmed, style="dim")
    text.append("\n\n")
    text.append(" y ", style="bold black on green")
    text.append(" allow once      ")
    text.append(" a ", style="bold black on green")
    text.append(" always allow this tool      ")
    text.append(" n ", style="bold white on red")
    text.append(" deny once      ")
    text.append(" d ", style="bold white on red")
    text.append(" always deny this tool")
    return text


def _active_chat_input():
    app = _APP
    if app is None:
        return None
    try:
        from .tui_app import ChatInput

        session_id = app.active_session_id()
        if session_id is None:
            return None
        return app.query_one(f"#{input_widget_id(session_id)}", ChatInput)
    except Exception:
        return None


def _tui_show_permission_overlay(tool_name: str, args: dict) -> None:
    app = _APP
    if app is None:
        return
    try:
        from textual.widgets import Static

        overlay = app.query_one(f"#{PERMISSION_OVERLAY_ID}", Static)
        overlay.update(format_permission_prompt(tool_name, args))
        overlay.display = True
    except Exception:
        pass
    # Disable the active tab's input while the overlay is up so ordinary typing can't
    # interfere with capturing the four watched key presses (see handle_permission_key) -
    # Widget.disabled also blurs it immediately if it currently holds focus.
    chat_input = _active_chat_input()
    if chat_input is not None:
        chat_input.disabled = True


def _tui_hide_permission_overlay() -> None:
    app = _APP
    if app is None:
        return
    try:
        from textual.widgets import Static

        overlay = app.query_one(f"#{PERMISSION_OVERLAY_ID}", Static)
        overlay.display = False
    except Exception:
        pass
    chat_input = _active_chat_input()
    if chat_input is not None:
        chat_input.disabled = False
        try:
            chat_input.focus()
        except Exception:
            pass


def request_permission_confirmation_tui(tool_name: str, args: dict) -> str:
    """The permissions.ConfirmCallback registered for the Textual UI (see ChatApp.on_mount).

    Called from whatever background thread is running the turn that hit an ask-policy tool
    (agent_runtime.py's run_tool_calls -> permissions.GATE.request) - it MUST block
    that calling thread until a decision is made, while safely updating the UI across
    threads. Follows the exact same cross-thread dispatch bridge every other render/output
    function in this module uses (_dispatch_to_ui, built on Textual's App.call_from_thread):
    a threading.Event plus a small mutable holder for the eventual decision string, an
    overlay-show dispatched through that bridge, then this thread waits on the event with no
    timeout. The four watched key presses (see ChatApp.on_key -> handle_permission_key,
    already running on the app's own thread) write the decision into the shared holder, hide
    the overlay again, and set the event.
    """
    global _pending_permission_event
    event = threading.Event()
    _pending_permission_result[0] = None
    _pending_permission_event = event

    if not _dispatch_to_ui(_tui_show_permission_overlay, tool_name, args):
        # No live Textual app to show the overlay in (shouldn't happen - this callback is
        # only ever registered while the Textual app is running) - fail safe the same way
        # permissions.py already treats "no callback registered" as an immediate denial.
        _pending_permission_event = None
        return "deny"

    event.wait()
    _pending_permission_event = None
    return _pending_permission_result[0] or "deny"


def handle_permission_key(key: str) -> bool:
    """Called from ChatApp.on_key (already on the app's own thread) while the overlay is
    visible. Returns True if the key matched one of the four decisions - the caller should
    stop/prevent-default the event in that case - False if it's not one of the watched keys
    (caller should let it bubble/be ignored as normal)."""
    if _pending_permission_event is None:
        return False
    decision = _PERMISSION_KEY_DECISIONS.get(key)
    if decision is None:
        return False
    _pending_permission_result[0] = decision
    event = _pending_permission_event
    _tui_hide_permission_overlay()
    event.set()
    return True


# =============================================================================
# Plain (non-Textual) input fallback
# =============================================================================

def use_textual_ui() -> bool:
    """Whether the interactive Textual app should be used for this process - a real tty,
    not disabled via AGENT_TUI=0, and Textual actually importable."""
    if not ENABLE_TUI or not sys.stdin.isatty():
        return False
    try:
        import textual  # noqa: F401
    except ImportError:
        return False
    return True


def read_user_input_plain() -> str:
    """Degraded REPL input for when use_textual_ui() is False: no live editing widget, no
    history recall, no border - just input(), matching what happened in exactly these same
    cases (non-tty, AGENT_TUI=0, or Textual missing) before this module moved onto Textual
    for the interactive case.

    The label is printed via ui_print (not input()'s own prompt argument) so its
    UnicodeEncodeError fallback applies - input() writes its prompt straight to stdout with
    no encoding safety net, which crashes on PROMPT_INPUT_LABEL's default "❯" under a legacy
    (non-UTF-8) codepage, e.g. when stdout is piped rather than a real console.
    """
    if core.VOICE_IN:
        # Speak the turn instead of typing it. Falls through to the keyboard on an empty
        # transcript (silence, a mis-hear, or voice having just disabled itself because a
        # dependency is missing) rather than looping on the microphone, so a failing mic can
        # never lock you out of your own REPL.
        spoken = voice.listen()
        if spoken:
            ui_print(f"{PROMPT_INPUT_LABEL}{spoken}")
            return normalize_prompt_input(spoken)
    ui_print(PROMPT_INPUT_LABEL, end="", flush=True)
    return normalize_prompt_input(input())


_PLAIN_PERMISSION_RESPONSES = {
    "y": "allow",
    "yes": "allow",
    "n": "deny",
    "no": "deny",
    "a": "always_allow",
    "always": "always_allow",
    "never": "always_deny",
}


def request_permission_confirmation_plain(tool_name: str, args: dict) -> str:
    """The permissions.ConfirmCallback registered for the plain (non-Textual) REPL fallback
    (see cli.py's main(), right where this fallback's own startup - terminal_ui.print_startup
    - is already called). This path is already single-threaded (the plain REPL loop calls
    handle_turn synchronously on the only thread there is), so a direct blocking input()
    prompt is sufficient - no cross-thread dispatch like request_permission_confirmation_tui
    needs.
    """
    args_json = json.dumps(args, ensure_ascii=False)
    ui_print(f"\n[permission] {tool_name}({args_json[:300]})")
    while True:
        ui_print("Allow this tool call? [y]es / [n]o / [a]lways / never: ", end="", flush=True)
        try:
            response = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "deny"
        decision = _PLAIN_PERMISSION_RESPONSES.get(response)
        if decision is not None:
            return decision
        ui_print("Please answer y(es), n(o), a(lways), or never.")


# =============================================================================
# Startup banner
# =============================================================================

def print_startup(model_name: str, shells: dict[str, str]) -> None:
    """The plain (non-Textual) fallback's startup banner - see cli.py's main(). The Textual
    path doesn't call this at all: each session tab mounts its own build_startup_widget(...)
    directly as part of tui_app.ChatApp._build_pane, instead of a separate post-mount call
    into whichever tab happened to be active."""
    clear_terminal_screen()
    if not (_RICH_AVAILABLE and USE_RICH):
        _print_startup_plain(model_name, shells)
        return
    try:
        _print_startup_rich(model_name, shells)
    except Exception:
        _print_startup_plain(model_name, shells)


def build_startup_widget(model_name: str, session_name: str, shells: dict[str, str]):
    """Compact Claude-Code-style welcome widget, mounted once per session tab (see
    tui_app.ChatApp._build_pane) - replaces the old ~15-row Model/Thinking/Stream/Workspace/
    RAG-DB/MCP-config/... config dump with a short, bordered box: title, a one-line ready
    message, a compact model/thinking/stream line, and a short tips line. Full config detail
    is still available via /ui - it just isn't dumped in front of the user on every tab open.
    """
    from textual.widgets import Static
    from rich.console import Group
    from rich.text import Text

    thinking = "on" if core.SHOW_THINKING else "off"
    stream = "on" if core.STREAM_RESPONSES else "off"
    usage_flag = "on" if core.SHOW_USAGE else "off"
    body = Group(
        Text.assemble(("harness-agent", "bold cyan"), f"  ·  {session_name}"),
        Text(""),
        Text.assemble(("Ready", "bold"), f" in {core.WORKSPACE}"),
        Text.assemble(
            ("Model ", "dim"), model_name,
            ("   Thinking ", "dim"), thinking,
            ("   Stream ", "dim"), stream,
            ("   Usage ", "dim"), usage_flag,
        ),
        Text(""),
        Text(
            "/session new for another tab  ·  Tab to autocomplete  ·  /ui for full config  ·  /exit to quit",
            style="dim",
        ),
    )
    return Static(body, classes="startup-banner")


def _startup_status_table(model_name: str, shells: dict[str, str], terminal_ui_value: str):
    from rich.table import Table
    import platform

    status = Table.grid(padding=(0, 1))
    status.add_column(style="cyan", no_wrap=True, width=14)
    status.add_column(style="white", overflow="fold")
    status.add_row("Model", model_name)
    status.add_row("Thinking", "on" if core.SHOW_THINKING else "off")
    status.add_row("Stream", "on" if core.STREAM_RESPONSES else "off")
    status.add_row("Usage footer", "on" if core.SHOW_USAGE else "off")
    status.add_row("LiteLLM proxy", core.LITELLM_BASE_URL)
    status.add_row("Workspace", str(core.WORKSPACE))
    status.add_row("SearXNG", core.SEARXNG_URL)
    status.add_row("RAG DB", str(core.RAG_DB_PATH))
    status.add_row("Memory DB", str(core.MEMORY_DB_PATH))
    status.add_row("MCP config", str(core.MCP_CONFIG_PATH))
    status.add_row("RAG embedding", rag.rag_embedding_model_name())
    status.add_row("OS", platform.platform())
    status.add_row("Terminal UI", terminal_ui_value)
    status.add_row("PowerShell", shells.get("PowerShell") or "not found")
    status.add_row("CMD", shells.get("CMD") or "not found")
    status.add_row("Bash", shells.get("Bash") or "not found")
    return status


def _print_startup_rich(model_name: str, shells: dict[str, str]) -> None:
    from rich.text import Text

    _console.print(Text("harness-agent", style="bold cyan"))
    _console.print()
    _console.print(_startup_status_table(model_name, shells, "basic input"))
    _console.print()


def _print_startup_plain(model_name: str, shells: dict[str, str]) -> None:
    ui_print(f"LiteLLM proxy: {core.LITELLM_BASE_URL}")
    ui_print(f"Model: {model_name}")
    ui_print(f"Thinking: {'on' if core.SHOW_THINKING else 'off'}")
    ui_print(f"Stream: {'on' if core.STREAM_RESPONSES else 'off'}")
    ui_print(f"Usage footer: {'on' if core.SHOW_USAGE else 'off'}")
    ui_print(f"Workspace: {core.WORKSPACE}")
    ui_print(f"SearXNG: {core.SEARXNG_URL}")
    ui_print(f"RAG DB: {core.RAG_DB_PATH}")
    ui_print(f"Memory DB: {core.MEMORY_DB_PATH}")
    ui_print(f"MCP config: {core.MCP_CONFIG_PATH}")
    ui_print(f"RAG embedding: {rag.rag_embedding_model_name()}")
    import platform

    ui_print(f"OS: {platform.platform()}")
    ui_print("Terminal UI: basic input")
    ui_print(f"Prompt history: {core.PROMPT_HISTORY_PATH}")
    ui_print()
    ui_print("Detected shells:")
    ui_print(f"  PowerShell: {shells.get('PowerShell') or 'not found'}")
    ui_print(f"  CMD:        {shells.get('CMD') or 'not found'}")
    ui_print(f"  Bash:       {shells.get('Bash') or 'not found'}")
    ui_print()


def clear_terminal_screen() -> None:
    # print_startup (the only other caller, besides the /clear slash command inside a turn
    # worker) is now only ever invoked for the non-Textual fallback - _APP is always None
    # there - so _dispatch_to_ui's "no live app" False case is the only way this reaches the
    # plain os.system("cls")/ANSI-clear path below; there's no same-thread case to special-
    # case here anymore.
    if _dispatch_to_ui(_tui_clear_history):
        return
    if os.name == "nt":
        os.system("cls")
    else:
        ui_write("\033[2J\033[H", flush=True)


async def _tui_clear_history() -> None:
    """Remove every mounted widget from the active tab's history container - VerticalScroll
    has no single clear() the way RichLog did, so this drops all children instead (including
    the startup banner; a fresh /clear really does mean an empty scrollback)."""
    container = _history_container()
    if container is None:
        return
    try:
        await container.remove_children()
    except Exception as exc:
        log_tui_error("_tui_clear_history", exc)


def terminal_ui_info() -> str:
    return core.to_json({
        "ok": True,
        "tui_enabled": ENABLE_TUI,
        "tui_backend": "textual" if _APP is not None else ("textual (available)" if use_textual_ui() else "basic input"),
        "stdin_is_tty": sys.stdin.isatty(),
        "history_path": str(core.PROMPT_HISTORY_PATH),
        "prompt_label": PROMPT_INPUT_LABEL,
        "multiline": True,
        "submit": "Enter",
        "newline": ["Ctrl+J", "Shift+Enter (best-effort, terminal-dependent)"],
        "history_recall": ["Ctrl+Up/Ctrl+Down", "Up/Down at start/end of the buffer"],
        "fallback_env": {
            "AGENT_TUI": "0 disables the Textual UI (falls back to basic input())",
            "AGENT_PROMPT_LABEL": "custom prompt label for the basic-input fallback",
        },
    })


# =============================================================================
# Tool-chain / message rendering
# =============================================================================

def _build_content_widget(content: str):
    """Pick the best-fitting Textual widget for a blob of tool args/result content, instead
    of always dumping it as one plain-text block - detected from the content itself since
    this layer has no reliable way to know the originating file type:
    - valid JSON -> syntax-highlighted (rich.syntax.Syntax, "monokai" theme, matches the
      highlighting already used elsewhere in this module)
    - looks like markdown (fenced code block, heading, list, bold) -> Textual's Markdown
      widget, which actually parses headings/code-fences/lists instead of showing raw '#'/
      '```' characters
    - anything else -> plain literal text (never markup-parsed - this is tool/model output,
      not this module's own trusted strings)
    """
    from textual.widgets import Markdown, Static
    from rich.text import Text

    stripped = content.strip()
    if not stripped:
        return Static(Text("(empty)", style="dim"))
    try:
        json.loads(stripped)
        return Static(Syntax(content, "json", theme="monokai", word_wrap=True, background_color="default"))
    except (ValueError, TypeError):
        pass
    markdown_signals = ("```", "\n# ", "\n## ", "\n- ", "\n* ", "\n1. ", "**")
    if stripped.startswith("#") or any(sig in content for sig in markdown_signals):
        return Markdown(content)
    return Static(Text(content))


def _tool_use_title(name: str, args_json: str) -> str:
    compact = args_json if len(args_json) <= 60 else args_json[:57] + "..."
    return f"→ {name}({compact})"


def render_tool_use(name: str, args: dict, content: str, session_id: str | None = None) -> None:
    """One combined, collapsed-by-default entry for a tool invocation (call + result
    together), replacing the old separate render_tool_call/render_tool_result pair - see
    agent_runtime.py's run_tool_calls, the only caller."""
    args_json = json.dumps(args, ensure_ascii=False)
    snippet = content[:4000]
    if _dispatch_to_ui(_write_tool_use_history, name, args_json, snippet, session_id):
        return
    if not (_RICH_AVAILABLE and USE_RICH):
        ui_print(f"\n[tool-chain] {name}({args_json})")
        ui_print(f"[tool-result] {snippet}")
        return
    try:
        _console.print(f"\n[cyan]-> {name}[/]({args_json[:500]})")
        try:
            json.loads(snippet)
            body: Any = Syntax(snippet, "json", theme="monokai", word_wrap=True, background_color="default")
        except (ValueError, TypeError):
            body = snippet
        _console.print("[yellow]<- result[/]")
        _console.print(body)
    except Exception:
        ui_print(f"\n[tool-chain] {name}({args_json})")
        ui_print(f"[tool-result] {snippet}")


async def _write_tool_use_history(name: str, args_json: str, snippet: str, session_id: str | None = None) -> None:
    from textual.widgets import Collapsible, Static
    from rich.text import Text

    try:
        collapsible = Collapsible(
            Static(Text("args", style="dim italic")),
            _build_content_widget(args_json),
            Static(Text("result", style="dim italic")),
            _build_content_widget(snippet),
            title=_tool_use_title(name, args_json),
            collapsed=True,
        )
        await _mount_history(collapsible, session_id)
    except Exception as exc:
        log_tui_error(f"_write_tool_use_history({name!r})", exc)


def render_user_message(text: str, session_id: str | None = None) -> None:
    """Echo the just-submitted prompt as a distinct styled line, the way Claude Code shows
    your message before the response - instead of leaving it as plain terminal scrollback
    that looks identical to any other line. Call this right after a turn starts, before the
    task is run."""
    if _dispatch_to_ui(_write_user_message_history, text, session_id):
        return
    if not (_RICH_AVAILABLE and USE_RICH):
        ui_print(f"\n> {text}")
        return
    try:
        _console.print(f"\n[bold blue]>[/] {text}")
    except Exception:
        ui_print(f"\n> {text}")


async def _write_user_message_history(text: str, session_id: str | None = None) -> None:
    from textual.widgets import Static
    from rich.text import Text

    try:
        line = Text()
        line.append("> ", style="bold blue")
        line.append(text)  # appended as literal text, not parsed as markup - this is the
        # user's own typed content, which could itself contain '[...]'-shaped substrings.
        await _mount_history(Static(line), session_id)
    except Exception as exc:
        log_tui_error("_write_user_message_history", exc)


def render_thinking(text: str, session_id: str | None = None) -> None:
    """Render model reasoning/thinking content (e.g. DeepSeek-R1-style reasoning_content),
    only ever called when core.SHOW_THINKING is on. Distinct dim styling so it visually
    reads as "the model's scratch work", not the final answer."""
    if not text:
        return
    if _dispatch_to_ui(_write_thinking_history, text, session_id):
        return
    if not (_RICH_AVAILABLE and USE_RICH):
        ui_print(f"\n[thinking] {text}")
        return
    try:
        _console.print("\n[dim italic]thinking[/]")
        _console.print(f"[dim]{text}[/]")
    except Exception:
        ui_print(f"\n[thinking] {text}")


async def _write_thinking_history(text: str, session_id: str | None = None) -> None:
    """Collapsed by default, same as render_tool_use - reasoning is useful to have around
    but shouldn't dominate the scrollback the way it would as always-expanded text."""
    from textual.widgets import Collapsible, Static
    from rich.text import Text

    try:
        collapsible = Collapsible(
            Static(Text(text, style="dim")),  # literal text, not markup - this is model output
            title="thinking",
            collapsed=True,
        )
        await _mount_history(collapsible, session_id)
    except Exception as exc:
        log_tui_error("_write_thinking_history", exc)


def render_assistant_message(text: str, session_id: str | None = None) -> None:
    # Speak before rendering, and before the _dispatch_to_ui early return: this is the one
    # funnel BOTH front-ends use for a non-streamed reply, so hooking it here covers the
    # Textual UI and the plain fallback in a single place. voice.say is a no-op when voice
    # output is off, and hands off to a background thread when it isn't, so neither path
    # blocks on synthesis.
    voice.say(text)
    if _dispatch_to_ui(_write_assistant_message_history, text, session_id):
        return
    if not (_RICH_AVAILABLE and USE_RICH):
        ui_print(f"\nassistant> {text}\n")
        return
    try:
        _console.print()
        _console.print("[bold green]assistant[/]")
        _console.print(text)
        _console.print()
    except Exception:
        ui_print(f"\nassistant> {text}\n")


async def _write_assistant_message_history(text: str, session_id: str | None = None) -> None:
    """Rendered via Textual's Markdown widget, not a plain Static - LLM replies are
    typically markdown-shaped (headings, code fences, lists, bold), and Markdown actually
    parses that instead of showing raw '#'/'```' characters. Never collapsed: this is the
    answer, not supporting detail like a tool call or reasoning trace."""
    from textual.widgets import Markdown, Static
    from rich.text import Text

    try:
        await _mount_history(Static(Text.from_markup("\n[bold green]assistant[/]")), session_id)
        await _mount_history(Markdown(text), session_id)  # text itself is never markup-parsed
    except Exception as exc:
        log_tui_error("_write_assistant_message_history", exc)


def render_usage(text: str, session_id: str | None = None) -> None:
    """The per-reply token/speed footer (see usage.Usage.format_line). Dim and unadorned:
    it prints after every single turn, so it has to read as a margin note under the answer
    rather than as another block competing with it."""
    if not text:
        return
    if _dispatch_to_ui(_write_usage_history, text, session_id):
        return
    if not (_RICH_AVAILABLE and USE_RICH):
        ui_print(text)
        return
    try:
        _console.print(f"[dim]{text}[/]")
        _console.print()
    except Exception:
        ui_print(text)


async def _write_usage_history(text: str, session_id: str | None = None) -> None:
    from textual.widgets import Static
    from rich.text import Text

    try:
        await _mount_history(Static(Text(text, style="dim")), session_id)
    except Exception as exc:
        log_tui_error("_write_usage_history", exc)


# =============================================================================
# Streaming
# =============================================================================

_STREAM_TEXT: list[str] = []


def stream_start(session_id: str | None = None) -> None:
    """Header printed once before the first streamed token, matching
    render_assistant_message's styling since the streamed tokens ARE that message,
    just arriving incrementally instead of all at once."""
    _STREAM_TEXT.clear()
    # Reset the speaker's per-message state so this reply picks its own emotional cue instead
    # of inheriting the previous one's.
    voice.begin_stream()
    if _dispatch_to_ui(_tui_stream_start, session_id):
        return
    if _RICH_AVAILABLE and USE_RICH:
        try:
            _console.print()
            _console.print("[bold green]assistant[/]")
            return
        except Exception:
            pass
    ui_print("\nassistant>", end=" ")


async def _tui_stream_start(session_id: str | None = None) -> None:
    app = _APP
    if app is None:
        return
    from rich.text import Text
    from textual.widgets import Static

    try:
        await _mount_history(Static(Text.from_markup("\n[bold green]assistant[/]")), session_id)
    except Exception as exc:
        log_tui_error("_tui_stream_start", exc)
    try:
        streaming = app.query_one("#streaming", Static)
        streaming.update("")
        streaming.display = True
    except Exception:
        pass


def stream_token(text: str, session_id: str | None = None) -> None:
    """Write one incremental token/chunk of a streamed assistant reply, updating the
    in-progress display in place - used by AgentRuntime.invoke_streaming's on_token
    callback. Not routed through ui_write/RichLog directly: RichLog.write() always starts a
    new line per call, so per-token writes there would put every token on its own line
    instead of accumulating readable text.

    session_id is only used here to decide whether the FINAL commit (stream_end) lands in
    the right tab - the #streaming widget itself is a single, app-wide widget (see
    tui_app.ChatApp.compose), not one per session tab, since only one turn ever runs
    app-wide at a time (see tui_app.py's module docstring) - so there's nothing to
    disambiguate for the live in-progress preview itself.
    """
    _STREAM_TEXT.append(text)
    # Feed the synthesiser incrementally rather than waiting for stream_end: voice.feed_stream
    # buffers until a sentence completes, then speaks it, so audio starts about a second into
    # a reply instead of after the model has finished generating the whole thing.
    voice.feed_stream(text)
    if _dispatch_to_ui(_tui_stream_token):
        return
    ui_write(text)


def _tui_stream_token() -> None:
    app = _APP
    if app is None:
        return
    try:
        from textual.widgets import Static

        app.query_one("#streaming", Static).update("".join(_STREAM_TEXT))
    except Exception:
        pass


def stream_end(session_id: str | None = None) -> None:
    # Speak whatever trailing text never reached a sentence boundary (a reply ending without
    # terminal punctuation would otherwise be silently dropped).
    voice.end_stream()
    if _dispatch_to_ui(_tui_stream_end, session_id):
        return
    ui_write("\n\n")


async def _tui_stream_end(session_id: str | None = None) -> None:
    app = _APP
    if app is None:
        return
    from textual.widgets import Markdown, Static

    final_text = "".join(_STREAM_TEXT)
    _STREAM_TEXT.clear()
    try:
        if final_text:
            await _mount_history(Markdown(final_text), session_id)  # never markup-parsed - model output
    except Exception as exc:
        log_tui_error("_tui_stream_end", exc)
    try:
        streaming = app.query_one("#streaming", Static)
        streaming.update("")
        streaming.display = False
    except Exception:
        pass


@contextlib.contextmanager
def agent_thinking_status(label: str = "agent running"):
    """Show a running/thinking indicator while a non-streaming turn is in flight.

    When Textual is active, ChatApp already shows/hides its own status widget around the
    *entire* turn (slash commands included) in tui_app.ChatApp.on_chat_input_submitted /
    _turn_finished - so this is a no-op passthrough in that case, rather than a second,
    possibly-conflicting widget update for what both codepaths would otherwise show for
    the same duration.
    """
    if _APP is not None:
        yield None
        return
    if _RICH_AVAILABLE and USE_RICH:
        try:
            with _console.status(f"[bold cyan]{label}...[/]", spinner="dots") as s:
                yield s
            return
        except Exception:
            pass
    ui_status(label)
    yield None
