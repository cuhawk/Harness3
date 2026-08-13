"""Textual app that owns the interactive terminal UI.

See terminal_ui.py's module docstring for why this replaced prompt_toolkit, and for the
_dispatch_to_ui thread-bridge that lets agent_runtime.py, automations.py, and shell_git.py
keep calling terminal_ui's render_*/ui_* functions unchanged from whatever thread they run
on, without knowing anything about Textual.

Single Screen (composed directly on the App, no push_screen) - deliberately, since pushing a
new Screen right after a resize is a known Textual/Windows-Terminal quirk (a screen pushed in
that window can render at the stale pre-resize size until the next resize event); a
single-screen app never hits that.

Multi-session tabs: the app owns one SessionManager (the same instance cli.py's main()
constructs, passed in here rather than a second one being created - see SESSION_MANAGER in
cli.py) and renders every one of its currently-open sessions as a TabbedContent/TabPane pair,
each pane holding its own uniquely-id'd history container (a VerticalScroll of mounted
Static/Markdown/Collapsible widgets, not a RichLog - see terminal_ui.py's module docstring
for why) and ChatInput. Widget id scheme (shared with terminal_ui.py, defined there as the
single source of truth so both modules agree on it): pane id "session-<session_id>", log id
"log-<session_id>", input id "input-<session_id>".

KNOWN SIMPLIFICATION (per this pass's scope): only one turn runs at a time app-wide
(self.turn_running is a single shared flag, not one per session/tab) - submitting from a
second tab while a turn is in flight just bells like it always did, rather than truly running
sessions concurrently. This keeps SESSION_MANAGER.set_active()/active_session() (which
cli.handle_turn relies on for chat turns and several slash commands) from racing across
threads. Automations/scheduled workflows are unaffected - those already run each session's
AgentRuntime independently of anything in this file.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from textual import events
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static, TabbedContent, TabPane, TextArea

if TYPE_CHECKING:
    from .sessions import Session, SessionManager

_HINT_TEXT = "Enter sends | Ctrl+J or Shift+Enter for a newline | Ctrl+Up/Down for history | Tab to autocomplete | /exit quits"


class ChatInput(TextArea):
    """Multiline input: Enter submits, Ctrl+J/Shift+Enter insert a newline, Ctrl+Up/Down (or
    plain Up/Down at the start/end of the buffer) recall history, Tab accepts/cycles a
    slash-command suggestion - same semantics as the old prompt_toolkit key bindings this
    replaces, with one confirmed difference: Esc-then-Enter (a genuine two-keypress chord in
    prompt_toolkit) isn't reproduced, since Textual reports each keypress as its own event
    with no built-in chord-binding primitive for it, and Ctrl+J already covers the same
    "insert a literal newline" need reliably in every terminal. Shift+Enter is included as a
    single-keypress best-effort check - it only fires on terminals that report it as a
    distinguishable key (most don't, same caveat the old prompt_toolkit binding's comment
    already noted).

    One of these is mounted per open session tab (see ChatApp._build_pane) - session_id says
    which session this particular input belongs to, so ChatApp.on_chat_input_submitted knows
    which session's turn to run without needing to consult "whichever tab is active".
    """

    class Submitted(Message):
        def __init__(self, chat_input: "ChatInput", text: str) -> None:
            self.chat_input = chat_input
            self.text = text
            super().__init__()

    DEFAULT_CSS = """
    ChatInput {
        height: auto;
        max-height: 40%;
    }
    """

    def __init__(self, session_id: str, **kwargs) -> None:
        super().__init__(show_line_numbers=False, **kwargs)
        self.session_id = session_id
        self._history: list[str] = []
        self._history_index: int = 0
        self._draft: str = ""
        self._autocomplete_matches: list[str] = []
        self._autocomplete_index: int = 0

    def on_mount(self) -> None:
        from . import terminal_ui

        self._history = terminal_ui.load_prompt_history()
        self._history_index = len(self._history)

    def remember(self, text: str) -> None:
        """Called by ChatApp right after a submission is recorded to disk, so recall sees
        it without needing to reload the history file."""
        self._history.append(text)
        self._history_index = len(self._history)
        self._draft = ""

    def _move_cursor_to_end(self) -> None:
        last_line = self.document.line_count - 1
        self.move_cursor((last_line, len(self.document[last_line])))

    def _recall(self, direction: int) -> None:
        if not self._history:
            return
        if self._history_index == len(self._history):
            self._draft = self.text
        new_index = max(0, min(len(self._history), self._history_index + direction))
        if new_index == self._history_index:
            return
        self._history_index = new_index
        self.text = self._draft if new_index == len(self._history) else self._history[new_index]
        self._move_cursor_to_end()

    # -------------------------------------------------------------------
    # Slash-command autocomplete
    #
    # Deliberately not the textual-autocomplete package (installed alongside this for
    # reference) - it targets the single-line Input widget specifically (hooks its cursor/
    # value events), and ChatInput is a multiline TextArea for the Ctrl+J-newline support
    # required elsewhere - the two widgets' APIs differ enough that retrofitting it wasn't
    # worth the risk. This hand-rolled version only ever needs to match a short, static list
    # (terminal_ui.SLASH_COMMANDS) against the first line's leading "/word", which is a much
    # narrower job.
    # -------------------------------------------------------------------

    def _autocomplete_widget(self):
        try:
            return self.app.query_one("#autocomplete", Static)
        except Exception:
            return None

    def _update_autocomplete(self) -> None:
        from . import terminal_ui

        cursor_row, cursor_col = self.cursor_location
        if cursor_row != 0:
            self._hide_autocomplete()
            return
        first_line = self.document[0] if self.document.line_count else ""
        prefix = first_line[:cursor_col]
        if not prefix.startswith("/") or " " in prefix:
            self._hide_autocomplete()
            return
        matches = [cmd for cmd, _desc in terminal_ui.SLASH_COMMANDS if cmd.startswith(prefix)]
        if not matches or matches == [prefix]:
            self._hide_autocomplete()
            return
        self._autocomplete_matches = matches
        self._autocomplete_index = 0
        self._show_autocomplete()

    def _show_autocomplete(self) -> None:
        from . import terminal_ui

        panel = self._autocomplete_widget()
        if panel is None:
            return
        descriptions = dict(terminal_ui.SLASH_COMMANDS)
        lines = []
        for i, cmd in enumerate(self._autocomplete_matches[:8]):
            marker = "▶ " if i == self._autocomplete_index else "  "
            desc = descriptions.get(cmd, "")
            line = f"{marker}{cmd}  {desc}"
            lines.append(f"[reverse]{line}[/]" if i == self._autocomplete_index else line)
        panel.update("\n".join(lines))
        panel.display = True

    def _hide_autocomplete(self) -> None:
        self._autocomplete_matches = []
        panel = self._autocomplete_widget()
        if panel is not None:
            panel.display = False

    def _accept_autocomplete(self) -> None:
        if not self._autocomplete_matches:
            return
        cmd = self._autocomplete_matches[self._autocomplete_index]
        cursor_row, cursor_col = self.cursor_location
        first_line = self.document[0] if self.document.line_count else ""
        rest_of_first_line = first_line[cursor_col:].lstrip()
        remaining_lines = [self.document[i] for i in range(1, self.document.line_count)]
        # Keep the trailing space when there's nothing after the cursor yet (so typing can
        # continue straight into the command's argument) - .rstrip() on the combined string
        # would strip that deliberate space back off in exactly that case.
        new_first_line = f"{cmd} {rest_of_first_line}" if rest_of_first_line else f"{cmd} "
        self.text = "\n".join([new_first_line, *remaining_lines])
        self.move_cursor((0, len(new_first_line)))
        self._hide_autocomplete()

    async def _on_key(self, event: events.Key) -> None:
        if self._autocomplete_matches and event.key == "tab":
            event.prevent_default()
            event.stop()
            self._accept_autocomplete()
            return
        if self._autocomplete_matches and event.key == "escape":
            event.prevent_default()
            event.stop()
            self._hide_autocomplete()
            return
        if self._autocomplete_matches and event.key in ("up", "down"):
            event.prevent_default()
            event.stop()
            direction = -1 if event.key == "up" else 1
            self._autocomplete_index = (self._autocomplete_index + direction) % len(self._autocomplete_matches)
            self._show_autocomplete()
            return
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            text = self.text
            self.clear()
            self._hide_autocomplete()
            self.post_message(self.Submitted(self, text))
            return
        if event.key in ("ctrl+j", "shift+enter"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            self._update_autocomplete()
            return
        if event.key == "ctrl+up" or (event.key == "up" and self.cursor_at_start_of_text):
            event.prevent_default()
            event.stop()
            self._recall(-1)
            self._update_autocomplete()
            return
        if event.key == "ctrl+down" or (event.key == "down" and self.cursor_at_end_of_text):
            event.prevent_default()
            event.stop()
            self._recall(1)
            self._update_autocomplete()
            return
        await super()._on_key(event)
        self._update_autocomplete()


class ChatApp(App):
    """One TabbedContent, one TabPane per currently-open session (each with its own
    scrollback history container + ChatInput), plus the shared status/streaming/autocomplete/
    hint strip below it."""

    TITLE = "harness-agent"

    CSS = """
    Screen {
        layout: vertical;
    }

    #session-tabs {
        height: 1fr;
    }

    /* TabbedContent's own internal ContentSwitcher (the container that actually holds
       each TabPane) defaults to height: auto in Textual's built-in CSS - without this
       override, the height chain collapses (auto -> 100% of auto -> auto) and every
       history container ends up with exactly 1 row of real screen space no matter how
       much content it holds, regardless of the height rules below. */
    #session-tabs > ContentSwitcher {
        height: 1fr;
    }

    TabPane {
        height: 100%;
    }

    TabPane VerticalScroll {
        height: 1fr;
        padding: 0 1;
    }

    .startup-banner {
        border: round cyan;
        padding: 1 2;
        height: auto;
        margin: 0 0 1 0;
    }

    Collapsible {
        margin: 0 0 1 0;
    }

    #permission-overlay {
        height: auto;
        display: none;
        background: darkred;
        color: white;
        padding: 1;
        border: solid white;
    }

    #todos-panel {
        height: auto;
        max-height: 30%;
        display: none;
        color: magenta;
        padding: 0 1;
    }

    #streaming {
        height: auto;
        display: none;
        padding: 0 1;
    }

    #status {
        height: auto;
        display: none;
        color: cyan;
        text-style: bold;
        padding: 0 1;
    }

    #autocomplete {
        height: auto;
        max-height: 10;
        display: none;
        background: $panel;
        color: $text;
        padding: 0 1;
    }

    ChatInput {
        border-top: solid cyan;
        border-bottom: solid cyan;
        padding: 0 1;
    }

    #hint {
        height: 1;
        color: $text-muted;
        padding: 0 1;
    }
    """

    def __init__(self, session_manager: "SessionManager", initial_session: "Session", shells: dict[str, str | None]) -> None:
        super().__init__()
        self.session_manager = session_manager
        self._initial_session = initial_session
        self._shells = shells
        self.turn_running = False

    # -------------------------------------------------------------------
    # Tab/pane construction
    # -------------------------------------------------------------------

    def _build_pane(self, session: "Session") -> TabPane:
        from . import terminal_ui

        history = VerticalScroll(
            terminal_ui.build_startup_widget(session.model_name, session.name, self._shells),
            id=terminal_ui.log_widget_id(session.id),
        )
        return TabPane(
            session.name,
            history,
            ChatInput(session.id, id=terminal_ui.input_widget_id(session.id)),
            id=terminal_ui.pane_widget_id(session.id),
        )

    def compose(self) -> ComposeResult:
        from . import terminal_ui

        open_sessions = self.session_manager.open_sessions()
        initial_pane_id = terminal_ui.pane_widget_id(self._initial_session.id)
        with TabbedContent(id="session-tabs", initial=initial_pane_id):
            for session in open_sessions:
                yield self._build_pane(session)
        yield Static("", id=terminal_ui.PERMISSION_OVERLAY_ID)
        yield Static("", id=terminal_ui.TODOS_PANEL_ID)
        yield Static("", id="streaming")
        yield Static("", id="status")
        yield Static("", id="autocomplete")
        yield Static(_HINT_TEXT, id="hint")

    def on_mount(self) -> None:
        from . import terminal_ui
        from . import permissions

        terminal_ui._APP = self
        # No terminal_ui.print_startup(...) call here - each pane already mounted its own
        # compact welcome widget as part of _build_pane, so there's nothing left to render
        # into "whichever tab happens to be active" at app-mount time.
        # Wire the permissions confirm-callback right alongside the terminal_ui app-reference
        # wiring above - request_permission_confirmation_tui blocks whatever background
        # thread calls it until one of the overlay's four key presses is handled (see
        # ChatApp.on_key / terminal_ui.handle_permission_key).
        permissions.set_confirm_callback(terminal_ui.request_permission_confirmation_tui)
        self._focus_active_input()
        terminal_ui.refresh_todos_panel()

    def on_unmount(self) -> None:
        from . import terminal_ui
        from . import permissions

        if terminal_ui._APP is self:
            terminal_ui._APP = None
        permissions.set_confirm_callback(None)

    # -------------------------------------------------------------------
    # Permission confirmation overlay key capture
    # -------------------------------------------------------------------

    async def on_key(self, event: events.Key) -> None:
        """Global key handler (bubbles up here last, after the focused widget - see
        ChatInput._on_key - since App sits at the root of the DOM). Only ever acts while
        terminal_ui's permission overlay is visible: terminal_ui.handle_permission_key
        watches for exactly the four keys the overlay describes (y/a/n/d), writes the
        decision into request_permission_confirmation_tui's waiting holder, and unblocks
        the worker thread it's running on - see terminal_ui.py for the full picture.

        The active tab's ChatInput is disabled for the overlay's whole lifetime (see
        terminal_ui._tui_show_permission_overlay/_tui_hide_permission_overlay), which is
        what keeps ordinary typing from reaching it (and thus from stopping/consuming these
        key events) while a confirmation is pending.
        """
        from . import terminal_ui

        if terminal_ui.handle_permission_key(event.key):
            event.stop()
            event.prevent_default()

    def on_text_selected(self, _event: events.TextSelected) -> None:
        """Mirror terminal copy-on-select for Textual-managed history selections."""
        selected_text = self.screen.get_selected_text()
        if selected_text:
            self.copy_to_clipboard(selected_text)

    # -------------------------------------------------------------------
    # Active-tab resolution (used by terminal_ui._history_container's active-tab fallback)
    # -------------------------------------------------------------------

    def active_session_id(self) -> str | None:
        """The session id backing whichever tab is currently visible, or None if the tab
        strip isn't mounted yet or no tab is active."""
        from . import terminal_ui

        try:
            tabs = self.query_one("#session-tabs", TabbedContent)
        except Exception as exc:
            terminal_ui.log_tui_error("active_session_id: query_one('#session-tabs') failed", exc)
            return None
        active_pane_id = tabs.active
        if not active_pane_id:
            terminal_ui.log_tui_error(
                "active_session_id: TabbedContent.active is empty",
                RuntimeError(f"tabs.active={active_pane_id!r}, tab ids={[p.id for p in tabs.query(TabPane)]!r}"),
            )
            return None
        return terminal_ui.session_id_from_pane_id(active_pane_id)

    def _focus_active_input(self) -> None:
        from . import terminal_ui

        session_id = self.active_session_id()
        if session_id is None:
            return
        try:
            self.query_one(f"#{terminal_ui.input_widget_id(session_id)}", ChatInput).focus()
        except Exception:
            pass

    def on_tabbed_content_tab_activated(self, message: TabbedContent.TabActivated) -> None:
        """Keep SessionManager's own notion of "active session" in sync with whichever tab
        the user actually switches to (click or keyboard) - several slash commands
        (/reset, /model, /session rename, ...) and cli.handle_turn's chat-turn path key off
        SESSION_MANAGER.active_session() rather than a session id passed explicitly."""
        pane = message.pane
        if pane is None or pane.id is None:
            return
        from . import tasks
        from . import terminal_ui

        session_id = terminal_ui.session_id_from_pane_id(pane.id)
        self.session_manager.set_active(session_id)
        # Keep the tasks module's active scope (see tasks.set_active_scope) in sync with the
        # tab the user just switched to as well, not just SESSION_MANAGER's own pointer -
        # otherwise the todo panel refresh below could render the previously-active
        # session's todos for a moment (or until the next slash command) rather than this
        # tab's own list.
        tasks.set_active_scope(session_id)
        self._focus_active_input()
        terminal_ui.refresh_todos_panel()

    # -------------------------------------------------------------------
    # Turn submission/execution
    # -------------------------------------------------------------------

    def on_chat_input_submitted(self, message: ChatInput.Submitted) -> None:
        from . import terminal_ui

        text = terminal_ui.normalize_prompt_input(message.text)
        stripped = text.strip()
        if not stripped:
            return

        terminal_ui.append_prompt_history(text)
        message.chat_input.remember(text)

        if self.turn_running:
            self.bell()
            return

        session_id = message.chat_input.session_id
        self.turn_running = True
        self._set_status_visible(True)
        self.run_worker(functools.partial(self._run_turn, session_id, text), thread=True, exclusive=False)

    def _run_turn(self, session_id: str, text: str) -> None:
        # Runs on a background thread (Textual's thread worker) - AgentRuntime.invoke()/
        # invoke_streaming() are both plain blocking calls (no asyncio anywhere in
        # agent_runtime.py), so this can't be a coroutine without blocking the event loop.
        # handle_turn's own ui_print/render_* calls reach the UI via terminal_ui's
        # _dispatch_to_ui bridge, which is safe to call from here.
        from . import cli

        # Make this the session manager's active session BEFORE calling into cli.handle_turn:
        # handle_turn's chat-turn path (and several of its slash-command handlers) resolve the
        # runtime to use via SESSION_MANAGER.active_session() rather than the id passed here -
        # see this module's docstring for why only one turn is ever allowed in flight at a
        # time app-wide, which is what keeps this from racing another tab's submission.
        self.session_manager.set_active(session_id)
        found = self.session_manager.get_session(session_id)
        if found is None:
            self.call_from_thread(self._turn_finished)
            return
        _session, runtime = found

        try:
            should_exit = cli.handle_turn(runtime, text)
        finally:
            self.call_from_thread(self._turn_finished)
        if should_exit:
            self.call_from_thread(self.exit)

    def _turn_finished(self) -> None:
        from . import terminal_ui

        self.turn_running = False
        self._set_status_visible(False)
        # Re-render the todo panel right after every turn (chat turn or slash command -
        # _set_status_visible above already covers both, see its call site) so it reflects
        # whatever the just-finished turn's todo_write/todo_read calls left behind.
        terminal_ui.refresh_todos_panel()
        # _sync_tabs is a coroutine (see its docstring for why: adding/removing a pane
        # returns an awaitable that must actually be awaited before the pane is mounted/
        # unmounted, otherwise e.g. activating a just-added tab races its own mount) - run it
        # as a worker rather than blocking _turn_finished (itself already running via
        # call_from_thread) on it.
        self.run_worker(self._sync_tabs(), exclusive=False)

    def _set_status_visible(self, visible: bool) -> None:
        status = self.query_one("#status", Static)
        if visible:
            status.update("agent running...")
        status.display = visible

    # -------------------------------------------------------------------
    # Tab lifecycle wiring for /session new|list|switch|resume|rename|close
    # -------------------------------------------------------------------

    async def _sync_tabs(self) -> None:
        """Reconcile the tab strip against SessionManager's current open-session set.

        Called after every turn (including slash-command-only turns), since that's the one
        place every /session mutation funnels through (handle_session_cli, called from
        cli.handle_turn) - rather than teaching each slash command handler about tabs
        directly, this just diffs "what's open now" against "what tabs exist now" and adds,
        removes, renames, or activates panes to match. Add/remove are awaited (both return an
        awaitable that must complete before the pane is actually mounted/unmounted -
        activating a pane before its add_pane() finishes mounting it is a no-op).
        """
        from . import terminal_ui

        try:
            tabs = self.query_one("#session-tabs", TabbedContent)
        except Exception:
            return

        open_sessions = {s.id: s for s in self.session_manager.open_sessions()}
        existing_pane_ids = {pane.id for pane in tabs.query(TabPane) if pane.id}
        existing_session_ids = {terminal_ui.session_id_from_pane_id(pid) for pid in existing_pane_ids}

        # Drop tabs for sessions that were closed (e.g. /session close).
        for pane_id in existing_pane_ids:
            session_id = terminal_ui.session_id_from_pane_id(pane_id)
            if session_id not in open_sessions:
                await tabs.remove_pane(pane_id)

        # Add tabs for newly-opened sessions (e.g. /session new, or /session switch|resume
        # onto a session that wasn't already a tab).
        for session_id, session in open_sessions.items():
            if session_id not in existing_session_ids:
                await tabs.add_pane(self._build_pane(session))

        # Keep tab labels in sync with /session rename.
        for session_id, session in open_sessions.items():
            pane_id = terminal_ui.pane_widget_id(session_id)
            try:
                tabs.get_tab(pane_id).label = session.name
            except Exception:
                pass

        # Reflect SessionManager's active session as the visually-active tab (e.g.
        # /session switch|resume, or the fallback resume /session close performs).
        active = self.session_manager.active_session()
        if active is not None:
            active_session, _runtime = active
            pane_id = terminal_ui.pane_widget_id(active_session.id)
            if tabs.active != pane_id:
                try:
                    tabs.active = pane_id
                except Exception:
                    pass
                else:
                    self._focus_active_input()
