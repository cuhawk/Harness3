"""Todo/plan tracking for the agent, using the Claude Code TodoWrite shape.

The whole list is replaced on every write rather than incrementally
create/update per item. That wholesale-replace shape is deliberate: it avoids
id-desync bugs where the model's idea of an item's status drifts from the
store because of a missed incremental update - the model just re-sends its
full current view of the plan and that becomes the new truth.

This module must NOT import sessions.py: a different task is building that
module in parallel and it may not exist on disk yet. Instead it follows the
exact dependency-injection pattern workflows.py already uses for its runtime
provider - a module-level :func:`set_active_scope` that whoever owns sessions
later calls (e.g. ``tasks.set_active_scope(session.id)``) with zero import
coupling.

Every path derived from the active workspace is read off ``core`` at call
time (``core.CACHE_DIR``) rather than imported by name, so that
``core.set_workspace()`` is honored - see core.py's module docstring for why.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import core

VALID_STATUSES = ("pending", "in_progress", "completed")

# Leading list decoration to strip when parsing a raw pasted list into task
# lines: "1.", "1)", "- ", "* ", "• ", "[ ] ", "[x] ", "> ". Matched at line
# start only, so a hyphen/number mid-sentence is left untouched.
_LIST_MARKER_RE = re.compile(
    r"^\s*(?:[-*•·‣▪]|\d+[.)]|\(?\d+\)|[a-zA-Z][.)]|\[[ xX]?\])\s+"
)

# Scope id used to compute the per-scope todos file path, mutated only via
# set_active_scope() - see module docstring for why this exists instead of
# importing sessions.py directly.
_ACTIVE_SCOPE = "default"


def set_active_scope(scope_id: str) -> None:
    global _ACTIVE_SCOPE
    _ACTIVE_SCOPE = core.safe_name(str(scope_id or "default"))


def _todos_path() -> Path:
    return core.CACHE_DIR / f"todos_{_ACTIVE_SCOPE}.json"


def _load_todos() -> list[dict[str, Any]]:
    path = _todos_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception:
        pass
    return []


def _save_todos(todos: list[dict[str, Any]]) -> None:
    path = _todos_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_todos(todos: Any) -> str | None:
    """Return an error string if todos is not a valid TodoWrite-shaped list, else None."""
    if not isinstance(todos, list):
        return "todos must be a JSON list of objects"
    for i, item in enumerate(todos):
        if not isinstance(item, dict):
            return f"item {i} is not an object"
        if not isinstance(item.get("id"), str) or not item.get("id", "").strip():
            return f"item {i} is missing a non-empty string 'id'"
        if not isinstance(item.get("content"), str) or not item.get("content", "").strip():
            return f"item {i} is missing a non-empty string 'content'"
        if item.get("status") not in VALID_STATUSES:
            return f"item {i} has invalid status {item.get('status')!r}; must be one of {VALID_STATUSES}"
    return None


def current_todos() -> list[dict[str, Any]]:
    """Plain (non-tool) accessor so a UI panel can poll the current todo list
    without going through LLM tool-calling machinery."""
    return _load_todos()


@tool
def todo_write(todos_json: str) -> str:
    """Replace the entire todo/plan list for the current session with the given list.

    todos_json must be a JSON list of objects, each with 'id' (string), 'content'
    (string), and 'status' (one of 'pending', 'in_progress', 'completed'). The whole
    list is overwritten wholesale on every call - always send the full current plan,
    not just the items that changed."""
    try:
        todos = core.parse_json_arg(todos_json, [])
        error = _validate_todos(todos)
        if error:
            return core.to_json({"ok": False, "error": error})
        _save_todos(todos)
        return core.to_json({"ok": True, "count": len(todos), "todos": todos})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def todo_read() -> str:
    """Read the current todo/plan list for the current session (empty list if none yet)."""
    try:
        todos = _load_todos()
        return core.to_json({"ok": True, "count": len(todos), "todos": todos})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


# =============================================================================
# todo_build - turn a raw pasted list of tasks into a structured todo list
# =============================================================================

def _split_task_lines(text: str) -> list[str]:
    """Deterministically parse a raw blob of pasted tasks into clean task strings.

    Handles the shapes a human actually pastes: newline-separated lines, numbered
    or bulleted lists (markers stripped), and - only when the whole blob is a
    single line - semicolon-separated items. Never calls out to a model, so it is
    instant and works with the router down; it is also the fallback when the
    optional LLM decompose pass fails.
    """
    raw_lines = [ln.strip() for ln in (text or "").replace("\r\n", "\n").split("\n")]
    lines = [_LIST_MARKER_RE.sub("", ln).strip() for ln in raw_lines if ln.strip()]
    # A single unbroken line is often "a; b; c" - split it so we don't make one
    # giant todo out of several tasks. Multi-line input is trusted as-is (a stray
    # semicolon inside one real task must not shatter it).
    if len(lines) == 1 and ";" in lines[0]:
        lines = [part.strip() for part in lines[0].split(";") if part.strip()]
    return lines


def _llm_decompose(text: str, model_name: str) -> list[str]:
    """Ask a cheap router model to clean the raw list and atomize compound items.

    Returns a list of task strings, or raises on any failure (no JSON, empty,
    router down) so the caller falls back to the deterministic split. Runs as a
    bare tool-less LLM call - it only needs to reshape text, not act.
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model_name or "bulk-worker",
        base_url=core.LITELLM_BASE_URL,
        api_key=core.LITELLM_API_KEY,
        temperature=0,
        timeout=90,
        max_retries=2,
    )
    prompt = (
        "Turn the raw task list below into a clean, ordered list of atomic, "
        "actionable todo items. Rules:\n"
        "- Split a compound item into separate items ONLY when it names genuinely "
        "separate deliverables; keep a single coherent task as one item.\n"
        "- Do NOT invent, pad, or editorialize; preserve the user's meaning and "
        "concrete details (names, paths, numbers).\n"
        "- Put items in a sensible execution order.\n"
        "- Return ONLY a JSON array of strings, nothing else.\n\n"
        f"RAW TASKS:\n{text}"
    )
    raw = str(getattr(llm.invoke(prompt), "content", "") or "")
    # Reuse the subagent fence-stripper so a ```json ... ``` wrap doesn't break parsing.
    from .subagent import _strip_code_fence

    parsed = json.loads(_strip_code_fence(raw))
    items = [str(x).strip() for x in parsed if str(x).strip()] if isinstance(parsed, list) else []
    if not items:
        raise ValueError("decompose returned no usable items")
    return items


def _next_id_maker(existing: list[dict[str, Any]]):
    """Return a callable that hands out 't<N>' ids not colliding with existing ones."""
    used = {str(item.get("id")) for item in existing}
    counter = {"n": 0}

    def make() -> str:
        while True:
            counter["n"] += 1
            candidate = f"t{counter['n']}"
            if candidate not in used:
                used.add(candidate)
                return candidate

    return make


@tool
def todo_build(
    tasks_text: str,
    decompose: bool = False,
    append: bool = False,
    model_name: str = "",
) -> str:
    """Build a structured todo list from a raw pasted list of tasks and save it for this session.

    Give it whatever the user handed over - a numbered list, bullets, checkboxes,
    or newline/semicolon-separated lines - and it strips the list decoration,
    assigns each item a stable id and status='pending', and stores it so todo_read
    (and any UI panel) picks it up immediately. This is the front door for "here's
    my list of tasks, track them"; todo_write remains the low-level whole-list
    replace for status updates.

    Args:
      tasks_text: the raw task list / blob to parse.
      decompose: if true, first send the blob to a cheap router model to atomize
        compound items and order them (offloads the work off the main model). Falls
        back to the deterministic line split automatically if the model is
        unavailable or returns nothing usable - the response's 'source' field says
        which path was taken.
      append: if true, add the parsed items after the existing todos (ids continue
        without collision) instead of replacing the list.
      model_name: router model alias for decompose (default 'bulk-worker').

    Returns the saved list plus counts, and (when decompose was requested) which
    parsing path actually produced the items."""
    try:
        source = "split"
        items: list[str] = []
        decompose_error = ""
        if decompose:
            try:
                items = _llm_decompose(tasks_text, model_name)
                source = "llm"
            except Exception as exc:
                decompose_error = repr(exc)
                items = []
        if not items:
            items = _split_task_lines(tasks_text)
        if not items:
            return core.to_json({"ok": False, "error": "no tasks found in tasks_text"})

        existing = _load_todos() if append else []
        make_id = _next_id_maker(existing)
        new_items = [{"id": make_id(), "content": content, "status": "pending"} for content in items]
        todos = existing + new_items

        error = _validate_todos(todos)
        if error:
            return core.to_json({"ok": False, "error": error})
        _save_todos(todos)

        result: dict[str, Any] = {
            "ok": True,
            "source": source,
            "added": len(new_items),
            "count": len(todos),
            "todos": todos,
        }
        if decompose and decompose_error:
            result["decompose_fallback"] = decompose_error
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
