"""Standing per-workspace context for delegated agents.

Every ``spawn_subagent`` call starts a sub-agent with empty history, so the caller ends up
re-pasting the same preamble ("the target is X, scope is Y, the bundles live in Z, report
findings as file:line") into every task description. That preamble is paid for on every
single call, by the expensive model, forever.

A briefing is that preamble stored once per workspace and prepended to the system prompt of
every agent run in it (see core.build_system_prompt). It is workspace-scoped because it is
target-specific: switching workspaces switches briefings, which is what you want when the
briefing says "the target is acme.com".

Deliberately NOT part of the memory subsystem: memory is recall-by-search, whereas this is
unconditional context every agent must see whether or not it thinks to look it up.
"""
from __future__ import annotations

from langchain_core.tools import tool

from . import core

MAX_BRIEFING_CHARS = 4000


@tool
def set_briefing(text: str, append: bool = False) -> str:
    """Set (or append to) the standing briefing for the current workspace: context every
    delegated sub-agent should always have, without you repeating it in each task description.

    Good briefing content: the target and its scope, where the relevant code/artifacts live,
    house rules ("never touch production", "always report findings as file:line"), the output
    format you always want. Keep it short - it is prepended to EVERY sub-agent's system prompt,
    so it is paid for on every delegated run. Pass an empty string to clear it.

    Once set, task descriptions can be just the task. Returns the stored briefing's size."""
    try:
        core.ensure_dirs()
        path = core.CACHE_DIR / core.BRIEFING_FILENAME
        body = (text or "").strip()

        if append and body:
            existing = core.load_briefing()
            body = f"{existing}\n{body}".strip() if existing else body

        if not body:
            if path.exists():
                path.unlink()
            return core.to_json({"ok": True, "cleared": True, "workspace": str(core.WORKSPACE)})

        if len(body) > MAX_BRIEFING_CHARS:
            return core.to_json({
                "ok": False,
                "error": (
                    f"briefing is {len(body)} chars, over the {MAX_BRIEFING_CHARS} limit. It is "
                    "prepended to every sub-agent's prompt - keep it to durable essentials and "
                    "put bulk reference material in RAG instead."
                ),
            })

        path.write_text(body, encoding="utf-8")
        return core.to_json({
            "ok": True,
            "workspace": str(core.WORKSPACE),
            "chars": len(body),
            "path": str(path),
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def get_briefing() -> str:
    """Show the standing briefing currently prepended to every delegated sub-agent's system
    prompt in this workspace (empty if none is set)."""
    try:
        body = core.load_briefing()
        return core.to_json({
            "ok": True,
            "workspace": str(core.WORKSPACE),
            "chars": len(body),
            "briefing": body,
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
