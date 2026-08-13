"""Conversation session persistence plus an in-memory multi-session manager.

A "session" is a named, persisted conversation (model + history) that can be listed,
resumed, renamed, or closed independently of any single process run - the same
save-the-whole-file-atomically pattern used elsewhere for JSON registries (see
workflows.py's workflow_load_all/workflow_save_all), but keyed by session id instead of
name, and split into one file per session so listing/renaming one session never touches
another's data.

No @tool-decorated functions live here on purpose: session switching is a human/CLI-driven
operation (slash commands wired up in a later integration pass), never something the agent
itself should be able to invoke mid-conversation.

This module must NOT import agent_runtime at module top level: agent_runtime depends
(transitively, via tools_registry) on many leaf modules, and importing it back here would
risk an import cycle. SessionManager.create_runtime imports it lazily inside the method
body instead - the same trick subagent.py already uses for exactly the same reason.

Workspace-derived paths are read off core.CACHE_DIR at the point of use (never bound to a
module-level constant) because core.set_workspace() rebinds core.CACHE_DIR at runtime - see
core.py's module docstring for why a frozen import would go stale.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from . import core

# Primitive JSON scalar types kept from additional_kwargs during serialization - that dict
# can also hold provider-specific non-primitive objects (nested structures, custom types)
# that are not reliably round-trippable, but the reasoning text this is meant to preserve
# (see agent_runtime._extract_reasoning's "reasoning_content"/"reasoning" keys) is always a
# primitive.
_PRIMITIVE_TYPES = (str, int, float, bool, type(None))


def _json_safe(value: Any) -> Any:
    """Return value if JSON-serializable, otherwise a string-coerced equivalent."""
    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except Exception:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))


# =============================================================================
# History (de)serialization
# =============================================================================

def serialize_history(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append({"type": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            out.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            primitive_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, _PRIMITIVE_TYPES)}
            out.append({
                "type": "ai",
                "content": msg.content,
                "tool_calls": msg.tool_calls,
                "additional_kwargs": primitive_kwargs,
            })
        elif isinstance(msg, ToolMessage):
            entry: dict[str, Any] = {"type": "tool", "content": msg.content}
            name = getattr(msg, "name", None)
            if name is not None:
                entry["name"] = name
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id is not None:
                entry["tool_call_id"] = tool_call_id
            artifact = getattr(msg, "artifact", None)
            if artifact is not None:
                entry["artifact"] = _json_safe(artifact)
            out.append(entry)
        else:
            out.append({"type": "unknown", "content": str(getattr(msg, "content", msg))})
    return out


def deserialize_history(data: list[dict[str, Any]]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        msg_type = entry.get("type")
        content = entry.get("content", "")
        if msg_type == "system":
            messages.append(SystemMessage(content=content))
        elif msg_type == "human":
            messages.append(HumanMessage(content=content))
        elif msg_type == "ai":
            messages.append(AIMessage(
                content=content,
                tool_calls=entry.get("tool_calls") or [],
                additional_kwargs=entry.get("additional_kwargs") or {},
            ))
        elif msg_type == "tool":
            tool_kwargs: dict[str, Any] = {"content": content, "tool_call_id": entry.get("tool_call_id", "")}
            if "name" in entry:
                tool_kwargs["name"] = entry["name"]
            if "artifact" in entry:
                tool_kwargs["artifact"] = entry["artifact"]
            messages.append(ToolMessage(**tool_kwargs))
        else:
            continue  # unrecognized type: skip silently rather than raise
    return messages


# =============================================================================
# Session metadata + on-disk store
# =============================================================================

@dataclass
class Session:
    id: str
    name: str
    model_name: str
    workspace: str
    created_at: str
    updated_at: str

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Session":
        return Session(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            model_name=str(data.get("model_name", "")),
            workspace=str(data.get("workspace", "")),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "model_name": self.model_name,
            "workspace": self.workspace,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class SessionStore:
    """Reads/writes one JSON file per session under core.CACHE_DIR / "sessions".

    Every method recomputes that directory instead of caching it, since core.CACHE_DIR
    changes whenever core.set_workspace() runs - see this module's docstring.
    """

    def _sessions_dir(self) -> Path:
        d = core.CACHE_DIR / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path_for_id(self, session_id: str) -> Path:
        return self._sessions_dir() / f"{core.safe_name(session_id)}.json"

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.tmp")
        tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(str(tmp_path), str(path))

    def _write_full(self, session: Session, raw_history: list[dict[str, Any]]) -> None:
        self._atomic_write(self._path_for_id(session.id), {"session": session.to_dict(), "history": raw_history})

    def create(self, name: str, model_name: str) -> Session:
        session_id = uuid.uuid4().hex[:16]
        name = str(name or "").strip() or f"session-{session_id[:8]}"
        now = core.now_str()
        session = Session(
            id=session_id,
            name=name,
            model_name=model_name,
            workspace=str(core.WORKSPACE),
            created_at=now,
            updated_at=now,
        )
        self._write_full(session, [])
        return session

    def list(self) -> list[Session]:
        sessions: list[Session] = []
        for path in self._sessions_dir().glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                sessions.append(Session.from_dict(data["session"]))
            except Exception:
                continue
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def load(self, session_id: str) -> tuple[Session, list[dict[str, Any]]]:
        path = self._path_for_id(session_id)
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        raw_history = data.get("history") or []
        if not isinstance(raw_history, list):
            raw_history = []
        return Session.from_dict(data["session"]), raw_history

    def save(self, session_id: str, session: Session, history: list[BaseMessage]) -> Session:
        session.id = session_id
        session.updated_at = core.now_str()
        self._write_full(session, serialize_history(history))
        return session

    def rename(self, session_id: str, new_name: str) -> Session:
        session, raw_history = self.load(session_id)
        session.name = str(new_name or "").strip() or session.name
        session.updated_at = core.now_str()
        self._write_full(session, raw_history)
        return session

    def delete(self, session_id: str) -> bool:
        path = self._path_for_id(session_id)
        if path.exists():
            path.unlink()
            return True
        return False


# =============================================================================
# In-memory multi-session manager
# =============================================================================

class SessionManager:
    """Owns the live (in-process) AgentRuntime for every currently-open session.

    Runtimes are kept in an untyped dict on purpose (documented as "values are
    AgentRuntime instances") so this module never has to import agent_runtime except
    lazily inside create_runtime - see the module docstring.
    """

    def __init__(self) -> None:
        self.store = SessionStore()
        self._runtimes: dict[str, Any] = {}
        self._sessions: dict[str, Session] = {}
        self._active_id: Optional[str] = None

    def create_runtime(self, session: Session, restore_history: Optional[list[dict[str, Any]]] = None) -> Any:
        from .agent_runtime import AgentRuntime  # lazy import: avoids a module import cycle, see docstring

        runtime = AgentRuntime(session.model_name, session_id=session.id)
        if restore_history:
            runtime.history = deserialize_history(restore_history)
        self._runtimes[session.id] = runtime
        self._sessions[session.id] = session
        return runtime

    def active_session(self) -> Optional[tuple[Session, Any]]:
        if self._active_id is None:
            return None
        runtime = self._runtimes.get(self._active_id)
        session = self._sessions.get(self._active_id)
        if runtime is None or session is None:
            return None
        return session, runtime

    def set_active(self, session_id: str) -> bool:
        if session_id in self._runtimes:
            self._active_id = session_id
            return True
        return False

    def get_session(self, session_id: str) -> Optional[tuple[Session, Any]]:
        """Look up a specific open session/runtime pair by id, without touching which
        session is active - unlike active_session(), which always reads self._active_id.
        Used by the Textual UI (see tui_app.ChatApp._run_turn) to dispatch a submitted turn
        to the exact session tab it came from."""
        runtime = self._runtimes.get(session_id)
        session = self._sessions.get(session_id)
        if runtime is None or session is None:
            return None
        return session, runtime

    def open_sessions(self) -> list[Session]:
        """Sessions that currently have a live in-memory runtime - i.e. sessions with an open
        tab - as opposed to list_sessions(), which lists every persisted session on disk
        whether or not it's currently open."""
        return list(self._sessions.values())

    def list_sessions(self) -> list[Session]:
        return self.store.list()

    def new_session(self, name: str, model_name: str) -> tuple[Session, Any]:
        session = self.store.create(name, model_name)
        runtime = self.create_runtime(session)
        self._active_id = session.id
        return session, runtime

    def resume_session(self, session_id: str) -> tuple[Session, Any]:
        if session_id in self._runtimes:
            runtime = self._runtimes[session_id]
            session = self._sessions[session_id]
        else:
            session, raw_history = self.store.load(session_id)
            runtime = self.create_runtime(session, restore_history=raw_history)
        self._active_id = session_id
        return session, runtime

    def rename_active(self, new_name: str) -> Optional[Session]:
        """Rename the active session (in the persisted store and in the in-memory Session
        object active_session() hands out) and return it, or None if nothing is active."""
        active = self.active_session()
        if active is None:
            return None
        session, _runtime = active
        renamed = self.store.rename(session.id, new_name)
        session.name = renamed.name
        session.updated_at = renamed.updated_at
        return session

    def persist_active(self) -> None:
        active = self.active_session()
        if active is None:
            return
        session, runtime = active
        self.store.save(session.id, session, runtime.history)

    def close_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        runtime = self._runtimes.get(session_id)
        if session is not None:
            history = runtime.history if runtime is not None else []
            self.store.save(session_id, session, history)
        self._runtimes.pop(session_id, None)
        self._sessions.pop(session_id, None)
        if self._active_id == session_id:
            self._active_id = None


# =============================================================================
# Shared post-turn persistence
# =============================================================================

def invoke_and_persist(
    store: SessionStore,
    session: Session,
    runtime: Any,
    user_text: str,
    streaming_on_token: Optional[Callable[[str], None]] = None,
) -> dict[str, Any]:
    """Run one turn and persist it - the single call site cli.py/tui_app.py should use so
    after-every-turn persistence isn't duplicated at each call site. Duck-typed on runtime
    (only .invoke/.invoke_streaming/.history are used) so this has zero import coupling to
    agent_runtime.py.
    """
    if streaming_on_token is not None:
        result = runtime.invoke_streaming(user_text, streaming_on_token)
    else:
        result = runtime.invoke(user_text)
    store.save(session.id, session, runtime.history)
    return result
