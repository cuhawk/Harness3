"""Three-tier memory: session / procedural / rag.

Completes the split described in tools_registry.py's and cli.py's "KNOWN DEVIATION"
docstrings - this module did not exist yet when those were written, so
``handle_memory_cli``'s ``from . import memory`` raised ModuleNotFoundError at call
time. Implements exactly the API cli.py already calls: memory_list, memory_search,
memory_search_debug_fixes, memory_get, memory_save, memory_delete,
memory_promote_from_session - plus memory_save_debug_fix, referenced by
core.py's system prompt ("Use memory_save_debug_fix for fixes specifically").

Three tiers (see core.py's build_system_prompt for the authoritative description):
- session (default): short-lived working notes scoped to the current task/run.
- procedural: durable, reusable "how to do X" knowledge that should survive long-term
  across sessions and workspaces.
- rag: bulky reference material; saving with kind="rag" also indexes the content into
  rag_search (collection="memory") via rag.rag_store_document, so it is retrievable by
  semantic/lexical search, not just by memory_id.

Workspace-derived paths are read through ``core.MEMORY_DB_PATH`` at call time (never
``from .core import MEMORY_DB_PATH``) so a runtime workspace switch via
``core.set_workspace()`` is honored - see core.py's module docstring for why.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any

from langchain_core.tools import tool

from . import core
from . import rag

ALLOWED_KINDS = {"session", "procedural", "rag"}
RAG_MEMORY_COLLECTION = "memory"


def memory_connect() -> sqlite3.Connection:
    core.ensure_dirs()
    conn = sqlite3.connect(str(core.MEMORY_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '',
            rag_source_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
        """
    )
    return conn


def memory_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "memory_id": row["memory_id"],
        "kind": row["kind"],
        "content": row["content"],
        "tags": row["tags"],
        "rag_source_id": row["rag_source_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@tool
def memory_save(content: str, kind: str = "session", tags: str = "") -> str:
    """Save a memory. kind must be one of: rag, session, procedural.
    - session (default): short-lived working notes scoped to the current task/run.
    - procedural: durable, reusable "how to do X" knowledge that should survive across
      sessions and workspaces (use memory_save_debug_fix for fixes specifically).
    - rag: large/bulky reference material - also indexed into rag_search
      (collection="memory") for semantic/lexical search, not just recall by memory_id.
    """
    try:
        kind = str(kind or "session").strip().lower()
        if kind not in ALLOWED_KINDS:
            return core.to_json({
                "ok": False,
                "error": f"kind must be one of: {', '.join(sorted(ALLOWED_KINDS))}",
                "kind": kind,
            })
        content = str(content or "")
        if not content.strip():
            return core.to_json({"ok": False, "error": "content is required"})

        memory_id = uuid.uuid4().hex[:32]
        now = core.now_str()

        rag_result: dict[str, Any] | None = None
        rag_source_id: str | None = None
        if kind == "rag":
            rag_result = rag.rag_store_document(
                collection=RAG_MEMORY_COLLECTION,
                source_type="memory",
                path_or_name=f"memory:{memory_id}",
                title=core.trim_text(content, 80),
                text_value=content,
                metadata={"memory_id": memory_id, "tags": tags},
            )
            if isinstance(rag_result, dict):
                rag_source_id = rag_result.get("source_id")

        conn = memory_connect()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO memories(memory_id, kind, content, tags, rag_source_id, created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (memory_id, kind, content, str(tags or ""), rag_source_id, now, now),
                )
        finally:
            conn.close()

        return core.to_json({
            "ok": True,
            "memory_id": memory_id,
            "kind": kind,
            "tags": tags,
            "created_at": now,
            "rag": rag_result,
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_save_debug_fix(content: str, tags: str = "") -> str:
    """Save a durable procedural memory specifically for a debugging fix/gotcha/runbook
    that worked, so it survives across sessions and workspaces. Convenience wrapper
    around memory_save(kind='procedural') that also tags the record 'debug_fix' so
    memory_search_debug_fixes can find it."""
    try:
        merged_tags = ",".join(t for t in [str(tags or "").strip(), "debug_fix"] if t)
        return memory_save.invoke({"content": content, "kind": "procedural", "tags": merged_tags})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_list(kind: str = "", limit: int = 100) -> str:
    """List saved memories, optionally filtered by kind (rag/session/procedural), most recently updated first."""
    try:
        limit = max(1, min(int(limit), 1000))
        conn = memory_connect()
        try:
            if kind.strip():
                k = str(kind).strip().lower()
                if k not in ALLOWED_KINDS:
                    return core.to_json({"ok": False, "error": f"kind must be one of: {', '.join(sorted(ALLOWED_KINDS))}"})
                rows = conn.execute("SELECT * FROM memories WHERE kind = ? ORDER BY updated_at DESC LIMIT ?", (k, limit)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        finally:
            conn.close()
        items = [memory_row_to_dict(r) for r in rows]
        return core.to_json({"ok": True, "count": len(items), "memories": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_get(memory_id: str) -> str:
    """Get a single saved memory record by memory_id."""
    try:
        conn = memory_connect()
        try:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        finally:
            conn.close()
        if not row:
            return core.to_json({"ok": False, "error": "Unknown memory_id", "memory_id": memory_id})
        return core.to_json({"ok": True, "memory": memory_row_to_dict(row)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_search(query: str, kind: str = "", limit: int = 20) -> str:
    """Lexically search saved session/procedural/rag memory records by content. Use
    rag_search (collection='memory') instead for deep semantic search over rag-kind memories."""
    try:
        limit = max(1, min(int(limit), 200))
        conn = memory_connect()
        try:
            if kind.strip():
                rows = conn.execute("SELECT * FROM memories WHERE kind = ?", (str(kind).strip().lower(),)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memories").fetchall()
        finally:
            conn.close()
        scored = []
        for row in rows:
            score = rag.rag_lexical_score(query, row["content"])
            if score > 0:
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [{**memory_row_to_dict(r), "score": round(float(s), 6)} for s, r in scored[:limit]]
        return core.to_json({"ok": True, "query": query, "count": len(results), "memories": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_search_debug_fixes(query: str, limit: int = 20) -> str:
    """Search saved debug-fix procedural memories (saved via memory_save_debug_fix) by content."""
    try:
        limit = max(1, min(int(limit), 200))
        conn = memory_connect()
        try:
            rows = conn.execute("SELECT * FROM memories WHERE kind = 'procedural' AND tags LIKE '%debug_fix%'").fetchall()
        finally:
            conn.close()
        scored = []
        q = str(query or "")
        for row in rows:
            score = rag.rag_lexical_score(q, row["content"]) if q.strip() else 0.0
            if score > 0 or not q.strip():
                scored.append((score, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [{**memory_row_to_dict(r), "score": round(float(s), 6)} for s, r in scored[:limit]]
        return core.to_json({"ok": True, "query": query, "count": len(results), "memories": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_delete(memory_id: str) -> str:
    """Delete a saved memory by memory_id. If it was kind='rag', also removes its indexed RAG chunks."""
    try:
        conn = memory_connect()
        try:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
            if not row:
                return core.to_json({"ok": False, "error": "Unknown memory_id", "memory_id": memory_id})
            with conn:
                conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
        finally:
            conn.close()
        rag_delete_result = None
        if row["rag_source_id"]:
            try:
                rag_delete_result = json.loads(rag.rag_delete_source.invoke({"source_id": row["rag_source_id"], "collection": RAG_MEMORY_COLLECTION}))
            except Exception as exc:
                rag_delete_result = {"ok": False, "error": repr(exc)}
        return core.to_json({"ok": True, "memory_id": memory_id, "rag_delete": rag_delete_result})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_update(memory_id: str, content: str = "", kind: str = "", tags: str = "") -> str:
    """Update a saved memory's content/kind/tags by memory_id. Empty fields are left
    unchanged. If kind is changed to/from 'rag', the RAG index is re-synced (old rag
    chunks removed if leaving 'rag', newly indexed if entering it)."""
    try:
        conn = memory_connect()
        try:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
            if not row:
                return core.to_json({"ok": False, "error": "Unknown memory_id", "memory_id": memory_id})

            new_kind = str(kind).strip().lower() if kind.strip() else row["kind"]
            if new_kind not in ALLOWED_KINDS:
                return core.to_json({"ok": False, "error": f"kind must be one of: {', '.join(sorted(ALLOWED_KINDS))}"})
            new_content = content if content.strip() else row["content"]
            new_tags = tags if tags.strip() else row["tags"]
            now = core.now_str()

            rag_source_id = row["rag_source_id"]
            rag_result = None
            if row["kind"] == "rag" and new_kind != "rag" and rag_source_id:
                rag_result = json.loads(rag.rag_delete_source.invoke({"source_id": rag_source_id, "collection": RAG_MEMORY_COLLECTION}))
                rag_source_id = None
            elif new_kind == "rag" and (row["kind"] != "rag" or content.strip()):
                rag_result = rag.rag_store_document(
                    collection=RAG_MEMORY_COLLECTION,
                    source_type="memory",
                    path_or_name=f"memory:{memory_id}",
                    title=core.trim_text(new_content, 80),
                    text_value=new_content,
                    metadata={"memory_id": memory_id, "tags": new_tags},
                )
                if isinstance(rag_result, dict):
                    rag_source_id = rag_result.get("source_id")

            with conn:
                conn.execute(
                    "UPDATE memories SET content = ?, kind = ?, tags = ?, rag_source_id = ?, updated_at = ? WHERE memory_id = ?",
                    (new_content, new_kind, new_tags, rag_source_id, now, memory_id),
                )
            updated = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        finally:
            conn.close()
        return core.to_json({"ok": True, "memory": memory_row_to_dict(updated), "rag": rag_result})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_clear_session() -> str:
    """Delete all kind='session' memories (short-lived working notes). Called automatically
    on /reset so stale session notes don't leak into a fresh conversation; procedural and
    rag memories are untouched since they are meant to survive across sessions."""
    try:
        conn = memory_connect()
        try:
            with conn:
                cur = conn.execute("DELETE FROM memories WHERE kind = 'session'")
                deleted = cur.rowcount
        finally:
            conn.close()
        return core.to_json({"ok": True, "deleted": deleted})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def memory_promote_from_session(memory_id: str) -> str:
    """Promote a session-kind memory to procedural (durable, cross-session) status by memory_id."""
    try:
        conn = memory_connect()
        try:
            row = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
            if not row:
                return core.to_json({"ok": False, "error": "Unknown memory_id", "memory_id": memory_id})
            if row["kind"] != "session":
                return core.to_json({
                    "ok": False,
                    "error": f"Memory is kind={row['kind']}, not session; only session memories can be promoted",
                    "memory_id": memory_id,
                })
            now = core.now_str()
            with conn:
                conn.execute("UPDATE memories SET kind = 'procedural', updated_at = ? WHERE memory_id = ?", (now, memory_id))
            updated = conn.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,)).fetchone()
        finally:
            conn.close()
        return core.to_json({"ok": True, "memory": memory_row_to_dict(updated)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
