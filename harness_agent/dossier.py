"""Typed, per-target knowledge that sub-agents WRITE BACK to, and later ones start with.

Everything else in this harness flows one way: a sub-agent discovers something, reports it in
its final answer, and dies. The next sub-agent - next hour or next month - rediscovers the same
endpoint on the operator's budget. A briefing (briefing.py) fixes the outbound half: what the
operator tells every agent. This fixes the inbound half: what the agents learn back.

Why not memory.py: memories are free text plus tags, recalled by search. A dossier is TYPED and
keyed by target (endpoints, params, roles, hosts, findings), which is what makes it injectable -
you can put a compact, deduplicated dossier into every sub-agent's system prompt, whereas you
cannot inject an unbounded pile of free-text notes.

The failure mode to design against is a junk drawer: unbounded accretion that costs tokens on
every single call forever. Hence every entry carries confidence and source, injection is capped
hard (INJECT_MAX_*), and low-confidence entries are the first to be dropped from the injected
view - they stay queryable, they just stop being free.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import core

KINDS = ("endpoint", "param", "role", "host", "credential", "finding", "tech", "note")
CONFIDENCES = ("confirmed", "likely", "suspected")

ACTIVE_TARGET_FILENAME = "dossier_active.txt"

# Injection budget. A dossier that grows without bound would silently inflate every delegated
# call, so the injected view is capped and the rest stays query-only.
INJECT_MAX_PER_KIND = 25
INJECT_MAX_CHARS = 3000


def _db_path() -> Path:
    return core.CACHE_DIR / "dossier.sqlite3"


def _connect() -> sqlite3.Connection:
    core.ensure_dirs()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS dossier (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            value TEXT NOT NULL,
            detail TEXT,
            confidence TEXT NOT NULL DEFAULT 'suspected',
            source TEXT,
            hits INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_dossier_uniq ON dossier(target, kind, value);
        CREATE INDEX IF NOT EXISTS idx_dossier_target ON dossier(target, kind);
        """
    )
    return conn


def active_target() -> str:
    try:
        return (core.CACHE_DIR / ACTIVE_TARGET_FILENAME).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError):
        return ""


def render_for_prompt(target: str = "") -> str:
    """Compact text view of a target's dossier for injection into a system prompt.

    Ordered confirmed -> likely -> suspected and capped, so the most load-bearing knowledge
    survives the budget and speculative noise is what gets cut.
    """
    target = (target or active_target()).strip()
    if not target:
        return ""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT kind, value, detail, confidence FROM dossier WHERE target = ?"
            " ORDER BY CASE confidence WHEN 'confirmed' THEN 0 WHEN 'likely' THEN 1 ELSE 2 END,"
            " hits DESC, updated_at DESC",
            (target,),
        ).fetchall()
    except Exception:
        return ""
    if not rows:
        return ""

    by_kind: dict[str, list[str]] = {}
    for r in rows:
        bucket = by_kind.setdefault(r["kind"], [])
        if len(bucket) >= INJECT_MAX_PER_KIND:
            continue
        mark = {"confirmed": "", "likely": " (likely)", "suspected": " (suspected)"}[r["confidence"]]
        detail = f" - {r['detail']}" if r["detail"] else ""
        bucket.append(f"{r['value']}{mark}{detail}")

    lines = [f"## Known intelligence for target: {target}",
             "(accumulated by earlier agent runs; treat 'suspected' as unverified)"]
    for kind in KINDS:
        if by_kind.get(kind):
            lines.append(f"{kind}s: " + "; ".join(by_kind[kind]))
    text = "\n".join(lines)
    if len(text) > INJECT_MAX_CHARS:
        text = text[:INJECT_MAX_CHARS] + "\n...[dossier truncated for prompt budget; query it with dossier_get]"
    return text


@tool
def dossier_note(
    value: str,
    kind: str = "note",
    target: str = "",
    detail: str = "",
    confidence: str = "suspected",
    source: str = "",
) -> str:
    """Record one durable fact about a target so future agent runs start already knowing it.

    Call this the moment you discover something worth keeping - an endpoint, a parameter, a role,
    a host, a technology, a finding - instead of only mentioning it in your final answer, which
    is discarded when your run ends. Anything recorded here is injected into the system prompt of
    every later sub-agent working this target, so the knowledge compounds across sessions.

    kind: endpoint, param, role, host, credential, finding, tech, note.
    confidence: confirmed (verified it yourself), likely, suspected (default - inferred, unverified).
    Be honest with confidence: only 'confirmed' facts are guaranteed to survive the injection
    budget. detail is a short qualifier ("POST, admin only"); source is where you saw it
    (file:line or URL). Re-recording the same value updates it instead of duplicating."""
    try:
        value = (value or "").strip()
        if not value:
            return core.to_json({"ok": False, "error": "value is required"})
        kind = (kind or "note").strip().lower()
        if kind not in KINDS:
            return core.to_json({"ok": False, "error": f"kind must be one of {list(KINDS)}"})
        confidence = (confidence or "suspected").strip().lower()
        if confidence not in CONFIDENCES:
            return core.to_json({"ok": False, "error": f"confidence must be one of {list(CONFIDENCES)}"})
        target = (target or active_target()).strip()
        if not target:
            return core.to_json({
                "ok": False,
                "error": "no target given and no active target set; pass target= or call dossier_set_active first",
            })

        now = core.now_str()
        conn = _connect()
        with conn:
            existing = conn.execute(
                "SELECT id, confidence, hits FROM dossier WHERE target = ? AND kind = ? AND value = ?",
                (target, kind, value[:500]),
            ).fetchone()
            if existing:
                # Confidence only ever ratchets UP: a later 'suspected' sighting must not
                # downgrade something an earlier run actually confirmed.
                rank = {c: i for i, c in enumerate(("suspected", "likely", "confirmed"))}
                best = max(existing["confidence"], confidence, key=lambda c: rank.get(c, 0))
                conn.execute(
                    "UPDATE dossier SET confidence = ?, hits = hits + 1, updated_at = ?,"
                    " detail = COALESCE(NULLIF(?, ''), detail), source = COALESCE(NULLIF(?, ''), source)"
                    " WHERE id = ?",
                    (best, now, detail.strip()[:300], source.strip()[:300], existing["id"]),
                )
                return core.to_json({"ok": True, "target": target, "kind": kind, "value": value[:500],
                                     "updated": True, "confidence": best, "hits": existing["hits"] + 1})
            conn.execute(
                "INSERT INTO dossier(target, kind, value, detail, confidence, source, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (target, kind, value[:500], detail.strip()[:300], confidence, source.strip()[:300], now, now),
            )
        return core.to_json({"ok": True, "target": target, "kind": kind, "value": value[:500], "created": True})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def dossier_get(target: str = "", kind: str = "", min_confidence: str = "", limit: int = 200) -> str:
    """Read what is known about a target: typed entries with confidence, detail, source, and how
    many times each was independently seen. Omit target to use the active one. Filter by kind
    (endpoint/param/role/host/credential/finding/tech/note) or by min_confidence
    (suspected/likely/confirmed) to see only what has been verified."""
    try:
        target = (target or active_target()).strip()
        if not target:
            return core.to_json({"ok": False, "error": "no target given and no active target set"})
        where, args = ["target = ?"], [target]
        if kind.strip():
            k = kind.strip().lower()
            if k not in KINDS:
                return core.to_json({"ok": False, "error": f"kind must be one of {list(KINDS)}"})
            where.append("kind = ?")
            args.append(k)
        if min_confidence.strip():
            mc = min_confidence.strip().lower()
            if mc not in CONFIDENCES:
                return core.to_json({"ok": False, "error": f"min_confidence must be one of {list(CONFIDENCES)}"})
            allowed = {"suspected": CONFIDENCES, "likely": ("likely", "confirmed"), "confirmed": ("confirmed",)}[mc]
            where.append(f"confidence IN ({','.join('?' * len(allowed))})")
            args.extend(allowed)
        clause = " AND ".join(where)

        conn = _connect()
        total = conn.execute(f"SELECT COUNT(*) FROM dossier WHERE {clause}", args).fetchone()[0]
        rows = conn.execute(
            f"SELECT kind, value, detail, confidence, source, hits, updated_at FROM dossier WHERE {clause}"
            f" ORDER BY kind, CASE confidence WHEN 'confirmed' THEN 0 WHEN 'likely' THEN 1 ELSE 2 END, hits DESC"
            f" LIMIT ?", args + [max(1, min(int(limit or 200), 1000))],
        ).fetchall()
        results = [dict(r) for r in rows]
        return core.to_json({
            "ok": True, "target": target, "matches_total": total, "returned": len(results),
            "capped": total > len(results), "entries": results,
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def dossier_set_active(target: str) -> str:
    """Set the active target whose dossier is injected into every sub-agent's system prompt in
    this workspace. Pass an empty string to stop injecting one. Set this at the start of a
    target engagement; sub-agents can then call dossier_note without repeating the target."""
    try:
        core.ensure_dirs()
        path = core.CACHE_DIR / ACTIVE_TARGET_FILENAME
        target = (target or "").strip()
        if not target:
            if path.exists():
                path.unlink()
            return core.to_json({"ok": True, "active_target": None, "cleared": True})
        path.write_text(target, encoding="utf-8")
        preview = render_for_prompt(target)
        return core.to_json({
            "ok": True, "active_target": target,
            "injected_chars": len(preview),
            "note": "This dossier is now prepended to every sub-agent's system prompt in this workspace.",
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def dossier_targets() -> str:
    """List every target that has a dossier, with entry counts and which one is active."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT target, COUNT(*) AS entries,"
            " SUM(CASE WHEN confidence = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,"
            " MAX(updated_at) AS last_updated"
            " FROM dossier GROUP BY target ORDER BY last_updated DESC"
        ).fetchall()
        return core.to_json({
            "ok": True, "active_target": active_target() or None,
            "count": len(rows), "targets": [dict(r) for r in rows], "db_path": str(_db_path()),
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def dossier_prune(target: str = "", confidence: str = "suspected", max_hits: int = 1) -> str:
    """Delete low-value dossier entries so the injected context stays worth its token cost.

    Defaults to removing 'suspected' entries seen only once - speculation that never got
    corroborated. This is the maintenance valve that stops a dossier becoming a junk drawer;
    run it occasionally on a long-running target. Confirmed entries are never touched unless
    you explicitly ask for that confidence level."""
    try:
        target = (target or active_target()).strip()
        if not target:
            return core.to_json({"ok": False, "error": "no target given and no active target set"})
        confidence = (confidence or "suspected").strip().lower()
        if confidence not in CONFIDENCES:
            return core.to_json({"ok": False, "error": f"confidence must be one of {list(CONFIDENCES)}"})
        conn = _connect()
        with conn:
            cur = conn.execute(
                "DELETE FROM dossier WHERE target = ? AND confidence = ? AND hits <= ?",
                (target, confidence, max(1, int(max_hits or 1))),
            )
            removed = cur.rowcount
        remaining = conn.execute("SELECT COUNT(*) FROM dossier WHERE target = ?", (target,)).fetchone()[0]
        return core.to_json({"ok": True, "target": target, "removed": removed, "remaining": remaining})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
