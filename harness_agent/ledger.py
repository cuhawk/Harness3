"""Delegation outcome ledger: one row per delegated run, so "does this model actually work
agentically for this shape of task" is a QUERY instead of an anecdote.

Why this exists
---------------
The interesting failures of a free-tier model pool are behavioural, not transport-level, and
they only show up over a run of delegations. Two found by hand: one model returns a clean EMPTY
answer in a real agent loop while passing single-turn probes, another emits corrupted tool-call
syntax as its final answer roughly one run in three. Findings like that used to live in a prose
document that goes stale the moment the router config changes. Here they are rows.

This is NOT about cost. The models are free; what is expensive is a delegation that comes back
confidently wrong and has to be re-done by hand.

task_shape - the bucket key
---------------------------
A coarse, stable, deliberately NON-semantic fingerprint. There is no embedding here and there
must not be one: a shape is only useful if the same kind of task lands in the same bucket run
after run, and a nearest-neighbour label is neither reproducible nor explainable.

    task_shape = "<verb>|<group>+<group>+...|<glob|noglob>"

  - verb   the first alphabetic token of the task, lowercased, mapped through the CLOSED
           synonym table below to one of _VERBS. Anything unrecognised collapses to "other".
           Closed on purpose: an open vocabulary yields one bucket per phrasing, and a table of
           1-run buckets answers no question at all.
  - groups tools_registry.select_group_names(task) - the same scorer the run itself used to
           decide which tool groups to bind - joined with "+", or "core" when it picked none.
  - glob   whether the caller reached this run by expanding a glob (spawn_map_reduce), or the
           task text itself contains a glob-ish pattern.

Changing that formula re-buckets the entire table: old rows keep their old shape strings and
stop joining with new ones. Treat it as a schema change, not a tweak.

Honesty note for the reader of delegation_stats
-----------------------------------------------
The `status` column is copied verbatim from the completeness envelope; this module never
derives a verdict of its own (see completeness.compute_status, the single site that does). Read
its clean verdict as NO CAP SIGNAL WAS OBSERVED, not as proof the run was exhaustive - several
tools truncate with no machine-readable signal, and none of that is visible from here.

The status strings themselves are deliberately not written out anywhere in this module: they
live in completeness.py alone, so no second code path can ever hand-stamp a clean verdict.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import core

# Kill switch for anything that must run a delegation with no disk side effects at all (the
# test suite sets it so FakeRuntime runs do not seed the operator's real ledger with junk).
DISABLE_ENV = "HARNESS_LEDGER_DISABLE"

# The closed verb vocabulary. Only these strings ever appear in a shape's first segment.
_VERBS = (
    "read", "search", "list", "count", "summarize", "extract",
    "classify", "write", "edit", "run", "verify", "explain", "other",
)

# Synonym -> canonical verb. Kept small and boring: every entry added here re-buckets the tasks
# that used to collapse to "other", so add deliberately rather than exhaustively.
_VERB_SYNONYMS: dict[str, str] = {
    "read": "read", "open": "read", "view": "read", "show": "read", "inspect": "read",
    "examine": "read", "look": "read", "cat": "read", "print": "read",

    "search": "search", "find": "search", "grep": "search", "locate": "search",
    "scan": "search", "hunt": "search", "match": "search",

    "list": "list", "enumerate": "list", "index": "list", "catalog": "list",

    "count": "count", "tally": "count", "measure": "count", "total": "count",

    "summarize": "summarize", "summarise": "summarize", "describe": "summarize",
    "recap": "summarize", "condense": "summarize", "brief": "summarize", "digest": "summarize",

    "extract": "extract", "pull": "extract", "collect": "extract", "gather": "extract",
    "harvest": "extract", "mine": "extract", "parse": "extract",

    "classify": "classify", "categorize": "classify", "categorise": "classify",
    "label": "classify", "triage": "classify", "rank": "classify", "group": "classify",
    "sort": "classify", "bucket": "classify",

    "write": "write", "create": "write", "generate": "write", "produce": "write",
    "draft": "write", "add": "write", "make": "write", "build": "write", "emit": "write",

    "edit": "edit", "update": "edit", "modify": "edit", "change": "edit", "fix": "edit",
    "refactor": "edit", "rename": "edit", "patch": "edit", "replace": "edit", "remove": "edit",
    "delete": "edit",

    "run": "run", "execute": "run", "invoke": "run", "launch": "run", "start": "run",
    "install": "run", "call": "run",

    "verify": "verify", "check": "verify", "validate": "verify", "confirm": "verify",
    "test": "verify", "audit": "verify", "review": "verify", "compare": "verify",
    "diff": "verify", "assert": "verify",

    "explain": "explain", "analyze": "explain", "analyse": "explain", "why": "explain",
    "how": "explain", "investigate": "explain", "reason": "explain", "determine": "explain",
    "identify": "explain",
}

# A glob-ish pattern in the task text: "**/", or a "*" glued to a path-like character. A bare
# "*" surrounded by spaces is prose (a bullet, a multiplication sign), not a glob.
_GLOB_RE = re.compile(r"\*\*/|[\w./\\-]\*|\*[\w./\\-]")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS delegations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    task_shape TEXT NOT NULL,
    model TEXT NOT NULL,
    ok INTEGER NOT NULL,
    accepted INTEGER NOT NULL,
    attempts INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,
    rejection_reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    validator_kinds TEXT NOT NULL DEFAULT '',
    entry_point TEXT NOT NULL DEFAULT '',
    workspace TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS delegation_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    delegation_id INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    model TEXT NOT NULL,
    ok INTEGER NOT NULL,
    accepted INTEGER NOT NULL,
    reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_deleg_shape_model ON delegations(task_shape, model);
CREATE INDEX IF NOT EXISTS idx_deleg_ts ON delegations(ts);
CREATE INDEX IF NOT EXISTS idx_attempts_parent ON delegation_attempts(delegation_id);
"""


# =============================================================================
# Store
# =============================================================================

def _db_path() -> Path:
    # A function, not a module constant: core.set_workspace rebinds CACHE_DIR, and a constant
    # captured at import time would keep writing into the workspace the process started in.
    return core.CACHE_DIR / "delegations.sqlite3"


def _connect() -> sqlite3.Connection:
    """Open a fresh connection with the schema in place.

    Two deliberate departures from extract_tools' otherwise-identical convention, both forced by
    threading: the ledger is written from spawn_map_reduce / spawn_subagents_parallel WORKER
    THREADS, while extract_tools is only ever called from one.
      - every call opens its own connection and the caller closes it in a finally; sqlite3
        connections may not be shared across threads.
      - WAL plus a real busy timeout (precedent: memory.py, rag.py), so N concurrent fan-out
        writers queue instead of raising "database is locked" - which on Windows is otherwise
        the normal outcome of a 30-file map_reduce finishing at once.
    """
    core.ensure_dirs()
    conn = sqlite3.connect(_db_path(), timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


# =============================================================================
# Shape fingerprint
# =============================================================================

def _leading_verb(task: str) -> str:
    match = re.search(r"[A-Za-z]+", task or "")
    if not match:
        return "other"
    return _VERB_SYNONYMS.get(match.group(0).lower(), "other")


def task_shape(task: str, glob_involved: bool = False) -> str:
    """Coarse, stable bucket key for a task - see the module docstring for the formula.

    Deterministic and explicitly not an embedding: the same task text always produces the same
    string, and the string can be read back to a human without a model in the loop.
    """
    # Lazy import: tools_registry imports this module (to register delegation_stats), so a
    # module-level import here would be a cycle.
    from . import tools_registry

    text = task or ""
    groups = tools_registry.select_group_names(text)
    has_glob = bool(glob_involved) or bool(_GLOB_RE.search(text))
    return f"{_leading_verb(text)}|{'+'.join(groups) or 'core'}|{'glob' if has_glob else 'noglob'}"


# =============================================================================
# Write path (best effort, never raises, never blocks the caller's result)
# =============================================================================

def record_delegation(
    *,
    task: str,
    model: str,
    ok: bool,
    accepted: bool,
    attempts: int,
    latency_ms: int,
    rejection_reason: str = "",
    status: str = "",
    validator_kinds: list[str] | None = None,
    entry_point: str = "",
    glob_involved: bool = False,
    validation_history: list[dict[str, Any]] | None = None,
) -> None:
    """Append one delegation (plus its per-attempt detail) to the ledger.

    The ENTIRE body is best-effort. A ledger that cannot be opened, a disk that is full, a
    schema that somehow drifted - none of it may change or delay the delegation result the
    caller is already holding. Bookkeeping that can break the thing it observes is worse than
    no bookkeeping, so this function swallows everything and returns.
    """
    if os.environ.get(DISABLE_ENV):
        return
    conn = None
    try:
        shape = task_shape(task, glob_involved)
        kinds = ",".join(sorted(str(k) for k in (validator_kinds or []) if k))
        conn = _connect()
        with conn:
            cur = conn.execute(
                "INSERT INTO delegations (ts, task_shape, model, ok, accepted, attempts,"
                " latency_ms, rejection_reason, status, validator_kinds, entry_point, workspace)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    core.now_str(), shape, str(model or ""), int(bool(ok)), int(bool(accepted)),
                    int(attempts or 0), int(latency_ms or 0),
                    core.trim_text(rejection_reason or "", 2000), str(status or ""),
                    kinds, str(entry_point or ""), str(core.WORKSPACE),
                ),
            )
            parent = cur.lastrowid
            conn.executemany(
                "INSERT INTO delegation_attempts (delegation_id, attempt, model, ok, accepted, reason)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (
                        parent, int(h.get("attempt") or 0), str(h.get("model") or ""),
                        int(bool(h.get("ok"))), int(bool(h.get("accepted"))),
                        core.trim_text(str(h.get("reason") or h.get("error") or ""), 1000),
                    )
                    for h in (validation_history or [])
                    if isinstance(h, dict)
                ],
            )
    except Exception:
        # Deliberately silent: even logging here could raise (closed stream, encoding), and the
        # caller is a delegation, not a maintenance job.
        return
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# =============================================================================
# Read tool
# =============================================================================

def _median(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return int(ordered[mid])
    return int((ordered[mid - 1] + ordered[mid]) / 2)


@tool
def delegation_stats(
    task_shape: str = "",
    model: str = "",
    since_days: int = 0,
    min_runs: int = 1,
    max_rows: int = 50,
) -> str:
    """Query the delegation outcome ledger: per (task_shape, model), how often delegations
    actually succeeded, how many attempts they needed, and how long they took.

    This is the answer to "which model should I hand this kind of work to" - measured on THIS
    workspace's real runs rather than remembered from a prose note. A model that passes a
    single-turn probe can still return empty answers 3/3 times in a real agent loop, and that
    only shows up as a success rate.

    task_shape is a coarse fingerprint of the kind of work: "<verb>|<tool groups>|<glob|noglob>",
    e.g. "search|extract|glob". Pass a shape or a model to filter, since_days to look at recent
    runs only, min_runs to hide buckets too small to mean anything.

    Caveat on status_counts: a clean completeness verdict means no tool REPORTED a cap during
    the run, not that the run was exhaustive - several tools truncate with no signal at all."""
    try:
        where: list[str] = []
        args: list[Any] = []
        # The "{d}" is a table-alias slot: the same conditions are reused in a join against
        # delegation_attempts, where every one of these column names is ambiguous.
        if (task_shape or "").strip():
            where.append("{d}task_shape LIKE ?")
            args.append(f"%{task_shape.strip()}%")
        if (model or "").strip():
            where.append("{d}model = ?")
            args.append(model.strip())
        if int(since_days or 0) > 0:
            cutoff = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(time.time() - int(since_days) * 86400)
            )
            # ts is stored in "%Y-%m-%d %H:%M:%S" local time, which sorts lexicographically.
            where.append("{d}ts >= ?")
            args.append(cutoff)
        clause = (" WHERE " + " AND ".join(where)).format(d="") if where else ""

        min_runs = max(1, int(min_runs or 1))
        max_rows = max(1, min(int(max_rows or 50), 500))

        conn = _connect()
        try:
            runs_total = conn.execute(f"SELECT COUNT(*) FROM delegations{clause}", args).fetchone()[0]
            group_rows = conn.execute(
                f"SELECT task_shape, model, COUNT(*) AS runs FROM delegations{clause}"
                " GROUP BY task_shape, model HAVING runs >= ? ORDER BY runs DESC, task_shape, model",
                args + [min_runs],
            ).fetchall()
            groups_total = len(group_rows)

            # Per-bucket detail is fetched row by row rather than aggregated in SQL: the attempt
            # distribution and the median both need the individual values, and SQLite has no
            # median. max_rows bounds how many of these run.
            group_conds = where + ["{d}task_shape = ?", "{d}model = ?"]
            group_clause = (" WHERE " + " AND ".join(group_conds)).format(d="")
            attempt_clause = (" WHERE " + " AND ".join(group_conds)).format(d="d.")
            stats = []
            for g in group_rows[:max_rows]:
                detail = conn.execute(
                    "SELECT ok, accepted, attempts, latency_ms, status, rejection_reason"
                    f" FROM delegations{group_clause}",
                    args + [g["task_shape"], g["model"]],
                ).fetchall()
                runs = len(detail)
                attempt_dist: dict[str, int] = {}
                status_counts: dict[str, int] = {}
                reasons: dict[str, int] = {}
                for r in detail:
                    attempt_dist[str(r["attempts"])] = attempt_dist.get(str(r["attempts"]), 0) + 1
                    st = r["status"] or "unknown"
                    status_counts[st] = status_counts.get(st, 0) + 1
                # Rejection reasons come from the ATTEMPT table, not from the run's final
                # rejection_reason. A run that was rejected twice and accepted on the third
                # attempt has ok=1 and an empty rejection_reason, so aggregating over runs
                # reports "no rejections" for exactly the retry-churn case worth measuring.
                for a in conn.execute(
                    "SELECT a.reason AS reason, COUNT(*) AS n FROM delegation_attempts a"
                    f" JOIN delegations d ON d.id = a.delegation_id{attempt_clause}"
                    " AND a.accepted = 0 AND a.reason != ''"
                    " GROUP BY a.reason ORDER BY n DESC, a.reason LIMIT 3",
                    args + [g["task_shape"], g["model"]],
                ).fetchall():
                    reasons[str(a["reason"])[:160]] = int(a["n"])
                retried = sum(1 for r in detail if int(r["attempts"] or 0) > 1)
                stats.append({
                    "task_shape": g["task_shape"],
                    "model": g["model"],
                    "runs": runs,
                    "ok_rate": round(sum(1 for r in detail if r["ok"]) / runs, 3) if runs else 0.0,
                    "accepted_rate": round(sum(1 for r in detail if r["accepted"]) / runs, 3) if runs else 0.0,
                    "retry_rate": round(retried / runs, 3) if runs else 0.0,
                    "attempt_distribution": dict(sorted(attempt_dist.items())),
                    "median_latency_ms": _median([int(r["latency_ms"] or 0) for r in detail]),
                    "status_counts": status_counts,
                    "top_rejection_reasons": [
                        {"reason": k, "count": v}
                        for k, v in sorted(reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    ],
                })
        finally:
            conn.close()

        out: dict[str, Any] = {
            "ok": True,
            "runs_total": runs_total,
            "matches_total": groups_total,
            "returned": len(stats),
            "capped": groups_total > len(stats),
            "db_path": str(_db_path()),
            "stats": stats,
        }
        if out["capped"]:
            out["note"] = (
                f"LISTING TRUNCATED: {groups_total} (shape, model) buckets match, "
                f"{len(stats)} returned. True total is matches_total."
            )
        return core.to_json(out)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
