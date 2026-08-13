"""Deterministic structured extraction over source trees and JS bundles.

RAG answers "what text is semantically near this question". That is the wrong instrument for
the thing a bug bounty hunter actually wants out of a 40 MB bundle, which is not fuzzy at all:
secrets, endpoints, GraphQL operations, internal hosts. Those match patterns exactly.

It is doubly wrong *here*, because ``RAG_EMBED_MODEL`` is empty by default and
``RAG_HASH_FALLBACK`` is on (core.py), so "semantic" search over minified JS degrades to hash
buckets over tokens - close to useless on identifiers like ``a.qX``.

So this module does the deterministic half: run a ruleset once, store deduped rows in SQLite,
and let callers query a TABLE. An LLM sweep over the same bundle costs tokens every time it is
re-asked and gives a different answer each time; a ruleset is free to re-query and reproducible.
Findings are a FIRST PASS meant to be triaged (regexes have false-positive tails) - not a report.

Storage: ``<workspace>/.agent_cache/findings.sqlite3``, keyed so re-scanning a file replaces its
rows rather than duplicating them.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from langchain_core.tools import tool

from . import core

# =============================================================================
# Ruleset
# =============================================================================
# (kind, name, compiled regex, capture group used as the finding "value").
# Ordered most-specific first so a generic rule does not shadow a precise one.

def _rules() -> list[tuple[str, str, re.Pattern[str], int]]:
    r = re.compile
    return [
        # --- secrets -------------------------------------------------------
        ("secret", "aws_access_key_id", r(r"\b((?:AKIA|ASIA)[0-9A-Z]{16})\b"), 1),
        ("secret", "google_api_key", r(r"\b(AIza[0-9A-Za-z\-_]{35})\b"), 1),
        ("secret", "slack_token", r(r"\b(xox[baprs]-[0-9A-Za-z\-]{10,})"), 1),
        ("secret", "github_token", r(r"\b((?:ghp|gho|ghu|ghs|ghr|github_pat)_[0-9A-Za-z_]{20,})\b"), 1),
        ("secret", "stripe_key", r(r"\b((?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{10,})\b"), 1),
        ("secret", "private_key_block", r(r"(-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----)"), 1),
        ("secret", "jwt", r(r"\b(eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,})\b"), 1),
        ("secret", "bearer_literal", r(r"[Bb]earer\s+([A-Za-z0-9\-._~+/]{20,}=*)"), 1),
        ("secret", "generic_assignment", r(
            r"""(?:api[_-]?key|apikey|secret|password|passwd|token|auth[_-]?token|client[_-]?secret)"""
            r"""["']?\s*[:=]\s*["']([^"'\s]{8,120})["']""", re.I), 1),
        # --- endpoints -----------------------------------------------------
        ("endpoint", "absolute_url", r(r"""["'`](https?://[^"'`\s<>]{4,300})["'`]"""), 1),
        ("endpoint", "api_path", r(r"""["'`](/(?:api|v\d|rest|graphql|auth|admin|internal)/[^"'`\s<>]{0,200})["'`]"""), 1),
        ("endpoint", "fetch_call", r(r"""(?:fetch|axios(?:\.\w+)?|\.(?:get|post|put|patch|delete))\(\s*["'`]([^"'`]{2,300})["'`]""", re.I), 1),
        # --- graphql -------------------------------------------------------
        ("graphql", "operation", r(r"\b(?:query|mutation|subscription)\s+([A-Za-z_][A-Za-z0-9_]{2,80})\s*[({]"), 1),
        ("graphql", "gql_tag", r(r"gql`\s*(?:query|mutation|subscription)\s+([A-Za-z_][A-Za-z0-9_]{2,80})"), 1),
        # --- hosts ---------------------------------------------------------
        ("host", "internal_host", r(
            r"\b([a-z0-9][a-z0-9\-]{0,62}(?:\.[a-z0-9\-]{1,63})*\."
            r"(?:internal|local|corp|intranet|test|staging|dev|qa))\b", re.I), 1),
        ("host", "private_ip", r(
            r"\b((?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3})\b"), 1),
        ("host", "s3_bucket", r(r"\b([a-z0-9.\-]{3,63}\.s3(?:[.-][a-z0-9\-]+)*\.amazonaws\.com)\b", re.I), 1),
        # --- flags ---------------------------------------------------------
        ("flag", "feature_flag", r(
            r"""["'`]((?:feature|ff|flag|enable|is)[_.\-][A-Za-z0-9_.\-]{3,60})["'`]""", re.I), 1),
    ]


KINDS = ("secret", "endpoint", "graphql", "host", "flag")

# Substrings that make a "secret" obviously not one. Minified bundles are full of these, and a
# findings table nobody trusts is a findings table nobody reads.
_SECRET_NOISE = (
    "example", "changeme", "your_", "yourkey", "placeholder", "xxxxx", "<", "${", "{{",
    "process.env", "undefined", "null", "true", "false", "redacted", "dummy", "sample", "test_key",
)


def _looks_like_noise(kind: str, value: str) -> bool:
    low = value.lower()
    if kind == "secret":
        if any(bad in low for bad in _SECRET_NOISE):
            return True
        # A "secret" with no character diversity is almost always a mask or a repeated literal.
        if len(set(value)) <= 4:
            return True
    if kind == "endpoint" and low.startswith(("http://www.w3.org", "https://www.w3.org")):
        return True  # SVG namespaces, everywhere in bundles, never interesting
    return False


# =============================================================================
# Store
# =============================================================================

def _db_path() -> Path:
    return core.CACHE_DIR / "findings.sqlite3"


def _connect() -> sqlite3.Connection:
    core.ensure_dirs()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            rule TEXT NOT NULL,
            value TEXT NOT NULL,
            path TEXT NOT NULL,
            line INTEGER NOT NULL,
            context TEXT,
            collection TEXT NOT NULL DEFAULT 'default',
            created_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_findings_uniq
            ON findings(collection, kind, value, path, line);
        CREATE INDEX IF NOT EXISTS idx_findings_kind ON findings(collection, kind);
        CREATE INDEX IF NOT EXISTS idx_findings_path ON findings(collection, path);
        """
    )
    return conn


def _iter_files(pattern: str, max_files: int, max_file_bytes: int) -> tuple[list[Path], int, int]:
    matched, skipped_large = [], 0
    for p in core.WORKSPACE.glob(pattern.lstrip("/\\") or "**/*"):
        try:
            rp = p.resolve(strict=False)
            rp.relative_to(core.WORKSPACE)
        except Exception:
            continue
        if not rp.is_file():
            continue
        try:
            if rp.stat().st_size > max_file_bytes:
                skipped_large += 1
                continue
        except OSError:
            continue
        matched.append(rp)
    total = len(matched)
    return matched[:max_files], total, skipped_large


def _scan_text(text: str, rules: Iterable[tuple[str, str, re.Pattern[str], int]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    lines = text.splitlines()
    # Minified bundles are one enormous line; cap per-line work rather than the file, so a
    # single-line 30MB bundle is still scanned instead of silently skipped.
    for line_no, line in enumerate(lines, start=1):
        for kind, rule_name, rx, group in rules:
            for m in rx.finditer(line):
                try:
                    value = m.group(group)
                except IndexError:
                    continue
                if not value or _looks_like_noise(kind, value):
                    continue
                start = max(0, m.start() - 40)
                found.append({
                    "kind": kind,
                    "rule": rule_name,
                    "value": value[:500],
                    "line": line_no,
                    "context": line[start:m.end() + 40][:300],
                })
    return found


# =============================================================================
# Tools
# =============================================================================

@tool
def extract_findings(
    glob_pattern: str = "**/*.js",
    kinds: str = "",
    collection: str = "default",
    max_files: int = 500,
    max_file_bytes: int = 50_000_000,
    rescan: bool = True,
) -> str:
    """Run a deterministic ruleset over files matching a glob and store structured findings
    (secrets, endpoints, GraphQL operations, hosts, feature flags) in a queryable table.

    This is the cheap, reproducible first pass over a JS bundle, sourcemap dump, or source tree:
    it finds what regexes find exactly, so an agent is not paid to read chunk after chunk hunting
    for things a pattern matches. Query the results with findings_search / findings_stats - the
    findings themselves never have to cross into a model's context unless asked for.

    kinds is an optional comma-separated filter (secret, endpoint, graphql, host, flag); empty
    means all. collection namespaces a scan so several targets can coexist. Re-scanning a file
    replaces that file's rows rather than duplicating them.

    Findings are a FIRST PASS, not a report: regex rules have false positives (obvious
    placeholders are already dropped). Triage before acting on any secret."""
    try:
        wanted = {k.strip().lower() for k in (kinds or "").split(",") if k.strip()}
        bad = wanted - set(KINDS)
        if bad:
            return core.to_json({"ok": False, "error": f"unknown kinds {sorted(bad)}; valid: {list(KINDS)}"})
        rules = [r for r in _rules() if not wanted or r[0] in wanted]
        collection = core.safe_name(collection or "default")

        files, total_matched, skipped_large = _iter_files(
            glob_pattern, max(1, min(int(max_files or 500), 5000)), int(max_file_bytes or 50_000_000)
        )
        if not files:
            return core.to_json({
                "ok": False,
                "error": f"no files matched {glob_pattern!r} under {core.WORKSPACE}",
                "files_skipped_too_large": skipped_large,
            })

        conn = _connect()
        now = core.now_str()
        stored = 0
        scanned = 0
        per_kind: dict[str, int] = {}
        with conn:
            for rp in files:
                rel = core.rel_path(rp)
                try:
                    text = rp.read_bytes().decode("utf-8", errors="replace")
                except Exception:
                    continue
                scanned += 1
                if rescan:
                    conn.execute("DELETE FROM findings WHERE collection = ? AND path = ?", (collection, rel))
                for f in _scan_text(text, rules):
                    try:
                        conn.execute(
                            "INSERT OR IGNORE INTO findings"
                            "(kind, rule, value, path, line, context, collection, created_at)"
                            " VALUES(?,?,?,?,?,?,?,?)",
                            (f["kind"], f["rule"], f["value"], rel, f["line"], f["context"], collection, now),
                        )
                        stored += 1
                        per_kind[f["kind"]] = per_kind.get(f["kind"], 0) + 1
                    except sqlite3.Error:
                        continue

        out: dict[str, Any] = {
            "ok": True,
            "collection": collection,
            "files_matched": total_matched,
            "files_scanned": scanned,
            "files_skipped_over_cap": max(0, total_matched - len(files)),
            "files_skipped_too_large": skipped_large,
            "findings_stored": stored,
            "by_kind": per_kind,
            "next": "query with findings_search(kind=..., query=...) or findings_stats()",
        }
        if total_matched > len(files):
            out["note"] = (
                f"{total_matched} files matched but only {len(files)} were scanned "
                f"(max_files). {total_matched - len(files)} were NOT examined."
            )
        return core.to_json(out)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def findings_search(
    kind: str = "",
    query: str = "",
    collection: str = "default",
    path_contains: str = "",
    limit: int = 100,
    distinct_values: bool = False,
) -> str:
    """Query the structured findings table produced by extract_findings.

    Returns rows (kind, rule, value, path, line, context), or with distinct_values=true just the
    unique values and how many times each occurs - which is usually what you want first, since
    one endpoint appears in fifty chunks. Always reports matches_total separately from the
    number returned, so a capped listing can never be mistaken for the true total."""
    try:
        collection = core.safe_name(collection or "default")
        limit = max(1, min(int(limit or 100), 1000))
        where = ["collection = ?"]
        args: list[Any] = [collection]
        if kind.strip():
            k = kind.strip().lower()
            if k not in KINDS:
                return core.to_json({"ok": False, "error": f"kind must be one of {list(KINDS)}"})
            where.append("kind = ?")
            args.append(k)
        if query.strip():
            where.append("value LIKE ?")
            args.append(f"%{query.strip()}%")
        if path_contains.strip():
            where.append("path LIKE ?")
            args.append(f"%{path_contains.strip()}%")
        clause = " AND ".join(where)

        conn = _connect()
        if distinct_values:
            total = conn.execute(
                f"SELECT COUNT(*) FROM (SELECT DISTINCT kind, value FROM findings WHERE {clause})", args
            ).fetchone()[0]
            rows = conn.execute(
                f"SELECT kind, value, COUNT(*) AS occurrences, MIN(path) AS first_path, MIN(line) AS first_line"
                f" FROM findings WHERE {clause} GROUP BY kind, value ORDER BY occurrences DESC LIMIT ?",
                args + [limit],
            ).fetchall()
        else:
            total = conn.execute(f"SELECT COUNT(*) FROM findings WHERE {clause}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT kind, rule, value, path, line, context FROM findings WHERE {clause}"
                f" ORDER BY kind, value LIMIT ?", args + [limit],
            ).fetchall()

        results = [dict(r) for r in rows]
        out: dict[str, Any] = {
            "ok": True,
            "collection": collection,
            "matches_total": total,
            "returned": len(results),
            "capped": total > len(results),
            "results": results,
        }
        if out["capped"]:
            out["note"] = f"LISTING TRUNCATED: {total} rows match, {len(results)} returned. True total is matches_total."
        return core.to_json(out)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def findings_stats(collection: str = "default") -> str:
    """Summarize the findings table for a collection: totals by kind and by rule, plus the
    files contributing the most findings. The cheap orientation call to make before drilling in
    with findings_search."""
    try:
        collection = core.safe_name(collection or "default")
        conn = _connect()
        by_kind = {r["kind"]: r["n"] for r in conn.execute(
            "SELECT kind, COUNT(*) AS n FROM findings WHERE collection = ? GROUP BY kind ORDER BY n DESC",
            (collection,))}
        by_rule = {r["rule"]: r["n"] for r in conn.execute(
            "SELECT rule, COUNT(*) AS n FROM findings WHERE collection = ? GROUP BY rule ORDER BY n DESC LIMIT 25",
            (collection,))}
        top_files = [dict(r) for r in conn.execute(
            "SELECT path, COUNT(*) AS n FROM findings WHERE collection = ? GROUP BY path ORDER BY n DESC LIMIT 10",
            (collection,))]
        total = conn.execute("SELECT COUNT(*) FROM findings WHERE collection = ?", (collection,)).fetchone()[0]
        collections = [r[0] for r in conn.execute("SELECT DISTINCT collection FROM findings")]
        return core.to_json({
            "ok": True, "collection": collection, "total": total,
            "by_kind": by_kind, "by_rule": by_rule, "top_files": top_files,
            "all_collections": collections, "db_path": str(_db_path()),
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
