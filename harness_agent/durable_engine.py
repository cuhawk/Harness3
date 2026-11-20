"""Durable execution of explicit, ordered, JSON-serializable tool calls."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .tenant_policy import Approval, Principal, TenantPolicy, ToolCall, canonical, fingerprint


@dataclass(frozen=True)
class Step:
    id: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Tool:
    """Use the supplied transaction for local effects, or deduplicate at the provider.

    An external provider must atomically store the operation key with its effect and
    return the original result when the key is seen again.
    """

    invoke: Callable[[dict[str, Any], str, sqlite3.Connection], Any]
    high_impact: bool = False


class DurableEngine:
    """Advance one checkpoint per transaction. Workers serialize via SQLite's writer lock."""

    def __init__(
        self, path: str | Path, policy: TenantPolicy, tools: Mapping[str, Tool], *,
        max_attempts: int = 4, base_delay: float = 1, max_delay: float = 60,
        clock: Callable[[], float] = time.time,
    ):
        if not 1 <= max_attempts <= 100 or not (
            math.isfinite(base_delay) and 0 < base_delay <= max_delay and math.isfinite(max_delay)
        ):
            raise ValueError("invalid retry limits")
        self.path = path
        self.policy = policy
        self.tools = dict(tools)
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.clock = clock
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS durable_runs (
                    tenant TEXT NOT NULL, run_id TEXT NOT NULL, subject TEXT NOT NULL,
                    definition TEXT NOT NULL, next_step INTEGER NOT NULL DEFAULT 0,
                    results TEXT NOT NULL DEFAULT '[]', attempts INTEGER NOT NULL DEFAULT 0,
                    ready_at REAL NOT NULL DEFAULT 0, status TEXT NOT NULL DEFAULT 'pending',
                    error_type TEXT, PRIMARY KEY (tenant, run_id)
                );
                CREATE INDEX IF NOT EXISTS durable_queue
                ON durable_runs (tenant, subject, status, ready_at, run_id);
            """)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    @contextmanager
    def _transaction(self):
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def run(
        self, principal: Principal, run_id: str, steps: Sequence[Step], *,
        version: str = "1", approvals: Mapping[str, Approval] | None = None,
    ) -> dict:
        """Run until complete, awaiting approval, scheduled for retry, or dead-lettered."""
        plan = [
            {"id": step.id, "tool": step.tool, "arguments": step.arguments}
            for step in steps
        ]
        if not run_id or not principal.tenant or not principal.subject or not plan or not version:
            raise ValueError("principal, run ID, steps and version are required")
        if len({step["id"] for step in plan}) != len(plan) or any(
            not step["id"] or step["tool"] not in self.tools or not isinstance(step["arguments"], dict)
            for step in plan
        ):
            raise ValueError("steps require unique IDs, registered tools and object arguments")
        definition = canonical([version, plan, self.max_attempts, self.base_delay, self.max_delay])
        plan = json.loads(definition)[1]
        key = (principal.tenant, run_id)
        with self._transaction() as db:
            db.execute(
                """INSERT INTO durable_runs (tenant, run_id, subject, definition)
                VALUES (?, ?, ?, ?) ON CONFLICT (tenant, run_id) DO NOTHING""",
                (*key, principal.subject, definition),
            )
        while True:
            with self._transaction() as db:
                row = db.execute(
                    "SELECT * FROM durable_runs WHERE tenant=? AND run_id=?", key
                ).fetchone()
                if row["subject"] != principal.subject:
                    raise PermissionError("workflow belongs to another subject")
                if row["definition"] != definition:
                    raise ValueError("workflow definition changed for existing run")
                outcome = self._advance(db, row, principal, plan, approvals or {})
                if outcome is not None:
                    return outcome

    def _advance(self, db, row, principal, plan, approvals):
        index = row["next_step"]
        results = json.loads(row["results"])
        key = (principal.tenant, row["run_id"])
        if row["status"] in {"completed", "dead_letter"} or row["ready_at"] > self.clock():
            return {"status": row["status"], "results": results, "next_step": index,
                    "attempts": row["attempts"], "ready_at": row["ready_at"],
                    "error_type": row["error_type"]}
        step = plan[index]
        tool = self.tools[step["tool"]]
        operation = fingerprint(["harness-step-v1", *key, row["definition"], index])
        call = ToolCall.bind(principal, operation, step["tool"], step["arguments"])
        decision = self.policy.authorize(call, approvals.get(step["id"]), high_impact=tool.high_impact)
        if not decision["ok"]:
            status = "blocked" if decision["decision"] == "rbac_denied" else "awaiting_approval"
            db.execute(
                "UPDATE durable_runs SET status=? WHERE tenant=? AND run_id=?", (status, *key)
            )
            return {"status": status, "step": step["id"], "next_step": index,
                    "results": results, "authorization": decision}

        db.execute("SAVEPOINT tool_effect")
        try:
            result = tool.invoke(step["arguments"], operation, db)
            checkpoint = canonical([*results, result])
        except Exception as exc:
            db.execute("ROLLBACK TO tool_effect")
            db.execute("RELEASE tool_effect")
            attempts = row["attempts"] + 1
            status = "dead_letter" if attempts >= self.max_attempts else "retry"
            ready_at = 0 if status == "dead_letter" else self.clock() + min(
                self.max_delay, self.base_delay * 2 ** (attempts - 1)
            )
            error_type = type(exc).__name__
            db.execute(
                """UPDATE durable_runs SET status=?, attempts=?, ready_at=?, error_type=?
                WHERE tenant=? AND run_id=?""",
                (status, attempts, ready_at, error_type, *key),
            )
            return {"status": status, "results": results, "next_step": index,
                    "attempts": attempts, "ready_at": ready_at, "error_type": error_type}

        db.execute("RELEASE tool_effect")
        db.execute(
            """UPDATE durable_runs SET next_step=?, results=?, status=?, attempts=0,
            ready_at=0, error_type=NULL WHERE tenant=? AND run_id=?""",
            (index + 1, checkpoint, "completed" if index + 1 == len(plan) else "pending", *key),
        )
        if index + 1 == len(plan):
            return {"status": "completed", "results": json.loads(checkpoint),
                    "next_step": index + 1, "attempts": 0, "ready_at": 0, "error_type": None}
        return None

    def resume(self, principal: Principal, run_id: str, *, approvals=None) -> dict:
        """Load the stored plan for a queued or interrupted run."""
        with closing(self._connect()) as db:
            row = db.execute(
                "SELECT definition FROM durable_runs WHERE tenant=? AND subject=? AND run_id=?",
                (principal.tenant, principal.subject, run_id),
            ).fetchone()
        if row is None:
            raise KeyError(run_id)
        version, plan, *_ = json.loads(row["definition"])
        return self.run(principal, run_id, [Step(**step) for step in plan],
                        version=version, approvals=approvals)

    def queue(self, principal: Principal, *, dead_letters: bool = False, limit: int = 100) -> list[str]:
        """Return a bounded batch of ready runs or terminal failures for this principal."""
        if not 1 <= limit <= 1000:
            raise ValueError("queue limit must be between 1 and 1000")
        with closing(self._connect()) as db:
            rows = db.execute(
                """SELECT run_id FROM durable_runs
                WHERE tenant=? AND subject=? AND status IN (?, ?) AND ready_at<=?
                ORDER BY ready_at, run_id LIMIT ?""",
                (principal.tenant, principal.subject,
                 "dead_letter" if dead_letters else "pending",
                 "dead_letter" if dead_letters else "retry", self.clock(), limit),
            )
            return [row["run_id"] for row in rows]
