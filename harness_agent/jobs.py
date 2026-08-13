"""Fire-and-forget delegation: start a sub-agent, get an id back immediately, poll it cheaply.

``spawn_subagent`` blocks until the sub-agent finishes. That is right for a 10-second task and
wrong for a 20-minute one: the caller sits idle, and a client-side tool timeout kills a job that
was doing fine. These tools run the same delegation on a background thread and return at once.

State lives in ``<workspace>/.agent_cache/jobs/<job_id>.json`` (atomic replace, same pattern as
sessions.SessionStore) rather than in memory, so a poll is a file read, a finished job's result
survives, and nothing is lost if the caller stops polling. The *thread*, however, dies with this
MCP server process - i.e. with the Claude Code session. Work that must outlive the session
belongs in automations.py (detached), not here.

Results are written to the job file, not returned inline, so a huge answer never lands in the
caller's context by surprise: ``check_job`` returns a bounded head plus a path.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from . import completeness
from . import core
from . import permissions

_HEAD_CHARS = 2000


def _jobs_dir() -> Path:
    d = core.CACHE_DIR / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _job_path(job_id: str) -> Path:
    return _jobs_dir() / f"{core.safe_name(job_id)}.json"


def _write(job_id: str, data: dict[str, Any]) -> None:
    path = _job_path(job_id)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def _read(job_id: str) -> dict[str, Any] | None:
    try:
        return json.loads(_job_path(job_id).read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def _progress_path(job_id: str) -> Path:
    return _jobs_dir() / f"{core.safe_name(job_id)}.progress"


def log_progress(job_id: str, message: str) -> None:
    """Append one timestamped line to a job's progress log.

    A job file only flips running -> done, which tells a poller nothing about a 20-minute run.
    This is an append-only sidecar (never rewritten, so a reader can tail it safely while the
    worker writes) that check_job surfaces, turning fire-and-pray into fire-and-watch.
    """
    if not job_id:
        return
    try:
        with _progress_path(job_id).open("a", encoding="utf-8") as fh:
            fh.write(f"{core.now_str()}  {str(message).strip()[:500]}\n")
    except Exception:
        pass  # progress logging must never break the job it is reporting on


def _progress_tail(job_id: str, lines: int = 12) -> list[str]:
    try:
        content = _progress_path(job_id).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    rows = [ln for ln in content.splitlines() if ln.strip()]
    return rows[-max(1, lines):]


def _run_job(job_id: str, record: dict[str, Any], kind: str, kwargs: dict[str, Any]) -> None:
    """Worker body. A contextvar does not cross a thread boundary, so re-mark this thread as
    background or every ask-policy tool the sub-agent uses would fall back to interactive
    handling that cannot work here."""
    permissions.mark_background()
    from . import subagent

    log_progress(job_id, f"started {kind} on model={record.get('model')}")
    started = time.time()
    try:
        if kind == "parallel":
            raw = subagent.spawn_subagents_parallel.invoke(kwargs)
        elif kind == "map_reduce":
            raw = subagent.spawn_map_reduce.invoke(kwargs)
        else:
            raw = subagent.spawn_subagent.invoke(kwargs)
        try:
            payload = json.loads(raw)
        except Exception:
            # An unparseable reply from a delegation tool means core.to_json trimmed it at
            # MAX_OUTPUT_BYTES - i.e. bytes were lost, and the completeness verdict was in them.
            # Calling that ok with a null status would restore the exact laundering this envelope
            # exists to stop, on the path (a big async sweep) where it matters most.
            payload = {
                "ok": True,
                "result": str(raw),
                **completeness.envelope(True, [{
                    "tool": kind, "reason": "tool_output_trimmed_256kb",
                    "returned": len(str(raw)), "total": None,
                }]),
            }
        record["status"] = "done"
        record["ok"] = bool(payload.get("ok", True))
        record["result"] = payload
        # Hoisted out of the payload because check_job's default view replaces the whole result
        # with a 2000-char head stub - a coverage verdict buried inside it would only be visible
        # to a caller who already knew to ask for full=true. Named delegation_status because
        # this record's own "status" already means running/done/error.
        record["delegation_status"] = payload.get("status")
        record["coverage"] = payload.get("coverage")
        log_progress(job_id, f"finished ok={record['ok']} in {round(time.time()-started,1)}s")
    except Exception as exc:
        record["status"] = "error"
        record["ok"] = False
        record["error"] = repr(exc)
        log_progress(job_id, f"FAILED after {round(time.time()-started,1)}s: {exc!r}")
    record["finished_at"] = core.now_str()
    record["elapsed_seconds"] = round(time.time() - started, 1)
    _write(job_id, record)


def _start(kind: str, label: str, model_name: str, kwargs: dict[str, Any]) -> str:
    core.ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    record: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "label": core.trim_text(label, 300),
        "model": model_name or core.DEFAULT_MODEL,
        "workspace": str(core.WORKSPACE),
        "status": "running",
        "started_at": core.now_str(),
    }
    _write(job_id, record)
    threading.Thread(
        target=_run_job, args=(job_id, record, kind, kwargs), daemon=True, name=f"job-{job_id}"
    ).start()
    return job_id


@tool
def spawn_subagent_async(
    task: str,
    model_name: str = "",
    session_name: str = "",
    max_result_chars: int = 6000,
    min_length: int = 0,
    require_json: bool = False,
    require_keys_json: str = "",
    judge_model: str = "",
    max_attempts: int = 0,
    require_citations: bool = False,
    citation_min: int = 1,
) -> str:
    """Start a delegated sub-task in the BACKGROUND and return a job_id immediately, instead of
    blocking until it finishes like spawn_subagent does.

    Use for work measured in minutes - a long scan, a sweep over a big tree, anything you would
    otherwise wait on. Do something else, then call check_job(job_id) to collect it. Same task
    contract as spawn_subagent, and the acceptance criteria exposed above (min_length,
    require_json/require_keys_json, judge_model, require_citations/citation_min, max_attempts) -
    require_substrings_json, require_regex and escalate_models are NOT exposed here; use the
    blocking spawn_subagent when you need them.

    The background thread dies with this MCP server process (i.e. with your session), so this is
    for long-running, not overnight - use an automation for that. Poll with check_job; the full
    result is stored on disk, so a slow poll never loses it."""
    if not (task or "").strip():
        return core.to_json({"ok": False, "error": "task is required"})
    try:
        job_id = _start("single", task, model_name, {
            "task": task,
            "model_name": model_name,
            "session_name": session_name,
            "max_result_chars": max_result_chars,
            "min_length": min_length,
            "require_json": require_json,
            "require_keys_json": require_keys_json,
            "judge_model": judge_model,
            "max_attempts": max_attempts,
            # This dict IS the call: _run_job invokes the tool with it, so a parameter added to
            # the signature above and not here is accepted from the caller and then dropped.
            "require_citations": require_citations,
            "citation_min": citation_min,
        })
        return core.to_json({
            "ok": True,
            "job_id": job_id,
            "status": "running",
            "next": f"call check_job with job_id={job_id}",
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def spawn_subagents_parallel_async(
    tasks_json: str, model_name: str = "", max_result_chars: int = 4000
) -> str:
    """Start a parallel fan-out in the BACKGROUND and return a job_id immediately. Same as
    spawn_subagents_parallel (tasks_json is a JSON list of task strings, each run by its own
    isolated sub-agent concurrently) but non-blocking - collect it later with check_job."""
    # No acceptance criteria are exposed here at all (unlike the two async wrappers above), so
    # citations are deliberately left off too - adding them as this tool's SOLE quality gate
    # would be an odd, inconsistent surface. Add the whole criteria set or none.
    if not (tasks_json or "").strip():
        return core.to_json({"ok": False, "error": "tasks_json is required"})
    try:
        job_id = _start("parallel", tasks_json, model_name, {
            "tasks_json": tasks_json,
            "model_name": model_name,
            "max_result_chars": max_result_chars,
        })
        return core.to_json({
            "ok": True,
            "job_id": job_id,
            "status": "running",
            "next": f"call check_job with job_id={job_id}",
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def spawn_map_reduce_async(
    glob_pattern: str,
    map_prompt: str,
    reduce_prompt: str = "",
    model_name: str = "",
    max_files: int = 200,
    max_result_chars: int = 1500,
    require_regex: str = "",
    require_citations: bool = False,
    citation_min: int = 1,
) -> str:
    """Start a map/reduce sweep over a glob in the BACKGROUND and return a job_id immediately.

    The blocking spawn_map_reduce is the right call for a handful of files; a few hundred takes
    long enough that you want it off your critical path. Same semantics otherwise - the harness
    expands the glob itself, so the file list never enters your context. Collect with check_job
    or wait_for_jobs.

    require_citations (with citation_min) applies to each per-file answer exactly as in
    spawn_map_reduce: the answer must cite that many real workspace files as path:line, verified
    against the filesystem rather than taken on trust."""
    if not (glob_pattern or "").strip() or not (map_prompt or "").strip():
        return core.to_json({"ok": False, "error": "glob_pattern and map_prompt are required"})
    try:
        job_id = _start("map_reduce", f"map_reduce {glob_pattern}", model_name, {
            "glob_pattern": glob_pattern,
            "map_prompt": map_prompt,
            "reduce_prompt": reduce_prompt,
            "model_name": model_name,
            "max_files": max_files,
            "max_result_chars": max_result_chars,
            "require_regex": require_regex,
            "require_citations": require_citations,
            "citation_min": citation_min,
        })
        return core.to_json({
            "ok": True, "job_id": job_id, "status": "running",
            "next": f"call check_job with job_id={job_id}",
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def check_job(job_id: str, full: bool = False) -> str:
    """Check a background delegation started by spawn_subagent_async or
    spawn_subagents_parallel_async.

    While running, returns just {status, started_at} - a cheap poll that costs almost nothing.
    Once finished, returns the result, bounded to a head unless full=true; the complete record
    is always on disk at job_path, so nothing is lost by polling cheaply.

    status is this JOB's lifecycle (running/done/error). The DELEGATION's own completeness verdict
    is delegation_status (complete/partial/failed) with its coverage dict, hoisted out of the
    result so it stays visible even when the result itself is replaced by a head stub. Read a
    clean delegation_status as no tool having reported a cap, not as proof of exhaustiveness; a
    partial one means coverage.signals names what was capped, truncated or skipped."""
    job_id = (job_id or "").strip()
    if not job_id:
        return core.to_json({"ok": False, "error": "job_id is required"})
    record = _read(job_id)
    if record is None:
        return core.to_json({"ok": False, "error": f"No such job: {job_id}. Use list_jobs."})

    status = record.get("status")
    if status == "running":
        return core.to_json({
            "ok": True,
            "job_id": job_id,
            "status": "running",
            "started_at": record.get("started_at"),
            "label": record.get("label"),
            "progress": _progress_tail(job_id),
            "progress_log": str(_progress_path(job_id)),
        })

    out: dict[str, Any] = {
        "ok": bool(record.get("ok", False)),
        "job_id": job_id,
        "status": status,
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "job_path": str(_job_path(job_id)),
        "progress": _progress_tail(job_id),
        "delegation_status": record.get("delegation_status"),
        "coverage": record.get("coverage"),
    }
    if record.get("error"):
        out["error"] = record["error"]
    result = record.get("result")
    if result is not None:
        if full:
            out["result"] = result
        else:
            body = json.dumps(result, ensure_ascii=False)
            out["result"] = result if len(body) <= _HEAD_CHARS else {
                "head": body[:_HEAD_CHARS],
                "truncated": True,
                "total_chars": len(body),
                "note": "Bounded view. Re-call with full=true, or read job_path, for everything.",
            }
    return core.to_json(out)


@tool
def wait_for_jobs(
    job_ids_json: str = "",
    timeout_seconds: int = 900,
    summarize_model: str = "",
    summarize_prompt: str = "",
) -> str:
    """Block until background jobs finish, then return ONE consolidated answer.

    This is the cheap way to run a fleet: polling check_job N times costs the caller N tool
    round-trips, while this costs exactly one no matter how many jobs or how long they take.
    The wait happens inside the harness (which serves other calls meanwhile), so nothing here
    is spent on the caller's side while it blocks.

    job_ids_json is a JSON list of job ids; omit it to wait for every job still running in this
    workspace. Returns per-job status plus a bounded head of each result.

    Set summarize_model to hand the finished batch to one extra cheap sub-agent that reads all
    the results and returns a single synthesis (with summarize_prompt saying what to synthesize
    for) - so a 20-job fan-out comes back as one answer instead of 20. On timeout it returns
    what has finished so far and flags the jobs still running; nothing is cancelled.

    Each job in the index carries delegation_status (complete/partial/failed) - the harness's own
    completeness verdict for that delegation, kept even when a synthesis replaces the heads,
    because the synthesis is a model's prose and cannot be trusted to mention a cap. Call
    check_job on any job that reports partial to see which tool capped."""
    import time

    try:
        deadline = time.time() + max(5, int(timeout_seconds or 900))
        wanted = core.parse_json_arg(job_ids_json, []) if (job_ids_json or "").strip() else []
        if not isinstance(wanted, list):
            return core.to_json({"ok": False, "error": "job_ids_json must be a JSON list of job ids"})
        ids = [str(j).strip() for j in wanted if str(j).strip()]

        if not ids:
            ids = [
                str(json.loads(p.read_text(encoding="utf-8", errors="replace")).get("job_id"))
                for p in _jobs_dir().glob("*.json")
                if _safe_status(p) == "running"
            ]
            ids = [j for j in ids if j and j != "None"]
        if not ids:
            return core.to_json({"ok": True, "waited_for": 0, "note": "No running jobs to wait for."})

        pending = set(ids)
        while pending and time.time() < deadline:
            for job_id in list(pending):
                record = _read(job_id)
                if record is None or record.get("status") != "running":
                    pending.discard(job_id)
            if pending:
                time.sleep(2)

        finished, still_running = [], []
        for job_id in ids:
            record = _read(job_id)
            if record is None:
                continue
            if record.get("status") == "running":
                still_running.append(job_id)
                continue
            body = json.dumps(record.get("result"), ensure_ascii=False) if record.get("result") is not None else ""
            finished.append({
                "job_id": job_id,
                "ok": bool(record.get("ok", False)),
                "label": core.trim_text(record.get("label", ""), 120),
                "error": record.get("error"),
                # Status only, not the full coverage dict: a 20-job fan-in must stay cheap, and
                # the signals are one check_job call away for any job that reports partial.
                "delegation_status": record.get("delegation_status"),
                "result_head": body[:_HEAD_CHARS],
                "result_chars": len(body),
                "job_path": str(_job_path(job_id)),
            })

        out: dict[str, Any] = {
            "ok": True,
            "waited_for": len(ids),
            "finished": len(finished),
            "failed": sum(1 for f in finished if not f["ok"]),
            "still_running": still_running,
            "timed_out": bool(still_running),
            "jobs": finished,
        }

        if summarize_model and finished:
            from . import subagent

            transcript = "\n\n".join(
                f"### job {f['job_id']} ({'ok' if f['ok'] else 'FAILED'}) - {f['label']}\n"
                f"{f['error'] or f['result_head']}"
                for f in finished
            )
            ask = summarize_prompt.strip() or (
                "Consolidate these parallel sub-task results into one answer. Merge duplicates, "
                "note disagreements, and call out any task that failed or returned nothing useful."
            )
            raw = subagent.spawn_subagent.invoke({
                "task": f"{ask}\n\nRESULTS TO CONSOLIDATE:\n{transcript}",
                "model_name": summarize_model,
                "max_result_chars": 6000,
            })
            try:
                out["summary"] = json.loads(raw).get("result")
            except Exception:
                out["summary"] = str(raw)[:6000]
            # The per-job heads are redundant once a synthesis exists; keep only the index.
            out["jobs"] = [
                # delegation_status stays in the index even when a synthesis replaced the heads:
                # the synthesis is a model's prose and cannot be trusted to mention a cap.
                {k: f[k] for k in ("job_id", "ok", "delegation_status", "label", "job_path", "result_chars")}
                for f in finished
            ]

        return core.to_json(out)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


def _safe_status(path: Path) -> str:
    try:
        return str(json.loads(path.read_text(encoding="utf-8", errors="replace")).get("status", ""))
    except Exception:
        return ""


@tool
def list_jobs(limit: int = 20, running_only: bool = False) -> str:
    """List background delegation jobs in this workspace, newest first: id, status, label,
    timestamps. Use it to find a job_id you lost, or to see what is still running."""
    try:
        records = []
        for path in _jobs_dir().glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                continue
            if running_only and data.get("status") != "running":
                continue
            records.append({
                "job_id": data.get("job_id"),
                "status": data.get("status"),
                "label": core.trim_text(data.get("label", ""), 120),
                "model": data.get("model"),
                "started_at": data.get("started_at"),
                "finished_at": data.get("finished_at"),
            })
        records.sort(key=lambda r: str(r.get("started_at") or ""), reverse=True)
        return core.to_json({
            "ok": True,
            "count": len(records),
            "running": sum(1 for r in records if r.get("status") == "running"),
            "jobs": records[: max(1, int(limit or 20))],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
