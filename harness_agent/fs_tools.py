"""Filesystem tools for the local agent.

Ported verbatim (same logic, same docstrings, same tool names) from the
"# Filesystem tools" section of the original monolithic main.py, plus a new
switch_workspace / current_workspace pair of tools.

Workspace-derived paths are read through ``core.NAME`` at call time (never
``from .core import NAME``) so that runtime workspace switches via
``core.set_workspace()`` are picked up immediately - see core.py's module
docstring.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from langchain_core.tools import tool

from . import core


@tool
def list_dir(path: str = ".", max_entries: int = 200) -> str:
    """List files and directories inside the workspace."""
    try:
        p = core.resolve_path(path)
        if not p.exists():
            return core.to_json({"ok": False, "error": "Path does not exist", "path": str(p)})
        if not p.is_dir():
            return core.to_json({"ok": False, "error": "Path is not a directory", "path": str(p)})
        max_entries = max(1, min(int(max_entries), 1000))
        entries = [core.path_metadata(child) for child in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))[:max_entries]]
        return core.to_json({"ok": True, "workspace": str(core.WORKSPACE), "path": core.rel_path(p), "count": len(entries), "entries": entries})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def list_tree(path: str = ".", max_entries: int = 500) -> str:
    """List a recursive tree of files/directories inside the workspace, skipping common huge folders."""
    try:
        root = core.resolve_path(path)
        if not root.exists():
            return core.to_json({"ok": False, "error": "Path does not exist"})
        if not root.is_dir():
            return core.to_json({"ok": False, "error": "Path is not a directory"})
        max_entries = max(1, min(int(max_entries), core.MAX_TREE_ENTRIES))
        results = []
        skip = {".git", "node_modules", "__pycache__", ".agent_cache", ".venv", "venv", "dist", "build"}
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in sorted(dirs) if d not in skip]
            for name in dirs + sorted(files):
                p = Path(current) / name
                results.append(core.path_metadata(p))
                if len(results) >= max_entries:
                    return core.to_json({"ok": True, "path": core.rel_path(root), "count": len(results), "truncated": True, "entries": results})
        return core.to_json({"ok": True, "path": core.rel_path(root), "count": len(results), "truncated": False, "entries": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def stat_path(path: str) -> str:
    """Get metadata for a file or directory inside the workspace."""
    try:
        p = core.resolve_path(path)
        return core.to_json({"ok": True, "metadata": core.path_metadata(p)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def read_file(path: str, offset: int = 0, max_bytes: int = 65536) -> str:
    """Read a UTF-8 text file inside the workspace."""
    try:
        p = core.resolve_path(path)
        if not p.exists():
            return core.to_json({"ok": False, "error": "File does not exist", "path": str(p)})
        if not p.is_file():
            return core.to_json({"ok": False, "error": "Path is not a file", "path": str(p)})
        offset = max(0, int(offset))
        max_bytes = max(1, min(int(max_bytes), core.MAX_READ_BYTES))
        with p.open("rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
        return core.to_json({
            "ok": True,
            "path": core.rel_path(p),
            "absolute_path": str(p),
            "offset": offset,
            "bytes_read": len(data),
            "file_size_bytes": p.stat().st_size,
            "content": data.decode("utf-8", errors="replace"),
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def write_file(path: str, content: str, mode: str = "overwrite", create_dirs: bool = True) -> str:
    """Write or append UTF-8 text to a file inside the workspace. mode must be overwrite or append."""
    try:
        p = core.resolve_path(path)
        if mode not in {"overwrite", "append"}:
            return core.to_json({"ok": False, "error": "mode must be overwrite or append"})
        raw = str(content).encode("utf-8")
        if len(raw) > core.MAX_WRITE_BYTES:
            return core.to_json({"ok": False, "error": f"Refusing to write {len(raw)} bytes; MAX_WRITE_BYTES={core.MAX_WRITE_BYTES}"})
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        if not p.parent.exists():
            return core.to_json({"ok": False, "error": "Parent directory does not exist"})
        write_mode = "w" if mode == "overwrite" else "a"
        with p.open(write_mode, encoding="utf-8", errors="replace", newline="") as f:
            f.write(str(content))
        return core.to_json({"ok": True, "path": core.rel_path(p), "absolute_path": str(p), "mode": mode, "bytes_written": len(raw)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def touch_file(path: str, create_dirs: bool = True) -> str:
    """Create a file if missing or update its timestamp inside the workspace."""
    try:
        p = core.resolve_path(path)
        if create_dirs:
            p.parent.mkdir(parents=True, exist_ok=True)
        p.touch(exist_ok=True)
        return core.to_json({"ok": True, "metadata": core.path_metadata(p)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def make_dir(path: str) -> str:
    """Create a directory inside the workspace."""
    try:
        p = core.resolve_path(path)
        p.mkdir(parents=True, exist_ok=True)
        return core.to_json({"ok": True, "path": core.rel_path(p), "absolute_path": str(p)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def copy_path(source: str, destination: str, overwrite: bool = False) -> str:
    """Copy a file or directory inside the workspace."""
    try:
        src = core.resolve_path(source)
        dst = core.resolve_path(destination)
        if not src.exists():
            return core.to_json({"ok": False, "error": "Source does not exist", "source": str(src)})
        if dst.exists() and not overwrite:
            return core.to_json({"ok": False, "error": "Destination exists and overwrite=false", "destination": str(dst)})
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists() and overwrite:
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return core.to_json({"ok": True, "source": core.rel_path(src), "destination": core.rel_path(dst), "metadata": core.path_metadata(dst)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def move_path(source: str, destination: str, overwrite: bool = False) -> str:
    """Move or rename a file/directory inside the workspace."""
    try:
        src = core.resolve_path(source)
        dst = core.resolve_path(destination)
        if not src.exists():
            return core.to_json({"ok": False, "error": "Source does not exist", "source": str(src)})
        if dst.exists() and not overwrite:
            return core.to_json({"ok": False, "error": "Destination exists and overwrite=false", "destination": str(dst)})
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        shutil.move(str(src), str(dst))
        return core.to_json({"ok": True, "source": source, "destination": core.rel_path(dst), "metadata": core.path_metadata(dst)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def delete_path(path: str, recursive: bool = False, confirm: str = "") -> str:
    """Delete a file/directory inside the workspace. Requires confirm='DELETE'. Direct workspace deletion is refused."""
    try:
        if not core.ALLOW_DELETE:
            return core.to_json({"ok": False, "error": "Deletion disabled by ALLOW_DELETE=0"})
        if confirm != "DELETE":
            return core.to_json({"ok": False, "error": "Deletion requires confirm='DELETE'"})
        p = core.resolve_path(path)
        if p == core.WORKSPACE:
            return core.to_json({"ok": False, "error": "Refusing to delete workspace root"})
        if not p.exists():
            return core.to_json({"ok": False, "error": "Path does not exist", "path": str(p)})
        if p.is_dir():
            if not recursive:
                return core.to_json({"ok": False, "error": "Directory deletion requires recursive=true"})
            shutil.rmtree(p)
        else:
            p.unlink()
        return core.to_json({"ok": True, "deleted": path})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def replace_in_file(path: str, old: str, new: str, count: int = 0) -> str:
    """Replace text in a UTF-8 file inside the workspace. count=0 means replace all."""
    try:
        p = core.resolve_path(path)
        if not p.is_file():
            return core.to_json({"ok": False, "error": "Path is not a file"})
        text = p.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old)
        if occurrences == 0:
            return core.to_json({"ok": False, "error": "old text not found", "path": core.rel_path(p)})
        new_text = text.replace(old, new, count if int(count) > 0 else -1)
        raw = new_text.encode("utf-8")
        if len(raw) > core.MAX_WRITE_BYTES:
            return core.to_json({"ok": False, "error": f"Refusing to write {len(raw)} bytes; MAX_WRITE_BYTES={core.MAX_WRITE_BYTES}"})
        p.write_text(new_text, encoding="utf-8", errors="replace")
        replaced = occurrences if int(count) <= 0 else min(occurrences, int(count))
        return core.to_json({"ok": True, "path": core.rel_path(p), "occurrences_found": occurrences, "occurrences_replaced": replaced})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def find_files(glob_pattern: str = "**/*", name_contains: str = "", max_results: int = 200) -> str:
    """Find files/directories by glob pattern and optional name substring inside the workspace.

    Returns matches_total (every match, counted independently of the cap), returned (how many
    are listed), and capped. Report matches_total as the answer to "how many" - `returned` is
    only the size of the listing."""
    try:
        max_results = max(1, min(int(max_results), 1000))
        name_contains_l = name_contains.lower().strip()
        pattern = (glob_pattern.strip() or "**/*").lstrip("/\\")
        results = []
        # Keep counting after the listing cap is reached. Stopping at the cap (the previous
        # behavior) made `count` the CAPPED number, which small models reliably report as the
        # true total - the exact failure this tool has to make impossible.
        matches_total = 0
        for p in core.WORKSPACE.glob(pattern):
            try:
                rp = p.resolve(strict=False)
                rp.relative_to(core.WORKSPACE)
            except Exception:
                continue
            if name_contains_l and name_contains_l not in p.name.lower():
                continue
            matches_total += 1
            if len(results) < max_results:
                results.append(core.path_metadata(rp))
        capped = matches_total > len(results)
        payload = {
            "ok": True,
            "glob_pattern": glob_pattern,
            "name_contains": name_contains,
            "matches_total": matches_total,
            "returned": len(results),
            "capped": capped,
            "results": results,
        }
        if capped:
            payload["note"] = (
                f"LISTING TRUNCATED: {matches_total} files matched, only {len(results)} are listed "
                f"(max_results={max_results}). The true total is matches_total."
            )
        return core.to_json(payload)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def grep_files(query: str, glob_pattern: str = "**/*", regex: bool = False, case_sensitive: bool = False, max_results: int = 100, max_file_bytes: int = 2_000_000) -> str:
    """Search text inside workspace files.

    Returns matches_total (every matching line, counted independently of the listing cap),
    returned, capped, and files_scanned/files_skipped. Report matches_total as the answer to
    "how many" - `returned` is only the size of the listing. The scan always completes, so
    matches_total is a true total even when the listing is truncated."""
    try:
        max_results = max(1, min(int(max_results), 1000))
        max_file_bytes = max(1, min(int(max_file_bytes), 20_000_000))
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(query if regex else re.escape(query), flags)
        pattern = (glob_pattern.strip() or "**/*").lstrip("/\\")
        results = []
        # The scan runs to completion instead of returning at the cap. Returning early made the
        # reported `count` the CAPPED value AND silently skipped every remaining file, so a
        # "47 matches" answer could really mean "47 shown, thousands unexamined". Files skipped
        # for being oversized are counted separately so that gap is visible too.
        matches_total = 0
        files_scanned = 0
        files_skipped_large = 0
        for p in core.WORKSPACE.glob(pattern):
            try:
                rp = p.resolve(strict=False)
                rp.relative_to(core.WORKSPACE)
            except Exception:
                continue
            if not rp.is_file():
                continue
            try:
                if rp.stat().st_size > max_file_bytes:
                    files_skipped_large += 1
                    continue
                text = rp.read_bytes().decode("utf-8", errors="replace")
            except Exception:
                continue
            files_scanned += 1
            for line_no, line in enumerate(text.splitlines(), start=1):
                if compiled.search(line):
                    matches_total += 1
                    if len(results) < max_results:
                        results.append({"path": core.rel_path(rp), "line": line_no, "text": core.trim_text(line, 1000)})
        capped = matches_total > len(results)
        payload = {
            "ok": True,
            "query": query,
            "matches_total": matches_total,
            "returned": len(results),
            "capped": capped,
            "files_scanned": files_scanned,
            "files_skipped_too_large": files_skipped_large,
            "results": results,
        }
        if capped:
            payload["note"] = (
                f"LISTING TRUNCATED: {matches_total} matching lines found, only {len(results)} are "
                f"listed (max_results={max_results}). The true total is matches_total."
            )
        return core.to_json(payload)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def file_hash(path: str, algorithm: str = "sha256") -> str:
    """Hash a file inside the workspace with sha256, sha1, or md5."""
    try:
        p = core.resolve_path(path)
        if not p.is_file():
            return core.to_json({"ok": False, "error": "Path is not a file"})
        if algorithm not in {"sha256", "sha1", "md5"}:
            return core.to_json({"ok": False, "error": "algorithm must be sha256, sha1, or md5"})
        h = hashlib.new(algorithm)
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return core.to_json({"ok": True, "path": core.rel_path(p), "algorithm": algorithm, "hash": h.hexdigest()})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def switch_workspace(path: str, create: bool = True) -> str:
    """Switch the active workspace root at runtime. All file/RAG/memory/MCP/workflow/automation
    paths move under the new workspace immediately. Returns the previous and new workspace."""
    try:
        result = core.set_workspace(path)
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def current_workspace() -> str:
    """Show the active workspace root and recent workspace switch history."""
    try:
        return core.to_json({"ok": True, "workspace": str(core.WORKSPACE), "history": core.WORKSPACE_HISTORY[-20:]})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
