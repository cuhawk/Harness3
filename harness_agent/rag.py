"""Generic local RAG (retrieval-augmented search) tools.

Ported verbatim (behaviorally) from the monolithic main.py "# Generic local RAG
tools" section. Workspace-derived paths are read through ``core.X`` at call
time (never imported by name) so a runtime workspace switch via
``core.set_workspace()`` is honored - see core.py's module docstring.

Note: rag_ingest_graphql_cache is NOT here - it moved to graphql_tools.py
because it needs graphql_cache_paths(), which now lives there.
"""

from __future__ import annotations

import base64
import bisect
import csv
import hashlib
import html
import json
import math
import mimetypes
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

from . import core

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # optional; fallback splitter is used when unavailable
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


# =============================================================================
# Generic local RAG tools
# =============================================================================

RAG_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".agent_cache", ".venv", "venv", "dist", "build", ".mypy_cache", ".pytest_cache"}
RAG_TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".log", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".csv", ".tsv",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".rb", ".swift",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd", ".sql", ".graphql", ".gql", ".xml", ".html", ".htm", ".css", ".scss",
    ".dockerfile", ".env", ".properties",
}


def rag_embedding_model_name() -> str:
    return core.RAG_EMBED_MODEL or f"local-hash-{core.RAG_EMBED_DIM}"


def rag_connect() -> sqlite3.Connection:
    core.ensure_dirs()
    conn = sqlite3.connect(str(core.RAG_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS rag_collections (
            name TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS rag_sources (
            source_id TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            source_type TEXT NOT NULL,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            chunk_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS rag_chunks (
            chunk_id TEXT PRIMARY KEY,
            collection TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_type TEXT NOT NULL,
            path TEXT NOT NULL,
            title TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            start_char INTEGER NOT NULL,
            end_char INTEGER NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(source_id) REFERENCES rag_sources(source_id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_collection ON rag_chunks(collection);
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON rag_chunks(source_id, ordinal);
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_path ON rag_chunks(path);
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_type ON rag_chunks(source_type);
        """
    )
    return conn


def rag_sha256_text(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8", errors="replace")).hexdigest()


def rag_source_id(collection: str, source_type: str, path_or_name: str) -> str:
    raw = f"{collection}\n{source_type}\n{path_or_name}"
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:32]


def rag_chunk_id(source_id: str, ordinal: int, content_hash: str) -> str:
    return hashlib.sha256(f"{source_id}:{ordinal}:{content_hash}".encode()).hexdigest()[:32]


def rag_tokenize(text_value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_./:-]{2,}", text_value.lower())


def rag_hash_embedding(text_value: str, dim: int = core.RAG_EMBED_DIM) -> list[float]:
    vec = [0.0] * max(8, int(dim))
    tokens = rag_tokenize(text_value)
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
        value = int.from_bytes(digest, "big", signed=False)
        idx = value % len(vec)
        sign = 1.0 if ((value >> 8) & 1) else -1.0
        weight = 1.0 + min(3, len(token) // 8) * 0.15
        vec[idx] += sign * weight
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def rag_embed_with_litellm(texts: list[str]) -> list[list[float]]:
    if not core.RAG_EMBED_MODEL:
        raise RuntimeError("RAG_EMBED_MODEL is not set")
    url = core.LITELLM_BASE_URL.rstrip("/") + "/embeddings"
    out: list[list[float]] = []
    batch_size = max(1, min(core.RAG_EMBED_BATCH, 128))
    headers = {"Authorization": f"Bearer {core.LITELLM_API_KEY}", "Content-Type": "application/json"}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": core.RAG_EMBED_MODEL, "input": batch}
        r = requests.post(url, headers=headers, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
        items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        if len(items) != len(batch):
            raise RuntimeError(f"Embedding endpoint returned {len(items)} vectors for {len(batch)} inputs")
        out.extend([list(map(float, item.get("embedding", []))) for item in items])
    return out


def rag_embed_texts(texts: list[str]) -> tuple[list[list[float]], str, int, str | None]:
    if not texts:
        return [], rag_embedding_model_name(), 0, None
    if core.RAG_EMBED_MODEL:
        try:
            vectors = rag_embed_with_litellm(texts)
            dim = len(vectors[0]) if vectors else 0
            return vectors, core.RAG_EMBED_MODEL, dim, None
        except Exception as exc:
            if not core.RAG_HASH_FALLBACK:
                raise
            vectors = [rag_hash_embedding(t) for t in texts]
            return vectors, rag_embedding_model_name(), len(vectors[0]) if vectors else core.RAG_EMBED_DIM, f"Embedding endpoint failed; used hash fallback: {repr(exc)}"
    vectors = [rag_hash_embedding(t) for t in texts]
    return vectors, rag_embedding_model_name(), len(vectors[0]) if vectors else core.RAG_EMBED_DIM, None


def rag_cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rag_lexical_score(query: str, text_value: str) -> float:
    q_tokens = set(rag_tokenize(query))
    if not q_tokens:
        return 0.0
    t_tokens = set(rag_tokenize(text_value))
    if not t_tokens:
        return 0.0
    overlap = len(q_tokens & t_tokens)
    return overlap / max(1, len(q_tokens))


def rag_line_starts(text_value: str) -> list[int]:
    starts = [0]
    for match in re.finditer("\n", text_value):
        starts.append(match.end())
    return starts


def rag_line_for_offset(starts: list[int], offset: int) -> int:
    return max(1, bisect.bisect_right(starts, max(0, offset)))


def rag_fallback_split(text_value: str, chunk_chars: int, overlap: int) -> list[str]:
    text_value = text_value or ""
    if len(text_value) <= chunk_chars:
        return [text_value] if text_value.strip() else []
    chunks: list[str] = []
    pos = 0
    step = max(1, chunk_chars - overlap)
    while pos < len(text_value):
        end = min(len(text_value), pos + chunk_chars)
        window = text_value[pos:end]
        if end < len(text_value):
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "), window.rfind("; "))
            if cut > chunk_chars * 0.45:
                end = pos + cut + 1
                window = text_value[pos:end]
        if window.strip():
            chunks.append(window.strip())
        if end >= len(text_value):
            break
        pos = max(end - overlap, pos + step)
    return chunks


def rag_split_text(text_value: str, source_path: str = "", chunk_chars: int = core.RAG_CHUNK_CHARS, overlap: int = core.RAG_CHUNK_OVERLAP) -> list[dict[str, Any]]:
    chunk_chars = max(200, min(int(chunk_chars), 20000))
    overlap = max(0, min(int(overlap), chunk_chars // 2))
    starts = rag_line_starts(text_value)
    raw_chunks: list[str]
    if RecursiveCharacterTextSplitter is not None:
        ext = Path(source_path).suffix.lower()
        separators = ["\n\n", "\n", " ", ""]
        if ext in {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".cs", ".php", ".rb"}:
            separators = ["\nclass ", "\ndef ", "\nfunction ", "\nasync function ", "\nexport ", "\n\n", "\n", " ", ""]
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_chars, chunk_overlap=overlap, separators=separators)
        raw_chunks = [c for c in splitter.split_text(text_value) if c.strip()]
    else:
        raw_chunks = rag_fallback_split(text_value, chunk_chars, overlap)
    chunks: list[dict[str, Any]] = []
    cursor = 0
    for i, chunk in enumerate(raw_chunks):
        found = text_value.find(chunk, cursor)
        if found < 0:
            found = text_value.find(chunk)
        if found < 0:
            found = cursor
        end = min(len(text_value), found + len(chunk))
        chunks.append({
            "ordinal": i,
            "text": chunk,
            "start_char": found,
            "end_char": end,
            "start_line": rag_line_for_offset(starts, found),
            "end_line": rag_line_for_offset(starts, max(found, end - 1)),
        })
        cursor = end
    return chunks


class RagHTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.links: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_l = tag.lower()
        if tag_l in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1
        if tag_l == "a":
            attrs_d = {k.lower(): v for k, v in attrs if k}
            href = attrs_d.get("href")
            if href:
                self.links.append(href)
        if tag_l in {"p", "div", "br", "li", "tr", "section", "article", "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_l = tag.lower()
        if tag_l in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1
        if tag_l in {"p", "div", "li", "tr", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        cleaned = html.unescape(data).strip()
        if cleaned:
            self.parts.append(cleaned + " ")

    def get_text(self) -> str:
        text_value = "".join(self.parts)
        text_value = re.sub(r"[ \t]+", " ", text_value)
        text_value = re.sub(r"\n\s*\n\s*\n+", "\n\n", text_value)
        if self.links:
            text_value += "\n\nLINKS:\n" + "\n".join(self.links[:1000])
        return text_value.strip()


def rag_json_to_text(obj: Any, prefix: str = "", max_items: int = 20000) -> str:
    lines: list[str] = []

    def walk(value: Any, key_path: str) -> None:
        if len(lines) >= max_items:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                child = f"{key_path}.{k}" if key_path else str(k)
                walk(v, child)
        elif isinstance(value, list):
            for i, v in enumerate(value[:1000]):
                child = f"{key_path}[{i}]" if key_path else f"[{i}]"
                walk(v, child)
            if len(value) > 1000:
                lines.append(f"{key_path}: ... {len(value) - 1000} more items")
        else:
            scalar = "" if value is None else str(value)
            if scalar.strip():
                lines.append(f"{key_path}: {scalar}" if key_path else scalar)

    walk(obj, prefix)
    return "\n".join(lines)


def rag_decode_bytes(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", errors="replace")


def rag_extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except Exception as exc:
            raise RuntimeError("PDF extraction requires pypdf or PyPDF2") from exc
    reader = PdfReader(str(path))
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            page_text = f"[page {i} extraction failed: {repr(exc)}]"
        if page_text.strip():
            parts.append(f"\n\n--- page {i} ---\n{page_text}")
    return "".join(parts).strip()


def rag_extract_docx(path: Path) -> str:
    try:
        import docx  # type: ignore
    except Exception as exc:
        raise RuntimeError("DOCX extraction requires python-docx") from exc
    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def rag_extract_csv(path: Path, delimiter: str | None = None) -> str:
    text_value = path.read_text(encoding="utf-8", errors="replace")
    sample = text_value[:4096]
    dialect = csv.excel
    if delimiter:
        dialect.delimiter = delimiter
    else:
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            pass
    rows = []
    reader = csv.reader(text_value.splitlines(), dialect)
    for i, row in enumerate(reader):
        rows.append(" | ".join(row))
        if i >= 10000:
            rows.append("... [csv truncated after 10000 rows for extraction] ...")
            break
    return "\n".join(rows)


def rag_extract_file_text(path: Path, include_binary: bool = False) -> tuple[str, dict[str, Any]]:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    size = path.stat().st_size
    if size > core.MAX_RAG_FILE_BYTES:
        raise ValueError(f"File too large for RAG ingest: {size} bytes > MAX_RAG_FILE_BYTES={core.MAX_RAG_FILE_BYTES}")
    ext = path.suffix.lower()
    mime, _ = mimetypes.guess_type(str(path))
    metadata: dict[str, Any] = {"extension": ext, "mime": mime or "", "size_bytes": size}
    if ext == ".pdf":
        return rag_extract_pdf(path), {**metadata, "extractor": "pdf"}
    if ext == ".docx":
        return rag_extract_docx(path), {**metadata, "extractor": "docx"}
    if ext in {".csv", ".tsv"}:
        return rag_extract_csv(path, "\t" if ext == ".tsv" else None), {**metadata, "extractor": "csv"}
    data = path.read_bytes()
    if not include_binary and b"\x00" in data[:8192]:
        raise ValueError("Looks like a binary file; pass include_binary=true to base64-index a small binary marker")
    text_value = rag_decode_bytes(data)
    if ext in {".html", ".htm", ".xml"} or (mime or "").lower().startswith("text/html"):
        parser = RagHTMLTextExtractor()
        parser.feed(text_value)
        return parser.get_text(), {**metadata, "extractor": "html"}
    if ext in {".json", ".jsonl"}:
        try:
            if ext == ".jsonl":
                lines = []
                for line in text_value.splitlines()[:20000]:
                    if line.strip():
                        lines.append(rag_json_to_text(json.loads(line)))
                return "\n".join(lines), {**metadata, "extractor": "jsonl"}
            return rag_json_to_text(json.loads(text_value)), {**metadata, "extractor": "json"}
        except Exception:
            return text_value, {**metadata, "extractor": "text-json-fallback"}
    if include_binary and ext not in RAG_TEXT_EXTENSIONS and not (mime or "").startswith("text/"):
        encoded = base64.b64encode(data[:65536]).decode("ascii")
        return f"BINARY_FILE name={path.name} size={size} sha256={hashlib.sha256(data).hexdigest()} first_65536_base64={encoded}", {**metadata, "extractor": "binary-marker"}
    return text_value, {**metadata, "extractor": "text"}


def rag_normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[str(k)] = v
        else:
            clean[str(k)] = json.dumps(v, ensure_ascii=False)[:4000]
    return clean


def rag_store_document(collection: str, source_type: str, path_or_name: str, title: str, text_value: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    collection = core.safe_name(collection)
    source_type = core.safe_name(source_type)
    path_or_name = path_or_name or title or source_type
    title = title or Path(path_or_name).name or path_or_name
    text_value = text_value or ""
    if not text_value.strip():
        return {"ok": False, "error": "No text extracted", "path": path_or_name, "source_type": source_type}
    metadata_clean = rag_normalize_metadata(metadata)
    content_hash = rag_sha256_text(text_value)
    source_id = rag_source_id(collection, source_type, path_or_name)
    chunks = rag_split_text(text_value, path_or_name)
    if not chunks:
        return {"ok": False, "error": "No chunks produced", "path": path_or_name, "source_type": source_type}
    vectors, embed_model, dim, warning = rag_embed_texts([c["text"] for c in chunks])
    now = core.now_str()
    conn = rag_connect()
    try:
        with conn:
            conn.execute("DELETE FROM rag_chunks WHERE source_id = ?", (source_id,))
            conn.execute("DELETE FROM rag_sources WHERE source_id = ?", (source_id,))
            conn.execute(
                "INSERT OR REPLACE INTO rag_collections(name, created_at, updated_at, embedding_model, dimension, chunk_count) VALUES(?, COALESCE((SELECT created_at FROM rag_collections WHERE name=?), ?), ?, ?, ?, COALESCE((SELECT chunk_count FROM rag_collections WHERE name=?), 0))",
                (collection, collection, now, now, embed_model, dim, collection),
            )
            conn.execute(
                "INSERT INTO rag_sources(source_id, collection, source_type, path, title, content_hash, metadata_json, created_at, updated_at, chunk_count) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (source_id, collection, source_type, path_or_name, title, content_hash, json.dumps(metadata_clean, ensure_ascii=False), now, now, len(chunks)),
            )
            for chunk, vector in zip(chunks, vectors):
                chash = rag_sha256_text(chunk["text"])
                cid = rag_chunk_id(source_id, int(chunk["ordinal"]), chash)
                conn.execute(
                    """
                    INSERT INTO rag_chunks(chunk_id, collection, source_id, source_type, path, title, ordinal, start_char, end_char, start_line, end_line, text, metadata_json, embedding_json, embedding_model, dimension, content_hash, created_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cid, collection, source_id, source_type, path_or_name, title, int(chunk["ordinal"]), int(chunk["start_char"]), int(chunk["end_char"]),
                        int(chunk["start_line"]), int(chunk["end_line"]), chunk["text"], json.dumps(metadata_clean, ensure_ascii=False),
                        json.dumps(vector), embed_model, dim, chash, now,
                    ),
                )
            total = conn.execute("SELECT COUNT(*) AS n FROM rag_chunks WHERE collection = ?", (collection,)).fetchone()["n"]
            conn.execute("UPDATE rag_collections SET updated_at=?, embedding_model=?, dimension=?, chunk_count=? WHERE name=?", (now, embed_model, dim, int(total), collection))
    finally:
        conn.close()
    return {
        "ok": True,
        "collection": collection,
        "source_id": source_id,
        "source_type": source_type,
        "path": path_or_name,
        "title": title,
        "content_hash": content_hash,
        "chunks": len(chunks),
        "embedding_model": embed_model,
        "dimension": dim,
        "warning": warning,
    }


def rag_candidate_files(root: Path, recursive: bool, glob_pattern: str, max_files: int) -> list[Path]:
    files: list[Path] = []
    pattern = (glob_pattern or "**/*").lstrip("/\\")
    iterator = root.rglob(pattern.replace("**/", "")) if recursive and "**" not in pattern else root.glob(pattern)
    for p in iterator:
        try:
            rp = p.resolve(strict=False)
            rp.relative_to(core.WORKSPACE)
        except Exception:
            continue
        if any(part in RAG_SKIP_DIRS for part in rp.parts):
            continue
        if rp.is_file():
            files.append(rp)
            if len(files) >= max_files:
                break
    return files


def rag_metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    for k, v in filters.items():
        actual = metadata.get(k)
        if isinstance(v, list):
            if actual not in v and str(actual) not in [str(x) for x in v]:
                return False
        elif str(actual) != str(v):
            return False
    return True


@tool
def rag_ingest_text(name: str, text: str, collection: str = "default", source_type: str = "text", metadata_json: str = "") -> str:
    """Ingest arbitrary large text into local RAG. metadata_json must be a JSON object string."""
    try:
        metadata = core.parse_json_arg(metadata_json, {})
        result = rag_store_document(collection=collection, source_type=source_type, path_or_name=name, title=name, text_value=text, metadata=metadata)
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_ingest_path(path: str, collection: str = "default", recursive: bool = True, glob_pattern: str = "**/*", include_binary: bool = False, max_files: int = 500) -> str:
    """Ingest a workspace file or directory into local RAG. Supports text/code/json/html/csv/pdf/docx when extractors are installed."""
    try:
        target = core.resolve_path(path)
        if not target.exists():
            return core.to_json({"ok": False, "error": "Path does not exist", "path": str(target)})
        max_files = max(1, min(int(max_files), 10000))
        targets = [target] if target.is_file() else rag_candidate_files(target, bool(recursive), glob_pattern, max_files)
        results = []
        errors = []
        for fp in targets:
            try:
                text_value, metadata = rag_extract_file_text(fp, include_binary=include_binary)
                rel = core.rel_path(fp)
                result = rag_store_document(collection=collection, source_type="file", path_or_name=rel, title=fp.name, text_value=text_value, metadata=metadata)
                results.append(result)
            except Exception as exc:
                errors.append({"path": core.rel_path(fp) if fp.exists() else str(fp), "error": repr(exc)})
        ok_count = sum(1 for r in results if r.get("ok"))
        chunk_count = sum(int(r.get("chunks", 0)) for r in results if r.get("ok"))
        return core.to_json({"ok": ok_count > 0, "collection": core.safe_name(collection), "requested_path": path, "files_seen": len(targets), "files_ingested": ok_count, "chunks": chunk_count, "results": results[:50], "errors": errors[:100], "truncated_results": len(results) > 50})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_ingest_browser_snapshot(name: str, browser_json: str, collection: str = "browser") -> str:
    """Ingest browser/Playwright output into RAG. Pass JSON with url,title,text,html,links,forms,network,console,accessibility, or any similar fields."""
    try:
        data = core.parse_json_arg(browser_json, {})
        if not isinstance(data, dict):
            data = {"value": data}
        url = str(data.get("url") or data.get("page_url") or data.get("href") or "")
        title = str(data.get("title") or name)
        parts: list[str] = []
        for key in ("title", "url", "visible_text", "text", "inner_text", "accessibility", "aria", "summary"):
            if data.get(key):
                parts.append(f"{key.upper()}:\n{data.get(key)}")
        if data.get("html") or data.get("dom"):
            parser = RagHTMLTextExtractor()
            parser.feed(str(data.get("html") or data.get("dom") or ""))
            visible = parser.get_text()
            if visible:
                parts.append("DOM_VISIBLE_TEXT:\n" + visible)
        for key in ("links", "forms", "inputs", "buttons", "network", "requests", "responses", "console", "errors"):
            if key in data:
                parts.append(f"{key.upper()}:\n{rag_json_to_text(data.get(key), key)}")
        if not parts:
            parts.append(rag_json_to_text(data))
        metadata = {"url": url, "browser_snapshot_name": name, "keys": list(data.keys())[:200]}
        result = rag_store_document(collection=collection, source_type="browser", path_or_name=name, title=title, text_value="\n\n".join(parts), metadata=metadata)
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_ingest_terminal_output(name: str, command: str, stdout: str = "", stderr: str = "", exit_code: int = 0, collection: str = "terminal") -> str:
    """Ingest long terminal output into RAG so future prompts can search it instead of carrying huge stdout/stderr in context."""
    try:
        text_value = f"COMMAND:\n{command}\n\nEXIT_CODE:\n{exit_code}\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
        metadata = {"command": command, "exit_code": int(exit_code)}
        result = rag_store_document(collection=collection, source_type="terminal", path_or_name=name, title=name, text_value=text_value, metadata=metadata)
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_search(query: str, collection: str = "default", top_k: int = core.RAG_TOP_K, source_type: str = "", path_contains: str = "", metadata_filter_json: str = "", include_text: bool = True) -> str:
    """Search local RAG chunks semantically and lexically. Use for huge files, browser output, logs, schemas, and codebases."""
    try:
        collection = core.safe_name(collection)
        top_k = max(1, min(int(top_k), 50))
        filters = core.parse_json_arg(metadata_filter_json, {})
        q_vectors, q_model, q_dim, warning = rag_embed_texts([query])
        q_vec = q_vectors[0] if q_vectors else []
        conn = rag_connect()
        try:
            clauses = ["collection = ?"]
            params: list[Any] = [collection]
            if source_type.strip():
                clauses.append("source_type = ?")
                params.append(core.safe_name(source_type))
            if path_contains.strip():
                clauses.append("path LIKE ?")
                params.append(f"%{path_contains.strip()}%")
            sql = "SELECT * FROM rag_chunks WHERE " + " AND ".join(clauses)
            rows = conn.execute(sql, params).fetchall()
        finally:
            conn.close()
        scored = []
        for row in rows:
            metadata = json.loads(row["metadata_json"] or "{}")
            if filters and not rag_metadata_matches(metadata, filters):
                continue
            try:
                vec = json.loads(row["embedding_json"])
            except Exception:
                vec = []
            semantic = rag_cosine(q_vec, vec) if len(q_vec) == len(vec) else 0.0
            lexical = rag_lexical_score(query, row["text"])
            score = (semantic * 0.78) + (lexical * 0.22)
            scored.append((score, semantic, lexical, row, metadata))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, semantic, lexical, row, metadata in scored[:top_k]:
            item = {
                "chunk_id": row["chunk_id"],
                "score": round(float(score), 6),
                "semantic": round(float(semantic), 6),
                "lexical": round(float(lexical), 6),
                "collection": row["collection"],
                "source_id": row["source_id"],
                "source_type": row["source_type"],
                "path": row["path"],
                "title": row["title"],
                "ordinal": row["ordinal"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "metadata": metadata,
            }
            if include_text:
                item["text"] = core.trim_text(row["text"], 3000)
            results.append(item)
        return core.to_json({"ok": True, "query": query, "collection": collection, "count_searched": len(rows), "count": len(results), "embedding_model": q_model, "dimension": q_dim, "warning": warning, "results": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_read_chunk(chunk_id: str, around: int = 0) -> str:
    """Read an exact RAG chunk by chunk_id, optionally including neighboring chunks from the same source."""
    try:
        around = max(0, min(int(around), 10))
        conn = rag_connect()
        try:
            row = conn.execute("SELECT * FROM rag_chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
            if not row:
                return core.to_json({"ok": False, "error": "Chunk not found", "chunk_id": chunk_id})
            rows = conn.execute(
                "SELECT * FROM rag_chunks WHERE source_id = ? AND ordinal BETWEEN ? AND ? ORDER BY ordinal",
                (row["source_id"], int(row["ordinal"]) - around, int(row["ordinal"]) + around),
            ).fetchall()
        finally:
            conn.close()
        chunks = []
        for r in rows:
            chunks.append({
                "chunk_id": r["chunk_id"],
                "collection": r["collection"],
                "source_id": r["source_id"],
                "source_type": r["source_type"],
                "path": r["path"],
                "title": r["title"],
                "ordinal": r["ordinal"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "metadata": json.loads(r["metadata_json"] or "{}"),
                "text": r["text"],
            })
        return core.to_json({"ok": True, "chunk_id": chunk_id, "around": around, "count": len(chunks), "chunks": chunks})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_delete_source(source_id: str = "", path: str = "", collection: str = "default") -> str:
    """Delete one indexed RAG source and its chunks by source_id or by path within a collection."""
    try:
        collection = core.safe_name(collection)
        conn = rag_connect()
        try:
            with conn:
                if source_id.strip():
                    src = conn.execute("SELECT source_id FROM rag_sources WHERE source_id = ?", (source_id.strip(),)).fetchone()
                elif path.strip():
                    src = conn.execute("SELECT source_id FROM rag_sources WHERE collection = ? AND path = ?", (collection, path.strip())).fetchone()
                else:
                    return core.to_json({"ok": False, "error": "Provide source_id or path"})
                if not src:
                    return core.to_json({"ok": False, "error": "Source not found"})
                sid = src["source_id"]
                deleted_chunks = conn.execute("DELETE FROM rag_chunks WHERE source_id = ?", (sid,)).rowcount
                conn.execute("DELETE FROM rag_sources WHERE source_id = ?", (sid,))
                total = conn.execute("SELECT COUNT(*) AS n FROM rag_chunks WHERE collection = ?", (collection,)).fetchone()["n"]
                conn.execute("UPDATE rag_collections SET updated_at=?, chunk_count=? WHERE name=?", (core.now_str(), int(total), collection))
        finally:
            conn.close()
        return core.to_json({"ok": True, "source_id": sid, "deleted_chunks": deleted_chunks})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_list_collections() -> str:
    """List local RAG collections and chunk counts."""
    try:
        conn = rag_connect()
        try:
            collections_rows = conn.execute("SELECT * FROM rag_collections ORDER BY updated_at DESC").fetchall()
            items = []
            for row in collections_rows:
                source_count = conn.execute("SELECT COUNT(*) AS n FROM rag_sources WHERE collection = ?", (row["name"],)).fetchone()["n"]
                items.append({
                    "name": row["name"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "embedding_model": row["embedding_model"],
                    "dimension": row["dimension"],
                    "sources": int(source_count),
                    "chunks": int(row["chunk_count"]),
                })
        finally:
            conn.close()
        return core.to_json({"ok": True, "db_path": core.rel_path(core.RAG_DB_PATH), "count": len(items), "collections": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_collection_stats(collection: str = "default") -> str:
    """Show detailed stats for one local RAG collection."""
    try:
        collection = core.safe_name(collection)
        conn = rag_connect()
        try:
            source_rows = conn.execute(
                "SELECT source_id, source_type, path, title, chunk_count, updated_at, metadata_json FROM rag_sources WHERE collection = ? ORDER BY updated_at DESC LIMIT 500",
                (collection,),
            ).fetchall()
            type_rows = conn.execute("SELECT source_type, COUNT(*) AS chunks FROM rag_chunks WHERE collection = ? GROUP BY source_type ORDER BY chunks DESC", (collection,)).fetchall()
        finally:
            conn.close()
        return core.to_json({
            "ok": True,
            "collection": collection,
            "source_count": len(source_rows),
            "by_source_type": [{"source_type": r["source_type"], "chunks": int(r["chunks"])} for r in type_rows],
            "sources": [{"source_id": r["source_id"], "source_type": r["source_type"], "path": r["path"], "title": r["title"], "chunks": int(r["chunk_count"]), "updated_at": r["updated_at"], "metadata": json.loads(r["metadata_json"] or "{}")} for r in source_rows],
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
