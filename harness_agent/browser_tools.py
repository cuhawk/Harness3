"""Built-in browser automation via the Python Playwright *sync* API (in-process, no Node
`@playwright/mcp` server involved).

The sync API is thread-affine: a browser/context/page created on one OS thread raises if
touched from another. But tool calls against the "same" named session are not guaranteed to
land on the same worker thread across turns/sub-agent calls in this application - the agent
runtime is free to schedule them anywhere. The only way to satisfy both constraints is to give
each named session exactly one dedicated background thread for its entire lifetime and route
every single Playwright operation through a request/response queue onto that thread, never
touching a Playwright object from the calling tool function's own thread. That is what
_BrowserWorker below does; everything else in this module is a thin @tool wrapper that builds a
small callable, hands it to the right worker via _call(), and turns the result (or a Playwright
timeout) into the house ok/error JSON shape.
"""

from __future__ import annotations

import collections
import json
import queue
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable

from langchain_core.tools import tool
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from . import core
from . import rag

_BROWSER_TYPES = {"chromium", "firefox", "webkit"}
_CONSOLE_LOG_MAXLEN = 500

# Playwright-level per-operation timeout (ms) passed into goto/click/fill/etc.
_NAV_TIMEOUT_MS = 15_000
# Wall-clock bound on Future.result() for a single queued operation; a little looser than
# _NAV_TIMEOUT_MS to leave room for queueing/serialization overhead on the worker thread.
_OP_TIMEOUT_SEC = 20.0
# Launching a browser process is slower than a page operation, so it gets its own budget.
_OPEN_TIMEOUT_SEC = 60.0
_SHUTDOWN_TIMEOUT_SEC = 10.0

_SENTINEL = object()


class _BrowserWorker:
    """Owns exactly one named Playwright session and the one OS thread allowed to touch it.

    The worker thread opens `sync_playwright()` exactly once and keeps it open for the
    worker's whole lifetime, pulling (callable, Future) pairs off an internal queue and
    running each callable with `(playwright_object, state_dict)`. `state` holds browser/
    context/page (all None until the open callable runs) plus a bounded deque of formatted
    console log lines that the page's "console" event handler appends to - also only ever
    touched from this same thread, including when read back out (see browser_console_logs).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self.state: dict[str, Any] = {
            "browser": None,
            "context": None,
            "page": None,
            "console_logs": collections.deque(maxlen=_CONSOLE_LOG_MAXLEN),
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"browser-worker-{self.name}", daemon=True)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, func: Callable[[Any, dict[str, Any]], Any]) -> "Future[Any]":
        fut: "Future[Any]" = Future()
        self._queue.put((func, fut))
        return fut

    def shutdown(self, timeout: float = _SHUTDOWN_TIMEOUT_SEC) -> None:
        self._queue.put(_SENTINEL)
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        with sync_playwright() as p:
            while True:
                item = self._queue.get()
                if item is _SENTINEL:
                    break
                func, fut = item
                try:
                    fut.set_result(func(p, self.state))
                except Exception as exc:
                    fut.set_exception(exc)
            for key in ("context", "browser"):
                obj = self.state.get(key)
                if obj is not None:
                    try:
                        obj.close()
                    except Exception:
                        pass


# Registry mutation (create/lookup/remove entries) is guarded by this lock. Individual
# worker queues need no lock of their own: each worker queue has exactly one consumer (its
# own thread), so submit()/shutdown() from arbitrary caller threads are already safe.
_REGISTRY: dict[str, _BrowserWorker] = {}
_REGISTRY_LOCK = threading.Lock()


def _call(name: str, func: Callable[[Any, dict[str, Any]], Any], timeout: float = _OP_TIMEOUT_SEC) -> Any:
    with _REGISTRY_LOCK:
        worker = _REGISTRY.get(name)
    if worker is None or not worker.is_alive():
        raise RuntimeError(f"No open browser session named {name!r}; call browser_open first")
    return worker.submit(func).result(timeout=timeout)


def _goto_op(url: str) -> Callable[[Any, dict[str, Any]], str]:
    def op(_p: Any, state: dict[str, Any]) -> str:
        state["page"].goto(url, timeout=_NAV_TIMEOUT_MS)
        return state["page"].url
    return op


def _current_url_op(_p: Any, state: dict[str, Any]) -> str:
    return state["page"].url


def _make_open_op(browser_type: str, headless: bool, url: str) -> Callable[[Any, dict[str, Any]], str]:
    def op(p: Any, state: dict[str, Any]) -> str:
        launcher = getattr(p, browser_type)
        browser = launcher.launch(headless=headless)
        context = browser.new_context()
        page = context.new_page()
        page.on("console", lambda msg: state["console_logs"].append(f"[{core.now_str()}] [{msg.type}] {msg.text}"))
        state["browser"] = browser
        state["context"] = context
        state["page"] = page
        if url.strip():
            page.goto(url, timeout=_NAV_TIMEOUT_MS)
        return page.url
    return op


@tool
def browser_open(name: str, url: str = "", headless: bool = True, browser_type: str = "chromium") -> str:
    """Open (or reuse) a named persistent browser session. If a session with this name is
    already running, it is reused (not relaunched) and just navigated to url when one is
    given. browser_type must be one of: chromium, firefox, webkit."""
    try:
        name = core.safe_name(name)
        browser_type = str(browser_type or "chromium").strip().lower()
        if browser_type not in _BROWSER_TYPES:
            return core.to_json({
                "ok": False,
                "error": f"browser_type must be one of: {', '.join(sorted(_BROWSER_TYPES))}",
                "browser_type": browser_type,
            })
        url = str(url or "")
        headless = bool(headless)

        with _REGISTRY_LOCK:
            worker = _REGISTRY.get(name)
            reused = worker is not None and worker.is_alive()
            if not reused:
                worker = _BrowserWorker(name)
                worker.start()
                _REGISTRY[name] = worker

        if reused:
            page_url = worker.submit(_goto_op(url) if url.strip() else _current_url_op).result(timeout=_OP_TIMEOUT_SEC)
        else:
            page_url = worker.submit(_make_open_op(browser_type, headless, url)).result(timeout=_OPEN_TIMEOUT_SEC)

        return core.to_json({
            "ok": True,
            "name": name,
            "reused": reused,
            "browser_type": browser_type,
            "headless": headless,
            "url": page_url,
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_goto(name: str, url: str) -> str:
    """Navigate the named browser session's page to url."""
    try:
        name = core.safe_name(name)
        page_url = _call(name, _goto_op(url))
        return core.to_json({"ok": True, "name": name, "url": page_url})
    except PlaywrightTimeoutError as e:
        return core.to_json({"ok": False, "error": f"Timed out navigating to {url!r}: {e}"})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_click(name: str, selector: str) -> str:
    """Click the element matching selector in the named browser session's page."""
    try:
        name = core.safe_name(name)

        def op(_p: Any, state: dict[str, Any]) -> str:
            state["page"].click(selector, timeout=_NAV_TIMEOUT_MS)
            return state["page"].url

        page_url = _call(name, op)
        return core.to_json({"ok": True, "name": name, "selector": selector, "url": page_url})
    except PlaywrightTimeoutError as e:
        return core.to_json({"ok": False, "error": f"Timed out clicking {selector!r}: {e}"})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_fill(name: str, selector: str, text: str) -> str:
    """Fill the element matching selector with text in the named browser session's page."""
    try:
        name = core.safe_name(name)

        def op(_p: Any, state: dict[str, Any]) -> str:
            state["page"].fill(selector, text, timeout=_NAV_TIMEOUT_MS)
            return state["page"].url

        page_url = _call(name, op)
        return core.to_json({"ok": True, "name": name, "selector": selector, "url": page_url})
    except PlaywrightTimeoutError as e:
        return core.to_json({"ok": False, "error": f"Timed out filling {selector!r}: {e}"})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_press_key(name: str, key: str) -> str:
    """Press a keyboard key (e.g. Enter, Tab, Control+A) on the named browser session's page."""
    try:
        name = core.safe_name(name)

        def op(_p: Any, state: dict[str, Any]) -> str:
            state["page"].keyboard.press(key)
            return state["page"].url

        page_url = _call(name, op)
        return core.to_json({"ok": True, "name": name, "key": key, "url": page_url})
    except PlaywrightTimeoutError as e:
        return core.to_json({"ok": False, "error": f"Timed out pressing {key!r}: {e}"})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_get_text(name: str, selector: str = "") -> str:
    """Extract inner text: the whole page body if selector is empty, otherwise the selector's
    locator. Text larger than core.RAG_AUTO_INGEST_OUTPUT_BYTES is also auto-ingested into RAG
    (mirroring shell_git.py's auto-ingest-on-size pattern) so it stays searchable without
    forcing a second round trip; the extracted text itself is still returned (trimmed)."""
    try:
        name = core.safe_name(name)
        selector = str(selector or "").strip()

        def op(_p: Any, state: dict[str, Any]) -> str:
            page = state["page"]
            if selector:
                return page.locator(selector).inner_text(timeout=_NAV_TIMEOUT_MS)
            return page.inner_text("body", timeout=_NAV_TIMEOUT_MS)

        text_value = _call(name, op)
        text_bytes = len(text_value.encode("utf-8", errors="replace"))

        rag_auto_ingest: dict[str, Any] | None = None
        if core.RAG_AUTO_INGEST_OUTPUT and text_bytes >= core.RAG_AUTO_INGEST_OUTPUT_BYTES:
            try:
                page_url = _call(name, _current_url_op)
                ingest_name = f"browser:{name}:{core.safe_name(selector) or 'body'}:{time.strftime('%Y%m%d-%H%M%S')}"
                rag_auto_ingest = json.loads(rag.rag_ingest_browser_snapshot.invoke({
                    "name": ingest_name,
                    "browser_json": json.dumps({"url": page_url, "title": name, "text": text_value}),
                }))
            except Exception as exc:
                rag_auto_ingest = {"ok": False, "error": repr(exc)}

        return core.to_json({
            "ok": True,
            "name": name,
            "selector": selector,
            "bytes": text_bytes,
            "text": core.trim_text(text_value),
            "rag_auto_ingest": rag_auto_ingest,
        })
    except PlaywrightTimeoutError as e:
        return core.to_json({"ok": False, "error": f"Timed out extracting text: {e}"})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_snapshot(name: str) -> str:
    """Return the page's accessibility tree snapshot (role/name per node, YAML-formatted).
    Playwright 1.60 removed the old Page.accessibility.snapshot() API in favor of
    Page.aria_snapshot(), which is what this calls; core.to_json trims oversized output."""
    try:
        name = core.safe_name(name)

        def op(_p: Any, state: dict[str, Any]) -> Any:
            return state["page"].aria_snapshot()

        snapshot = _call(name, op)
        return core.to_json({"ok": True, "name": name, "snapshot": snapshot})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_screenshot(name: str, path: str = "") -> str:
    """Take a screenshot of the page. If path is empty, saves under the workspace cache
    directory with a generated timestamped filename. Returns the workspace-relative path."""
    try:
        name = core.safe_name(name)
        if str(path or "").strip():
            target = core.resolve_path(path)
        else:
            core.ensure_dirs()
            target = core.CACHE_DIR / "screenshots" / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.png"
        target.parent.mkdir(parents=True, exist_ok=True)

        def op(_p: Any, state: dict[str, Any]) -> None:
            state["page"].screenshot(path=str(target))

        _call(name, op)
        return core.to_json({"ok": True, "name": name, "path": core.rel_path(target)})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_evaluate(name: str, js: str) -> str:
    """Evaluate a JavaScript expression/function in the page context and return the result.
    Falls back to a string representation if the result is not JSON-serializable."""
    try:
        name = core.safe_name(name)

        def op(_p: Any, state: dict[str, Any]) -> Any:
            return state["page"].evaluate(js)

        result = _call(name, op)
        try:
            json.dumps(result)
        except (TypeError, ValueError):
            result = str(result)
        return core.to_json({"ok": True, "name": name, "result": result})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_console_logs(name: str, max_lines: int = 200) -> str:
    """Read recent console log lines for the named session. The deque is only ever mutated by
    the page's console handler on the session's own worker thread, so it is read the same way -
    by submitting a callable to that thread - rather than touched directly from the caller."""
    try:
        name = core.safe_name(name)
        max_lines = max(1, min(int(max_lines), 5000))

        def op(_p: Any, state: dict[str, Any]) -> list[str]:
            return list(state["console_logs"])[-max_lines:]

        lines = _call(name, op)
        return core.to_json({"ok": True, "name": name, "count": len(lines), "logs": lines})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_list() -> str:
    """List every registered named browser session and whether its worker thread is alive."""
    try:
        with _REGISTRY_LOCK:
            items = [{"name": n, "alive": w.is_alive()} for n, w in _REGISTRY.items()]
        return core.to_json({"ok": True, "count": len(items), "sessions": items})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def browser_close(name: str) -> str:
    """Shut down the named browser session's worker thread (closing its browser/context) and
    remove it from the registry."""
    try:
        name = core.safe_name(name)
        with _REGISTRY_LOCK:
            worker = _REGISTRY.pop(name, None)
        if worker is None:
            return core.to_json({"ok": False, "error": "Unknown session", "name": name})
        worker.shutdown(timeout=_SHUTDOWN_TIMEOUT_SEC)
        return core.to_json({"ok": True, "name": name, "alive": worker.is_alive()})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
