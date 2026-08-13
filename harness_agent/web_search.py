"""SearXNG web search tool.

Ported verbatim from main.py's "# SearXNG tool" section. References
``core.SEARXNG_URL`` (not a top-level import) so that switching the
workspace at runtime keeps picking up the live value.
"""

from __future__ import annotations

from typing import Any

import requests
from langchain_core.tools import tool

from . import core


# =============================================================================
# SearXNG tool
# =============================================================================

@tool
def searxng_search(query: str, max_results: int = 10, categories: str = "general", engines: str = "", language: str = "all", safesearch: int = 0, timeout_sec: int = 30) -> str:
    """Search using a locally hosted SearXNG instance. Requires JSON format enabled in SearXNG settings."""
    try:
        max_results = max(1, min(int(max_results), 50))
        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "categories": categories,
            "language": language,
            "safesearch": int(safesearch),
        }
        if engines.strip():
            params["engines"] = engines.strip()
        r = requests.get(core.SEARXNG_URL.rstrip("/") + "/search", params=params, timeout=max(1, min(int(timeout_sec), 120)))
        content_type = r.headers.get("content-type", "")
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("results", [])[:max_results]:
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": core.trim_text(item.get("content", ""), 1200),
                "engine": item.get("engine"),
                "score": item.get("score"),
            })
        return core.to_json({"ok": True, "query": query, "content_type": content_type, "count": len(results), "results": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e), "hint": "Check SEARXNG_URL and ensure search.formats includes json in settings.yml."})
