"""GraphQL tools: introspection, schema search/lookup, and raw GraphQL requests.

Workspace-derived paths (GRAPHQL_CACHE_DIR, etc.) are read via ``core.NAME`` at call
time rather than imported directly, so a runtime workspace switch (core.set_workspace())
is picked up correctly. See core.py's module docstring for why.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from langchain_core.tools import tool

from . import core
from . import rag

GRAPHQL_INTROSPECTION_QUERY = """
query IntrospectionQuery {
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      kind
      name
      description
      fields(includeDeprecated: true) {
        name
        description
        args {
          name
          description
          type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
          defaultValue
        }
        type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
        isDeprecated
        deprecationReason
      }
      inputFields {
        name
        description
        type { kind name ofType { kind name ofType { kind name ofType { kind name } } } }
        defaultValue
      }
      interfaces { name kind }
      enumValues(includeDeprecated: true) { name description isDeprecated deprecationReason }
      possibleTypes { name kind }
    }
    directives {
      name
      description
      locations
      args { name description type { kind name ofType { kind name ofType { kind name } } } defaultValue }
    }
  }
}
""".strip()


def gql_type_to_str(t: Any) -> str:
    if not t:
        return "Unknown"
    kind = t.get("kind")
    name = t.get("name")
    of_type = t.get("ofType")
    if kind == "NON_NULL":
        return gql_type_to_str(of_type) + "!"
    if kind == "LIST":
        return "[" + gql_type_to_str(of_type) + "]"
    return name or kind or "Unknown"


def graphql_cache_paths(cache_name: str) -> tuple[Path, Path]:
    name = core.safe_name(cache_name)
    return core.GRAPHQL_CACHE_DIR / f"{name}.introspection.json", core.GRAPHQL_CACHE_DIR / f"{name}.summary.txt"


def build_graphql_summary(introspection: dict[str, Any]) -> str:
    schema = introspection.get("data", {}).get("__schema", {})
    lines: list[str] = []
    qt = (schema.get("queryType") or {}).get("name")
    mt = (schema.get("mutationType") or {}).get("name")
    st = (schema.get("subscriptionType") or {}).get("name")
    lines.append(f"ROOT query={qt} mutation={mt} subscription={st}")
    for t in schema.get("types", []):
        name = t.get("name") or ""
        if name.startswith("__"):
            continue
        kind = t.get("kind")
        desc = (t.get("description") or "").replace("\n", " ")[:300]
        lines.append(f"\n{kind} {name}: {desc}")
        fields = t.get("fields") or []
        if fields:
            for f in fields:
                args = f.get("args") or []
                arg_s = ", ".join(f"{a.get('name')}: {gql_type_to_str(a.get('type'))}" for a in args)
                dep = " DEPRECATED" if f.get("isDeprecated") else ""
                lines.append(f"  FIELD {name}.{f.get('name')}({arg_s}): {gql_type_to_str(f.get('type'))}{dep}")
        input_fields = t.get("inputFields") or []
        if input_fields:
            for f in input_fields:
                lines.append(f"  INPUT {name}.{f.get('name')}: {gql_type_to_str(f.get('type'))}")
        enum_values = t.get("enumValues") or []
        if enum_values:
            vals = ", ".join(v.get("name") or "" for v in enum_values)
            lines.append(f"  ENUM_VALUES {name}: {vals}")
    return "\n".join(lines)


@tool
def graphql_request(endpoint: str, query: str, variables_json: str = "", headers_json: str = "", operation_name: str = "", timeout_sec: int = 60) -> str:
    """Send a GraphQL POST request. variables_json and headers_json must be JSON objects as strings."""
    try:
        variables = core.parse_json_arg(variables_json, {})
        headers = core.parse_json_arg(headers_json, {})
        payload: dict[str, Any] = {"query": query, "variables": variables}
        if operation_name.strip():
            payload["operationName"] = operation_name.strip()
        request_headers = {"Content-Type": "application/json", **headers}
        r = requests.post(endpoint, headers=request_headers, json=payload, timeout=max(1, min(int(timeout_sec), 180)))
        content_type = r.headers.get("content-type", "")
        status_code = r.status_code
        try:
            body = r.json()
        except Exception:
            body = r.text
        return core.to_json({"ok": 200 <= status_code < 300, "status_code": status_code, "content_type": content_type, "body": body})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def graphql_introspect(endpoint: str, cache_name: str = "default", headers_json: str = "", timeout_sec: int = 120) -> str:
    """Run GraphQL introspection, cache raw schema and searchable summary under cache_name."""
    try:
        headers = core.parse_json_arg(headers_json, {})
        request_headers = {"Content-Type": "application/json", **headers}
        payload = {"query": GRAPHQL_INTROSPECTION_QUERY}
        r = requests.post(endpoint, headers=request_headers, json=payload, timeout=max(1, min(int(timeout_sec), 300)))
        status_code = r.status_code
        try:
            data = r.json()
        except Exception:
            return core.to_json({"ok": False, "status_code": status_code, "error": "Response was not JSON", "body": core.trim_text(r.text)})
        if not (200 <= status_code < 300) or "errors" in data:
            return core.to_json({"ok": False, "status_code": status_code, "body": data})
        raw_path, summary_path = graphql_cache_paths(cache_name)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = build_graphql_summary(data)
        summary_path.write_text(summary, encoding="utf-8")
        rag_result = None
        try:
            rag_result = rag.rag_store_document(
                collection="graphql",
                source_type="graphql",
                path_or_name=f"graphql:{core.safe_name(cache_name)}",
                title=f"GraphQL schema {core.safe_name(cache_name)}",
                text_value=summary,
                metadata={"cache_name": core.safe_name(cache_name), "raw_schema_path": core.rel_path(raw_path), "summary_path": core.rel_path(summary_path)},
            )
        except Exception as exc:
            rag_result = {"ok": False, "error": repr(exc)}
        schema = data.get("data", {}).get("__schema", {})
        return core.to_json({
            "ok": True,
            "cache_name": core.safe_name(cache_name),
            "raw_schema_path": core.rel_path(raw_path),
            "summary_path": core.rel_path(summary_path),
            "rag": rag_result,
            "query_type": (schema.get("queryType") or {}).get("name"),
            "mutation_type": (schema.get("mutationType") or {}).get("name"),
            "subscription_type": (schema.get("subscriptionType") or {}).get("name"),
            "type_count": len(schema.get("types", [])),
        })
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def graphql_schema_search(cache_name: str, query: str, max_results: int = 50) -> str:
    """Search a cached GraphQL schema summary by text. Use after graphql_introspect."""
    try:
        _, summary_path = graphql_cache_paths(cache_name)
        if not summary_path.exists():
            return core.to_json({"ok": False, "error": "Schema cache not found. Run graphql_introspect first.", "cache_name": core.safe_name(cache_name)})
        max_results = max(1, min(int(max_results), 200))
        q = query.lower()
        results = []
        for line_no, line in enumerate(summary_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            if q in line.lower():
                results.append({"line": line_no, "text": line})
                if len(results) >= max_results:
                    break
        return core.to_json({"ok": True, "cache_name": core.safe_name(cache_name), "query": query, "count": len(results), "results": results})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def graphql_get_type(cache_name: str, type_name: str) -> str:
    """Get exact details for a type from a cached GraphQL introspection schema."""
    try:
        raw_path, _ = graphql_cache_paths(cache_name)
        if not raw_path.exists():
            return core.to_json({"ok": False, "error": "Schema cache not found. Run graphql_introspect first.", "cache_name": core.safe_name(cache_name)})
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        types = data.get("data", {}).get("__schema", {}).get("types", [])
        for t in types:
            if t.get("name") == type_name:
                return core.to_json({"ok": True, "cache_name": core.safe_name(cache_name), "type": t})
        return core.to_json({"ok": False, "error": "Type not found", "type_name": type_name})
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})


@tool
def rag_ingest_graphql_cache(cache_name: str, collection: str = "graphql") -> str:
    """Ingest an already cached GraphQL schema summary into RAG. Use after graphql_introspect."""
    try:
        raw_path, summary_path = graphql_cache_paths(cache_name)
        if not summary_path.exists():
            return core.to_json({"ok": False, "error": "Schema summary cache not found. Run graphql_introspect first.", "cache_name": core.safe_name(cache_name)})
        text_value = summary_path.read_text(encoding="utf-8", errors="replace")
        metadata = {"cache_name": core.safe_name(cache_name), "raw_schema_path": core.rel_path(raw_path) if raw_path.exists() else "", "summary_path": core.rel_path(summary_path)}
        result = rag.rag_store_document(collection=collection, source_type="graphql", path_or_name=f"graphql:{core.safe_name(cache_name)}", title=f"GraphQL schema {core.safe_name(cache_name)}", text_value=text_value, metadata=metadata)
        return core.to_json(result)
    except Exception as e:
        return core.to_json({"ok": False, "error": repr(e)})
