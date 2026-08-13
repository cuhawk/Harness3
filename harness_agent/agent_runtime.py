"""LangGraph runtime with sequential tool execution, plus LiteLLM proxy model listing.

Ported from the "# LangGraph runtime with sequential tool execution" and
"# LiteLLM proxy model listing" sections of the original monolithic ``main.py``.

Differences from the original:
- The system prompt is no longer a frozen module-level string. ``core.build_system_prompt()``
  is called fresh for the initial history and again on every ``reset()``, so a workspace
  switch (``core.set_workspace()``) is reflected the next time the chat history is reset.
- ``TOOLS``/``TOOL_BY_NAME`` come from ``tools_registry`` (the flat registry assembled from
  the per-domain tool modules) instead of being defined in this file.
- The tool-chain console output goes through ``terminal_ui.render_tool_use`` (one combined,
  collapsed-by-default call+result entry in the Textual UI; rich-aware plain prints in the
  fallback) instead of a bare ``ui_print`` call.
"""
from __future__ import annotations

import json
from typing import Any, Callable

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from . import core
from . import hooks
from . import permissions
from . import sessions
from . import terminal_ui
from . import tools_registry

ALL_TOOLS: list[BaseTool] = tools_registry.ALL_TOOLS
TOOLS: list[BaseTool] = ALL_TOOLS
TOOL_BY_NAME: dict[str, BaseTool] = tools_registry.TOOL_BY_NAME


def _byte_len(value: str) -> int:
    return len(value.encode("utf-8", errors="replace"))


def _utf8_prefix(value: str, limit: int) -> tuple[str, int]:
    raw = value.encode("utf-8", errors="replace")[:max(0, int(limit))]
    text = raw.decode("utf-8", errors="replace")
    return text, _byte_len(text)


def _scalar_for_summary(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _byte_len(value) <= 1000:
            return value
        preview, preview_bytes = _utf8_prefix(value, 1000)
        return {
            "preview": preview,
            "original_bytes": _byte_len(value),
            "preview_bytes": preview_bytes,
            "truncated": True,
        }
    return None


def _compact_preview_value(value: Any) -> Any:
    scalar = _scalar_for_summary(value)
    if scalar is not None or value is None:
        return scalar
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            child_scalar = _scalar_for_summary(child)
            if child_scalar is not None or child is None:
                out[str(key)] = child_scalar
        return out
    return str(value)


def _list_preview(values: list[Any], budget_bytes: int) -> list[Any]:
    preview: list[Any] = []
    used = 2
    for item in values:
        compact_item = _compact_preview_value(item)
        item_bytes = _byte_len(json.dumps(compact_item, ensure_ascii=False))
        delimiter = 2 if preview else 0
        if preview and used + delimiter + item_bytes > budget_bytes:
            break
        preview.append(compact_item)
        used += delimiter + item_bytes
        if used >= budget_bytes:
            break
    return preview


def _compact_json_tool_result(payload: dict[str, Any], raw_bytes: int, limit_bytes: int) -> str:
    preview_budget = max(0, min(limit_bytes, max(256, limit_bytes - 2048)))
    summary: dict[str, Any] = {}

    for key, value in payload.items():
        if key == "content":
            continue
        if isinstance(value, list):
            preview = _list_preview(value, preview_budget)
            summary[key] = preview
            summary[f"{key}_original_count"] = len(value)
            summary[f"{key}_preview_count"] = len(preview)
            if len(preview) < len(value):
                summary[f"{key}_truncated_for_model"] = True
            continue
        scalar = _scalar_for_summary(value)
        if scalar is not None or value is None:
            summary[key] = scalar

    content = payload.get("content")
    if isinstance(content, str):
        preview, preview_bytes = _utf8_prefix(content, preview_budget)
        summary["content_preview"] = preview
        summary["content_original_bytes"] = _byte_len(content)
        summary["content_preview_bytes"] = preview_bytes

        path = payload.get("path")
        offset = payload.get("offset")
        file_size = payload.get("file_size_bytes")
        if isinstance(path, str) and isinstance(offset, int):
            next_offset = offset + preview_bytes
            args: dict[str, Any] = {"path": path, "offset": next_offset}
            if isinstance(file_size, int) and file_size > next_offset:
                args["max_bytes"] = min(65536, file_size - next_offset)
            summary["continue_with"] = {"tool": "read_file", "args": args}

    summary.update({
        "tool_result_compacted": True,
        "compaction_reason": "tool result exceeded model budget",
        "original_bytes": raw_bytes,
        "limit_bytes": limit_bytes,
        "full_result_saved": "ToolMessage.artifact.full_content",
    })
    return json.dumps(summary, ensure_ascii=False, indent=2)


def compact_tool_result_for_model(content: Any, limit: int | None = None) -> str:
    """Return a bounded tool result for the next LLM call.

    The terminal UI still renders the full result. This only trims the
    ToolMessage stored in conversation history, where large JSON/listing output
    gets resent on every following agent step.
    """
    text = str(content)
    max_prefix_bytes = max(0, int(core.MODEL_TOOL_RESULT_BYTES if limit is None else limit))
    raw = text.encode("utf-8", errors="replace")
    if len(raw) <= max_prefix_bytes:
        return text
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        return _compact_json_tool_result(parsed, len(raw), max_prefix_bytes)

    preview, preview_bytes = _utf8_prefix(text, max_prefix_bytes)
    return json.dumps({
        "tool_result_compacted": True,
        "compaction_reason": "tool result exceeded model budget",
        "original_bytes": len(raw),
        "limit_bytes": max_prefix_bytes,
        "text_preview": preview,
        "text_preview_bytes": preview_bytes,
        "full_result_saved": "ToolMessage.artifact.full_content",
    }, ensure_ascii=False, indent=2)


def tool_result_artifact(content: Any, model_content: str) -> dict[str, Any]:
    text = str(content)
    return {
        "full_content": text,
        "original_bytes": _byte_len(text),
        "model_content_compacted": model_content != text,
    }


def _extract_reasoning(message: Any) -> str:
    """Pull provider-supplied reasoning/thinking text off an AIMessage, if present.

    Different OpenAI-compatible backends surface this under different keys - DeepSeek-R1
    style APIs use "reasoning_content", some others use "reasoning". LangChain puts
    anything the provider returns beyond the standard fields into additional_kwargs, so
    check there rather than assuming a single provider's convention.
    """
    kwargs = getattr(message, "additional_kwargs", None) or {}
    for key in ("reasoning_content", "reasoning"):
        value = kwargs.get(key)
        if value:
            return str(value)
    return ""


# =============================================================================
# LiteLLM proxy model listing
# =============================================================================

def list_proxy_models() -> list[str]:
    url = core.LITELLM_BASE_URL.rstrip("/") + "/models"
    r = requests.get(url, headers={"Authorization": f"Bearer {core.LITELLM_API_KEY}"}, timeout=15)
    r.raise_for_status()
    payload = r.json()
    return sorted({item.get("id") for item in payload.get("data", []) if item.get("id")})


# =============================================================================
# LangGraph runtime with sequential tool execution
# =============================================================================

class AgentRuntime:
    def __init__(self, model_name: str, session_id: str | None = None):
        self.model_name = model_name
        # Set by sessions.SessionManager.create_runtime for every runtime it owns - threaded
        # into the render_thinking/render_tool_use calls below so a background tab's turn
        # always writes into ITS OWN scrollback, not whichever tab the user happens to be
        # looking at when a call fires (see terminal_ui._history_container's session_id
        # parameter and tui_app.py's module docstring on why this matters for multi-session
        # tabs).
        self.session_id = session_id
        self.history: list[BaseMessage] = [SystemMessage(content=core.build_system_prompt())]
        self._bound_llms: dict[tuple[str, ...], Any] = {}
        # (index of the last human message, selected tools) - select_tools_for_messages()
        # only looks at the latest human message, so re-deriving it on every graph loop
        # iteration (once per tool call in a multi-step task) is wasted work and, worse,
        # a chance for the bound tool set to drift mid-task if the derivation is ever
        # non-deterministic. Pinning it to the human-message index means it's recomputed
        # only when the user actually sends new input, not on every agent<->tool hop.
        self._tool_selection_cache: tuple[int, list[BaseTool]] | None = None
        self.graph = self._build_graph(model_name)

    def _build_llm(self, model_name: str, tools: list[BaseTool]):
        llm = ChatOpenAI(
            model=model_name,
            base_url=core.LITELLM_BASE_URL,
            api_key=core.LITELLM_API_KEY,
            temperature=0,
            timeout=120,
            max_retries=2,
        )
        return llm.bind_tools(tools)

    def _llm_for_tools(self, tools: list[BaseTool]):
        key = tuple(tool.name for tool in tools)
        if key not in self._bound_llms:
            self._bound_llms[key] = self._build_llm(self.model_name, tools)
        return self._bound_llms[key]

    def _build_graph(self, model_name: str):
        def call_model(state: MessagesState):
            messages = state["messages"]
            last_human_idx = -1
            for idx, message in enumerate(messages):
                if getattr(message, "type", "") == "human" or message.__class__.__name__ == "HumanMessage":
                    last_human_idx = idx
            if self._tool_selection_cache is not None and self._tool_selection_cache[0] == last_human_idx:
                selected_tools = self._tool_selection_cache[1]
            else:
                selected_tools = tools_registry.select_tools_for_messages(messages)
                self._tool_selection_cache = (last_human_idx, selected_tools)
            llm_with_tools = self._llm_for_tools(selected_tools)
            # .stream() + chunk accumulation, not .invoke(): LangGraph's stream_mode=
            # "messages" (used by invoke_streaming below) only forwards real token-by-token
            # deltas if the node itself streams from the model - a plain .invoke() call
            # here would make every graph.stream() caller receive the whole response as one
            # chunk regardless of stream_mode, since there'd be nothing incremental for
            # LangGraph to tap into. AIMessageChunk's + operator correctly merges partial
            # tool_call fragments across chunks into complete tool_calls in the final
            # accumulated message - this is standard, well-tested LangChain behavior, not
            # something this file has to reimplement. The non-streaming invoke() path above
            # calls this same node function, so it also benefits (identical final result,
            # just assembled from chunks instead of one blocking call).
            response = None
            for chunk in llm_with_tools.stream(state["messages"]):
                response = chunk if response is None else response + chunk
            if core.SHOW_THINKING:
                reasoning = _extract_reasoning(response)
                if reasoning:
                    terminal_ui.render_thinking(reasoning, session_id=self.session_id)
            return {"messages": [response]}

        def should_continue(state: MessagesState):
            last = state["messages"][-1]
            tool_calls = getattr(last, "tool_calls", None)
            if tool_calls:
                return "tools"
            return END

        def sequential_tool_node(state: MessagesState):
            last = state["messages"][-1]
            tool_calls = getattr(last, "tool_calls", None) or []
            outputs: list[ToolMessage] = []
            for call in tool_calls:
                name = call.get("name")
                args = call.get("args") or {}
                call_id = call.get("id") or "missing-tool-call-id"
                tool_obj = TOOL_BY_NAME.get(name)
                if tool_obj is None:
                    content = core.to_json({"ok": False, "error": f"Unknown tool: {name}"})
                else:
                    pre_result = hooks.run_hooks("PreToolUse", {"tool_name": name, "args": args})
                    if pre_result.get("blocked"):
                        content = core.to_json({
                            "ok": False,
                            "tool": name,
                            "error": "blocked by hook",
                            "reason": pre_result.get("reason"),
                        })
                    else:
                        allowed, reason = permissions.GATE.request(name, args)
                        if not allowed:
                            content = core.to_json({
                                "ok": False,
                                "tool": name,
                                "error": "blocked by permission policy",
                                "reason": reason,
                            })
                        else:
                            try:
                                content = tool_obj.invoke(args)
                            except Exception as exc:
                                content = core.to_json({"ok": False, "tool": name, "error": repr(exc)})
                            else:
                                try:
                                    hooks.run_hooks("PostToolUse", {
                                        "tool_name": name,
                                        "args": args,
                                        "result": core.trim_text(str(content), 4000),
                                    })
                                except Exception:
                                    pass
                if core.SHOW_THINKING:
                    # One combined call+result entry (collapsed by default in the Textual UI -
                    # see terminal_ui.render_tool_use) rather than two separate render calls,
                    # so a tool invocation reads as a single foldable block instead of two
                    # disconnected lines.
                    terminal_ui.render_tool_use(name or "unknown", args, str(content), session_id=self.session_id)
                model_content = compact_tool_result_for_model(content)
                outputs.append(ToolMessage(
                    content=model_content,
                    name=name or "unknown",
                    tool_call_id=call_id,
                    artifact=tool_result_artifact(content, model_content),
                ))
            return {"messages": outputs}

        builder = StateGraph(MessagesState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", sequential_tool_node)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        return builder.compile()

    def switch_model(self, model_name: str) -> None:
        self.model_name = model_name
        self._bound_llms = {}
        self.graph = self._build_graph(model_name)
        self.history.append(SystemMessage(content=f"Model switched to {model_name}. Continue with the same workspace and tools."))

    def reset(self) -> None:
        self.history = [SystemMessage(content=core.build_system_prompt())]

    def serialize_history(self) -> list[dict[str, Any]]:
        return sessions.serialize_history(self.history)

    def load_serialized_history(self, data: list[dict[str, Any]]) -> None:
        self.history = sessions.deserialize_history(data)

    def invoke(self, user_text: str) -> dict[str, Any]:
        input_messages = self.history + [HumanMessage(content=user_text)]
        result = self.graph.invoke(
            {"messages": input_messages},
            config={"recursion_limit": core.RECURSION_LIMIT},
        )
        self.history = result["messages"]
        return result

    def invoke_streaming(self, user_text: str, on_token: Callable[[str], None]) -> dict[str, Any]:
        """Like invoke(), but calls on_token(text) for each incremental chunk of the
        model's natural-language output as it's generated, instead of waiting for the full
        response. Tool calls/results still render through the normal render_tool_use path
        inside sequential_tool_node - only the model's own text streams token by token,
        since tool-call args arrive as one structured blob rather than readable text.

        stream_mode=["messages", "values"] tags every yielded item as (mode, payload):
        "messages" payloads are (message_chunk, metadata) pairs as tokens are generated;
        "values" payloads are the full accumulated state after each step, so the last one
        seen is equivalent to what invoke() returns.
        """
        input_messages = self.history + [HumanMessage(content=user_text)]
        final_state: dict[str, Any] | None = None
        for stream_mode, payload in self.graph.stream(
            {"messages": input_messages},
            config={"recursion_limit": core.RECURSION_LIMIT},
            stream_mode=["messages", "values"],
        ):
            if stream_mode == "messages":
                chunk, metadata = payload
                if metadata.get("langgraph_node") == "agent":
                    text = getattr(chunk, "content", None)
                    if text:
                        on_token(str(text))
            elif stream_mode == "values":
                final_state = payload

        if final_state is None:
            final_state = {"messages": input_messages}
        self.history = final_state["messages"]
        return final_state
