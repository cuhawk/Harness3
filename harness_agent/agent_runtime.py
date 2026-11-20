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
import os
import time
from typing import Any, Callable

import requests
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from . import completeness
from . import core
from . import hooks
from . import permissions
from . import sessions
from . import terminal_ui
from . import tools_registry
from .tenant_policy import TenantToolAccess
from . import usage as usage_module

ALL_TOOLS: list[BaseTool] = tools_registry.ALL_TOOLS
TOOLS: list[BaseTool] = ALL_TOOLS
TOOL_BY_NAME: dict[str, BaseTool] = tools_registry.TOOL_BY_NAME


def _initial_history() -> list[BaseMessage]:
    """Seed history with the system prompt, or with nothing under HARNESS_BARE_PROMPT.

    Returning [] rather than [SystemMessage(content="")] matters: an empty system
    message still renders its role header through the chat template, so it would cost
    tokens and, worse, leave a stray empty turn for the model to interpret. Built here
    rather than inline so __init__ and reset() cannot drift apart.
    """
    prompt = core.build_system_prompt()
    return [SystemMessage(content=prompt)] if prompt else []


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
    # ONE budget shared across every preview in this result, decremented as it is spent.
    # It used to be a per-field allowance: the same value was passed to _list_preview on
    # EVERY list key and then again to the content preview, so a payload with three list
    # keys plus content emitted ~4x limit_bytes. That made MODEL_TOOL_RESULT_BYTES a
    # per-field hint rather than the total cap it is documented to be. It matters most on
    # a local llama.cpp backend, where an oversized tool result is not just context spend:
    # on the moe-direct engine every additional ubatch of prompt costs one whole-model
    # expert sweep (~40 s measured), so overshooting the cap is paid in wall clock.
    budget = max(0, min(limit_bytes, max(256, limit_bytes - 2048)))

    content = payload.get("content")
    has_content = isinstance(content, str) and content != ""
    # Reserve half up front when content is present. Without this the shared budget would
    # be strictly worse than the old bug for read_file-shaped payloads: dict iteration
    # order lets arbitrary list keys consume everything and starve content_preview to "",
    # which is the one field the model usually needs. Anything the lists do not spend
    # rolls back to content below, so a payload with no lists still gets the full budget.
    content_reserve = budget // 2 if has_content else 0
    remaining = budget - content_reserve

    summary: dict[str, Any] = {}

    for key, value in payload.items():
        if key == "content":
            continue
        if isinstance(value, list):
            preview = _list_preview(value, remaining)
            remaining = max(0, remaining - _byte_len(json.dumps(preview, ensure_ascii=False)))
            summary[key] = preview
            summary[f"{key}_original_count"] = len(value)
            summary[f"{key}_preview_count"] = len(preview)
            if len(preview) < len(value):
                summary[f"{key}_truncated_for_model"] = True
            continue
        scalar = _scalar_for_summary(value)
        if scalar is not None or value is None:
            summary[key] = scalar

    if isinstance(content, str):
        preview, preview_bytes = _utf8_prefix(content, content_reserve + remaining)
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
    def __init__(
        self,
        model_name: str,
        session_id: str | None = None,
        pin_tools_to_task: bool = False,
        tool_groups: list[str] | None = None,
        *,
        tenant_access: TenantToolAccess | None = None,
    ):
        self.model_name = model_name
        self.tenant_access = tenant_access
        # Delegated runs set this: score tool selection off the ORIGINAL task plus the latest
        # turn, not the latest turn alone. See tools_registry.select_tools_for_messages.
        # Left False for interactive sessions, where following the newest message is correct.
        self.pin_tools_to_task = pin_tools_to_task
        # An explicit pin from the caller, which skips scoring altogether. Resolved once here
        # so a bad group name fails at construction - where the caller can see it - instead of
        # inside the graph loop on the first model call.
        self.tool_groups = tools_registry.resolve_pinned_groups(tool_groups)
        # Set by sessions.SessionManager.create_runtime for every runtime it owns - threaded
        # into the render_thinking/render_tool_use calls below so a background tab's turn
        # always writes into ITS OWN scrollback, not whichever tab the user happens to be
        # looking at when a call fires (see terminal_ui._history_container's session_id
        # parameter and tui_app.py's module docstring on why this matters for multi-session
        # tabs).
        self.session_id = session_id
        self.history: list[BaseMessage] = _initial_history()
        self._bound_llms: dict[tuple[str, ...], Any] = {}
        # (index of the last human message, selected tools) - select_tools_for_messages()
        # only looks at the latest human message, so re-deriving it on every graph loop
        # iteration (once per tool call in a multi-step task) is wasted work and, worse,
        # a chance for the bound tool set to drift mid-task if the derivation is ever
        # non-deterministic. Pinning it to the human-message index means it's recomputed
        # only when the user actually sends new input, not on every agent<->tool hop.
        self._tool_selection_cache: tuple[int, list[BaseTool]] | None = None
        # Every cap/truncation this runtime's tools admit to, so a delegation can report how
        # complete it actually was instead of trusting the model's prose. run_tool_calls reads
        # self.completeness at CALL time, so the recorder deliberately survives switch_model()'s
        # graph rebuild - an escalation mid-delegation must not amnesty the caps the earlier
        # attempt hit.
        self.completeness = completeness.CompletenessRecorder()
        # Token/time accounting. turn_usage is replaced wholesale at the top of every
        # invoke()/invoke_streaming() and read by the caller once the turn returns
        # (cli.handle_turn renders it); session_usage accumulates across turns for /usage.
        # Both deliberately survive switch_model()'s graph rebuild - a mid-session model
        # switch shouldn't zero what the session has already spent.
        self.turn_usage = usage_module.Usage()
        self.session_usage = usage_module.Usage()
        self.graph = self._build_graph(model_name)

    def _build_llm(self, model_name: str, tools: list[BaseTool]):
        # temperature 0 is the obvious choice for tool work and may be the wrong one. Greedy
        # decoding takes the argmax token every time, so when a rare identifier has a common
        # near-neighbour (tsg_id vs tsk_id, i.e. task id) a model that narrowly prefers the
        # common one emits it CONSISTENTLY for that context - which is the shape of the
        # silent-zero bug. Every path that gets this right (Claude Code over the same router,
        # a direct API call) sends no temperature at all. HARNESS_TEMPERATURE makes that
        # testable without editing code; unset keeps the current behaviour.
        raw = (os.environ.get("HARNESS_TEMPERATURE") or "").strip()
        try:
            temperature = float(raw) if raw else 0.0
        except ValueError:
            temperature = 0.0
        llm = ChatOpenAI(
            model=model_name,
            base_url=core.LITELLM_BASE_URL,
            api_key=core.LITELLM_API_KEY,
            temperature=temperature,
            timeout=core.LLM_TIMEOUT,
            max_retries=core.LLM_MAX_RETRIES,
            # Every model call in this class goes through .stream() (see call_model), and a
            # streamed OpenAI-compatible response carries no usage block unless the request
            # asks for one. Without this the per-turn token footer has nothing to report but
            # a chars/4 estimate. See core.STREAM_USAGE for why it stays overridable.
            stream_usage=core.STREAM_USAGE,
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
            if self.tenant_access is not None:
                selected_tools = list(self.tenant_access.tools.values())
            elif self._tool_selection_cache is not None and self._tool_selection_cache[0] == last_human_idx:
                selected_tools = self._tool_selection_cache[1]
            else:
                selected_tools = tools_registry.select_tools_for_messages(
                    messages, pin_to_first=self.pin_tools_to_task, groups=self.tool_groups
                )
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
            # Timed here rather than around the whole turn: a turn that calls tools spends
            # most of its wall clock inside them, so output_tokens/wall_time would report a
            # tok/s far below the model's real decode speed. first_chunk_at is the first
            # chunk carrying actual output (a leading role-only chunk is not a token), which
            # splits prompt-processing latency out of the decode window - see usage.py.
            #
            # ttft ENDS AT THE FIRST CHUNK OF ANY KIND, deliberately - do not re-add a content
            # test here. On a reasoning model the thinking phase is generated token-by-token
            # and billed inside completion_tokens, but langchain-openai 1.3.3 DROPS the
            # server's reasoning_content deltas: those chunks arrive with content='',
            # additional_kwargs={} and no content_blocks, so reasoning is undetectable from
            # this side (which also means _extract_reasoning below returns "" and SHOW_THINKING
            # renders nothing for such a model). Gating on content therefore charged the whole
            # thinking phase to ttft while its tokens stayed in the numerator, making tok/s
            # (reasoning + visible) / time(visible) - unbounded as the answer shortens. Measured
            # against Qwen3.8 on llama.cpp: first chunk 0.373s, first content 1.530s, end 2.581s
            # -> 152.3 tok/s reported against a true 72.5, and a "1+1" turn reported 6925.0.
            # The first chunk is llama.cpp's role-only frame, emitted when generation starts,
            # i.e. exactly the end of prefill. Caveat: a backend that flushes that frame on
            # request receipt rather than at generation start would understate ttft here.
            started = time.perf_counter()
            first_chunk_at: float | None = None
            first_content_at: float | None = None
            for chunk in llm_with_tools.stream(state["messages"]):
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                if first_content_at is None and (
                    getattr(chunk, "content", None) or getattr(chunk, "tool_call_chunks", None)
                ):
                    first_content_at = time.perf_counter()
                response = chunk if response is None else response + chunk
            self.turn_usage.record_call(
                response,
                started=started,
                first_chunk_at=first_chunk_at,
                first_content_at=first_content_at,
                ended=time.perf_counter(),
            )
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

        builder = StateGraph(MessagesState)
        builder.add_node("agent", call_model)
        builder.add_node("tools", self.run_tool_calls)
        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        builder.add_edge("tools", "agent")
        return builder.compile()

    def run_tool_calls(self, state: MessagesState):
        """The graph's tool node. A METHOD rather than a closure inside _build_graph so a test
        can drive one real tool call through it - this is the only production site that feeds the
        completeness recorder, and while it lived inside the closure the whole feature could be
        deleted with the suite still green (compute_status returns a clean verdict when no signal
        was recorded, so a broken recorder looks exactly like a clean run)."""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        outputs: list[ToolMessage] = []
        for call in tool_calls:
            name = call.get("name")
            args = call.get("args") or {}
            call_id = call.get("id") or "missing-tool-call-id"
            access = self.tenant_access
            tool_obj = (TOOL_BY_NAME if access is None else access.tools).get(name)
            # Only a real tool invocation is evidence. The unknown-tool, hook-block,
            # permission-block and exception envelopes below are synthesised by this node, and
            # scanning them would let an error masquerade as a coverage gap.
            invoked = False
            if tool_obj is None:
                content = core.to_json({"ok": False, "error": f"Unknown tool: {name}"})
            elif access is not None:
                decision = access.authorize(name, args)
                if not decision["ok"]:
                    content = core.to_json(decision)
                else:
                    try:
                        content = tool_obj.invoke(args)
                    except Exception as exc:
                        content = core.to_json({"ok": False, "tool": name, "error": type(exc).__name__})
                    else:
                        invoked = True
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
                            invoked = True
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
            if invoked:
                # BOTH halves, and they have to be recorded together: the RAW content carries the
                # cap keys the detector matches on (compaction rewrites them away), while the
                # difference between raw and model_content IS the harness's own truncation - the
                # commonest one in the system and, before this, the only one nobody reported.
                self.completeness.record(name or "unknown", content, model_content=model_content)
            outputs.append(ToolMessage(
                content=model_content,
                name=name or "unknown",
                tool_call_id=call_id,
                artifact=tool_result_artifact(content, model_content),
            ))
        return {"messages": outputs}

    def switch_model(self, model_name: str) -> None:
        self.model_name = model_name
        self._bound_llms = {}
        self.graph = self._build_graph(model_name)
        self.history.append(SystemMessage(content=f"Model switched to {model_name}. Continue with the same workspace and tools."))

    def reset(self) -> None:
        self.history = _initial_history()

    def serialize_history(self) -> list[dict[str, Any]]:
        return sessions.serialize_history(self.history)

    def load_serialized_history(self, data: list[dict[str, Any]]) -> None:
        self.history = sessions.deserialize_history(data)

    def _start_turn(self) -> float:
        """Reset per-turn accounting and return the turn's start timestamp. Paired with
        _finish_turn in a try/finally so a turn that raises still reports what it burned
        before failing, instead of leaving stale numbers from the previous turn."""
        self.turn_usage = usage_module.Usage(turns=1)
        return time.perf_counter()

    def _finish_turn(self, started: float) -> None:
        self.turn_usage.wall_seconds = max(0.0, time.perf_counter() - started)
        self.session_usage.merge(self.turn_usage)

    def invoke(self, user_text: str) -> dict[str, Any]:
        input_messages = self.history + [HumanMessage(content=user_text)]
        started = self._start_turn()
        try:
            result = self.graph.invoke(
                {"messages": input_messages},
                config={"recursion_limit": core.RECURSION_LIMIT},
            )
        finally:
            self._finish_turn(started)
        self.history = result["messages"]
        return result

    def invoke_streaming(self, user_text: str, on_token: Callable[[str], None]) -> dict[str, Any]:
        """Like invoke(), but calls on_token(text) for each incremental chunk of the
        model's natural-language output as it's generated, instead of waiting for the full
        response. Tool calls/results still render through the normal render_tool_use path
        inside run_tool_calls - only the model's own text streams token by token,
        since tool-call args arrive as one structured blob rather than readable text.

        stream_mode=["messages", "values"] tags every yielded item as (mode, payload):
        "messages" payloads are (message_chunk, metadata) pairs as tokens are generated;
        "values" payloads are the full accumulated state after each step, so the last one
        seen is equivalent to what invoke() returns.
        """
        input_messages = self.history + [HumanMessage(content=user_text)]
        final_state: dict[str, Any] | None = None
        started = self._start_turn()
        try:
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
        finally:
            self._finish_turn(started)

        if final_state is None:
            final_state = {"messages": input_messages}
        self.history = final_state["messages"]
        return final_state
