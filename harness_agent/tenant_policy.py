"""Host-managed tenant grants and signed approval decisions."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping
from uuid import uuid4
from dataclasses import asdict, dataclass

from opentelemetry import trace

_LOG = logging.getLogger(__name__)
_TRACER = trace.get_tracer(__name__)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Principal:
    tenant: str
    subject: str


@dataclass(frozen=True)
class ToolCall:
    principal: Principal
    operation: str
    tool: str
    arguments_hash: str

    @classmethod
    def bind(cls, principal, operation, tool, arguments):
        return cls(principal, operation, tool, fingerprint(arguments))


@dataclass(frozen=True)
class Approval:
    approver: str
    expires_at: int
    signature: str


class TenantPolicy:
    """Resolve roles from trusted host configuration, keyed by tenant."""

    def __init__(self, *, memberships, grants, signing_key: bytes, high_impact=(), clock=time.time):
        if len(signing_key) < 32:
            raise ValueError("approval signing key must contain at least 32 bytes")
        self.memberships = {key: frozenset(roles) for key, roles in memberships.items()}
        self.grants = {key: frozenset(tools) for key, tools in grants.items()}
        self.high_impact = frozenset(high_impact)
        self._signing_key = signing_key
        self.clock = clock

    def _roles(self, principal: Principal):
        return self.memberships.get((principal.tenant, principal.subject), ())

    def _can_approve(self, call: ToolCall, subject: str) -> bool:
        return subject != call.principal.subject and "approver" in self._roles(
            Principal(call.principal.tenant, subject)
        )

    def _signature(self, call: ToolCall, subject: str, expires_at: int) -> str:
        payload = canonical(["harness-hitl-v1", asdict(call), subject, expires_at]).encode()
        return hmac.digest(self._signing_key, payload, "sha256").hex()

    def sign(self, call: ToolCall, approver: Principal, *, ttl_seconds: int = 300) -> Approval:
        """Called by the host after authenticating the human and confirming the call."""
        if approver.tenant != call.principal.tenant or not self._can_approve(call, approver.subject):
            raise PermissionError("a separate approver in the same tenant is required")
        if not 0 < ttl_seconds <= 900:
            raise ValueError("approval TTL must be between 1 and 900 seconds")
        expires_at = int(self.clock()) + ttl_seconds
        return Approval(approver.subject, expires_at, self._signature(call, approver.subject, expires_at))

    def authorize(self, call: ToolCall, approval: Approval | None = None, *, high_impact=False) -> dict:
        principal = call.principal
        attributes = {
            "harness.tenant": principal.tenant,
            "harness.subject": principal.subject,
            "harness.operation": call.operation,
            "harness.tool": call.tool,
        }
        with _TRACER.start_as_current_span("harness.tool.authorize", attributes=attributes) as span:
            granted = any(
                call.tool in self.grants.get((principal.tenant, role), ())
                for role in self._roles(principal)
            )
            decision = "allowed"
            if not granted:
                decision = "rbac_denied"
            elif high_impact or call.tool in self.high_impact:
                if approval is None:
                    decision = "approval_required"
                elif (
                    approval.expires_at <= self.clock()
                    or not self._can_approve(call, approval.approver)
                    or not hmac.compare_digest(
                        approval.signature.encode(),
                        self._signature(call, approval.approver, approval.expires_at).encode(),
                    )
                ):
                    decision = "approval_invalid"
            span.set_attribute("harness.decision", decision)
            event = {"ok": decision == "allowed", "event": "tool_authorization",
                     "decision": decision, **asdict(call)}
            context = span.get_span_context()
            if context.is_valid:
                event["trace_id"] = format(context.trace_id, "032x")
                event["span_id"] = format(context.span_id, "016x")
            _LOG.info("%s", canonical(event))
            return event


class TenantToolAccess:
    """Bind a runtime to authenticated host identity and tenant-specific tool instances."""

    def __init__(
        self, principal: Principal, policy: TenantPolicy, tools: Mapping[str, Any], *,
        approval_provider: Callable[[ToolCall, dict], Approval | None] | None = None,
        high_impact=(),
    ):
        if any(name != tool.name for name, tool in tools.items()):
            raise ValueError("tool mapping keys must match registered names")
        self.principal = principal
        self.policy = policy
        self.tools = MappingProxyType(dict(tools))
        self.approval_provider = approval_provider
        self.high_impact = frozenset(high_impact)

    def authorize(self, name: str, arguments: dict) -> dict:
        from .permissions import default_policy_for

        call = ToolCall.bind(self.principal, uuid4().hex, name, arguments)
        high_impact = name in self.high_impact or default_policy_for(name) == "ask"
        decision = self.policy.authorize(call, high_impact=high_impact)
        if decision["decision"] == "approval_required" and self.approval_provider is not None:
            # The host gets a copy for display. The signed arguments are the ones executed.
            approval = self.approval_provider(call, json.loads(canonical(arguments)))
            decision = self.policy.authorize(call, approval, high_impact=high_impact)
        return decision
