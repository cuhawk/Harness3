# Harness Agent

Harness Agent is a local tool-calling runtime with saved workflows, delegation,
and an MCP bridge. Wave 5 adds an explicit durable execution path and tenant
authorization for host-configured tool calls.

```text
 Authenticated host ---- tenant policy <---- authenticated human
          |                  |                 signed approval
          |                  +------> JSON decision + OTel span
          v                  |
 Explicit plan ----> Durable Engine -----> Registered tool adapter
          ^                  |                     |
          |                  v                     v
          +---- resume --- SQLite            Provider deduplication
                          checkpoints        using operation key
                          retry state
                          dead letters

 Tenant-configured AgentRuntime ----> same tenant policy ----> tenant tools
```

## Durable workflows

Construct a `DurableEngine` with a persistent database path, a `TenantPolicy`,
and a mapping of trusted tool names to `Tool` adapters. Pass the host-authenticated
`Principal`, a tenant-unique run ID, and an ordered list of `Step` objects to
`run`. Pass a new definition version when changing adapter semantics.

`resume(principal, run_id)` reloads the persisted plan and continues at the
first uncommitted step. The status indicates completion, an authorization
block, a scheduled retry, or a dead letter. `queue(principal)` lists ready runs.
Use `dead_letters=True` to list terminal failures. Tool exceptions consume the
step's retry budget, with exponential delays capped by configuration. A process
exit rolls back the current transaction so a replacement worker can retry it.

For local effects, use the SQLite transaction supplied to the adapter and let
the engine commit it with the checkpoint. For remote effects, the provider
must record the supplied idempotency key atomically with the effect and return
the original result when the key is repeated. This covers the window between
a provider commit and the local checkpoint. SQLite serializes writes across
workers, so keep adapter calls bounded. Saved prompt workflows continue to run
through the existing `workflow_run` tool. Durable plans use the explicit engine
API and registered adapters.

## Tenant policy and approvals

The host supplies subject-to-role memberships and role-to-tool grants keyed by
tenant. The host also controls the signing key and tool registrations. In
tenant mode, pass `TenantToolAccess` to `AgentRuntime` with tool instances scoped
to that tenant's resources. Keep unrestricted delegation and generic tool
proxies outside the tenant tool mapping.

When a durable step returns `awaiting_approval`, its authorization object
identifies the requester, operation, tool, and argument hash. Show the original
arguments to an authenticated reviewer in the same tenant, sign the matching
`ToolCall` using `TenantPolicy.sign`, and pass the approval to `resume` keyed
by step ID. For live runtime calls, a host approval callback can confirm and
sign the same binding. The authorization event is emitted as JSON at INFO
level. Configure an OpenTelemetry SDK exporter to collect authorization spans
and correlate them with the event's trace and span IDs.

## Verification

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python -m benchmarks.run_bench
```

Observed locally in Codex Cloud on Linux with Python 3.12.14. Pytest passed
**300 tests and 13 subtests**. Replay recovered from **6 process exits with 0
duplicate side effects**. Across 200 samples per metric, checkpoint overhead
was **1.4042 ms p50 / 2.2433 ms p95** and HITL signing and authorization was
**0.0233 ms p50 / 0.0509 ms p95**.

The benchmark measures the interval from a no-op adapter to the committed
checkpoint and return. HITL timing covers signing and verification in-process,
without human response time or span exporter overhead. See
[benchmark results](benchmarks/RESULTS.md).
