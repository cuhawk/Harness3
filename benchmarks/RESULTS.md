# Wave 5 verification

Observed in Codex Cloud on Linux with Python 3.12.14 using `python -m pytest -q`
and `python -m benchmarks.run_bench`.

- Pytest passed 300 tests and 13 subtests.
- Replay recovered from all 6 injected process exits with 0 duplicate side effects.

| Measurement | Samples | p50 (ms) | p95 (ms) |
| --- | ---: | ---: | ---: |
| SQLite checkpoint overhead | 200 | 1.4042 | 2.2433 |
| HITL signing and authorization | 200 | 0.0233 | 0.0509 |
