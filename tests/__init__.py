"""Test package marker plus the one piece of global test isolation.

The delegation outcome ledger writes a row for EVERY ``_run_subagent_task`` call, including the
scripted FakeRuntime runs several test modules make. With no workspace override those rows land
in the operator's real ``<workspace>/.agent_cache/delegations.sqlite3`` and pollute the very
table the feature exists to make trustworthy - a fabricated model name with a fabricated success
rate is worse than an empty ledger.

So the suite runs with the ledger's kill switch on by default. ``tests/test_ledger.py`` clears
it for its own cases (against a temporary workspace), which is the only place ledger writes are
under test. An explicit value in the environment is respected, so this cannot mask a run the
operator deliberately configured.
"""
import os

os.environ.setdefault("HARNESS_LEDGER_DISABLE", "1")
