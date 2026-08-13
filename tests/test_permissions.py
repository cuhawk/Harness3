import shutil
import tempfile
import unittest
from pathlib import Path

from harness_agent import core, permissions


class PermissionEvaluateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="harness_perm_test_"))
        core.set_workspace(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_default_policy_ask_for_mutating_tool(self) -> None:
        self.assertEqual(permissions.default_policy_for("write_file"), "ask")
        self.assertEqual(permissions.default_policy_for("run_terminal"), "ask")

    def test_default_policy_allow_for_read_only_tool(self) -> None:
        self.assertEqual(permissions.default_policy_for("read_file"), "allow")
        self.assertEqual(permissions.default_policy_for("some_unknown_future_tool"), "allow")

    def test_evaluate_falls_back_to_default_with_no_persisted_rules(self) -> None:
        self.assertEqual(permissions.evaluate("write_file", {"path": "x"}), "ask")
        self.assertEqual(permissions.evaluate("read_file", {"path": "x"}), "allow")

    def test_permission_set_overrides_default(self) -> None:
        result = permissions.permission_set("write_file", "allow")
        self.assertTrue(result["ok"])
        self.assertEqual(permissions.evaluate("write_file", {"path": "x"}), "allow")

    def test_permission_set_rejects_invalid_policy(self) -> None:
        result = permissions.permission_set("write_file", "sometimes")
        self.assertFalse(result["ok"])

    def test_pattern_rule_matches_before_default(self) -> None:
        permissions.permission_add_rule("run_terminal", r"^git status", "allow")
        self.assertEqual(
            permissions.evaluate("run_terminal", {"command": "git status --short"}),
            "allow",
        )
        # A command that doesn't match the pattern still falls through to the ask default.
        self.assertEqual(
            permissions.evaluate("run_terminal", {"command": "rm -rf something"}),
            "ask",
        )

    def test_permission_remove_rule(self) -> None:
        permissions.permission_add_rule("run_terminal", r"^git status", "allow")
        removed = permissions.permission_remove_rule("run_terminal", r"^git status")
        self.assertTrue(removed["ok"])
        self.assertTrue(removed["removed"])
        self.assertEqual(
            permissions.evaluate("run_terminal", {"command": "git status --short"}),
            "ask",
        )


class PermissionGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="harness_perm_gate_test_"))
        core.set_workspace(self._tmp)
        self.gate = permissions.PermissionGate()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_allow_policy_never_calls_confirm_callback(self) -> None:
        called = []
        self.gate.set_confirm_callback(lambda name, args: called.append(1) or "deny")
        allowed, _reason = self.gate.request("read_file", {"path": "x"})
        self.assertTrue(allowed)
        self.assertEqual(called, [])

    def test_ask_policy_with_no_callback_auto_denies(self) -> None:
        allowed, reason = self.gate.request("write_file", {"path": "x"})
        self.assertFalse(allowed)
        self.assertIn("auto-denied", reason)

    def test_ask_policy_in_background_context_auto_denies_even_with_callback(self) -> None:
        self.gate.set_confirm_callback(lambda name, args: "allow")
        token = permissions._permission_context.set("background")
        try:
            allowed, _reason = self.gate.request("write_file", {"path": "x"})
        finally:
            permissions._permission_context.reset(token)
        self.assertFalse(allowed)

    def test_ask_policy_always_allow_persists_policy(self) -> None:
        self.gate.set_confirm_callback(lambda name, args: "always_allow")
        allowed, _reason = self.gate.request("write_file", {"path": "x"})
        self.assertTrue(allowed)
        self.assertEqual(permissions.evaluate("write_file", {"path": "x"}), "allow")


if __name__ == "__main__":
    unittest.main()
