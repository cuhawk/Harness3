import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from harness_agent import core, hooks

# A hook "command" that reads the JSON payload piped on stdin and unconditionally prints a
# block decision - built from sys.executable so it works the same on Windows/POSIX without
# depending on a shell-specific echo/cat builtin.
_BLOCK_COMMAND = f'"{sys.executable}" -c "import sys,json; json.load(sys.stdin); print(json.dumps({{\\"decision\\": \\"block\\", \\"reason\\": \\"blocked for test\\"}}))"'
_NOOP_COMMAND = f'"{sys.executable}" -c "import sys; sys.stdin.read()"'


class HooksMatcherTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="harness_hooks_test_"))
        core.set_workspace(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_run_hooks_with_no_hooks_never_blocks(self) -> None:
        result = hooks.run_hooks("PreToolUse", {"tool_name": "write_file", "args": {}})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["results"], [])

    def test_pretooluse_hook_matching_tool_name_blocks(self) -> None:
        hooks.hook_add("PreToolUse", "^write_file$", _BLOCK_COMMAND, timeout_sec=15)
        result = hooks.run_hooks("PreToolUse", {"tool_name": "write_file", "args": {"path": "x"}})
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "blocked for test")

    def test_pretooluse_hook_non_matching_tool_name_does_not_block(self) -> None:
        hooks.hook_add("PreToolUse", "^write_file$", _BLOCK_COMMAND, timeout_sec=15)
        result = hooks.run_hooks("PreToolUse", {"tool_name": "read_file", "args": {"path": "x"}})
        self.assertFalse(result["blocked"])

    def test_disabled_hook_is_skipped(self) -> None:
        added = hooks.hook_add("PreToolUse", "^write_file$", _BLOCK_COMMAND, timeout_sec=15)
        hooks.hook_set_enabled(added["hook"]["id"], False)
        result = hooks.run_hooks("PreToolUse", {"tool_name": "write_file", "args": {}})
        self.assertFalse(result["blocked"])

    def test_posttooluse_hook_never_blocks_even_if_it_says_so(self) -> None:
        hooks.hook_add("PostToolUse", "", _BLOCK_COMMAND, timeout_sec=15)
        result = hooks.run_hooks("PostToolUse", {"tool_name": "write_file", "args": {}, "result": "ok"})
        self.assertFalse(result["blocked"])

    def test_hook_remove(self) -> None:
        added = hooks.hook_add("SessionStart", "", _NOOP_COMMAND, timeout_sec=15)
        removed = hooks.hook_remove(added["hook"]["id"])
        self.assertTrue(removed["ok"])
        self.assertEqual(hooks.load_hooks(), [])


if __name__ == "__main__":
    unittest.main()
