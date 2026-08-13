import json
import shutil
import tempfile
import unittest
from pathlib import Path

from harness_agent import core, tasks


def _payload(tool_result: str) -> dict:
    return json.loads(tool_result)


class SplitTaskLinesTests(unittest.TestCase):
    def test_strips_numbered_and_bulleted_markers(self) -> None:
        text = "1. first task\n2) second task\n- third\n* fourth\n[ ] fifth\n[x] sixth"
        self.assertEqual(
            tasks._split_task_lines(text),
            ["first task", "second task", "third", "fourth", "fifth", "sixth"],
        )

    def test_blank_lines_dropped(self) -> None:
        self.assertEqual(tasks._split_task_lines("a\n\n\n  \nb"), ["a", "b"])

    def test_single_line_semicolons_split(self) -> None:
        self.assertEqual(tasks._split_task_lines("a; b ; c"), ["a", "b", "c"])

    def test_multiline_semicolon_not_shattered(self) -> None:
        # A stray semicolon inside one of several real lines must not split that line.
        self.assertEqual(
            tasks._split_task_lines("run x; then y\nsecond task"),
            ["run x; then y", "second task"],
        )

    def test_midsentence_hyphen_or_number_preserved(self) -> None:
        self.assertEqual(
            tasks._split_task_lines("re-run the 3-step flow"),
            ["re-run the 3-step flow"],
        )


class NextIdMakerTests(unittest.TestCase):
    def test_ids_avoid_collision_with_existing(self) -> None:
        make = tasks._next_id_maker([{"id": "t1"}, {"id": "t2"}])
        self.assertEqual([make(), make()], ["t3", "t4"])


class TodoBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="harness_tasks_test_"))
        core.set_workspace(self._tmp)
        tasks.set_active_scope("test")

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_build_creates_pending_todos_readable_by_todo_read(self) -> None:
        out = _payload(tasks.todo_build.invoke({"tasks_text": "1. alpha\n2. beta"}))
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "split")
        self.assertEqual(out["added"], 2)
        self.assertEqual([t["content"] for t in out["todos"]], ["alpha", "beta"])
        self.assertTrue(all(t["status"] == "pending" for t in out["todos"]))
        self.assertEqual(len({t["id"] for t in out["todos"]}), 2)

        read = _payload(tasks.todo_read.invoke({}))
        self.assertEqual(read["count"], 2)

    def test_append_continues_without_replacing_or_colliding(self) -> None:
        tasks.todo_build.invoke({"tasks_text": "alpha\nbeta"})
        out = _payload(tasks.todo_build.invoke({"tasks_text": "gamma", "append": True}))
        self.assertEqual(out["count"], 3)
        self.assertEqual([t["content"] for t in out["todos"]], ["alpha", "beta", "gamma"])
        self.assertEqual(len({t["id"] for t in out["todos"]}), 3)

    def test_no_append_replaces(self) -> None:
        tasks.todo_build.invoke({"tasks_text": "alpha\nbeta"})
        out = _payload(tasks.todo_build.invoke({"tasks_text": "gamma"}))
        self.assertEqual(out["count"], 1)
        self.assertEqual(out["todos"][0]["content"], "gamma")

    def test_empty_input_errors(self) -> None:
        out = _payload(tasks.todo_build.invoke({"tasks_text": "   \n  "}))
        self.assertFalse(out["ok"])

    def test_decompose_falls_back_to_split_when_model_unavailable(self) -> None:
        # No router is running in the test env, so decompose must degrade to the
        # deterministic split rather than failing.
        out = _payload(tasks.todo_build.invoke({"tasks_text": "alpha\nbeta", "decompose": True}))
        self.assertTrue(out["ok"])
        self.assertEqual(out["source"], "split")
        self.assertIn("decompose_fallback", out)
        self.assertEqual([t["content"] for t in out["todos"]], ["alpha", "beta"])


class RegistryTests(unittest.TestCase):
    def test_todo_build_registered_for_mcp(self) -> None:
        from harness_agent.tools_registry import TOOL_BY_NAME

        self.assertIn("todo_build", TOOL_BY_NAME)


if __name__ == "__main__":
    unittest.main()
