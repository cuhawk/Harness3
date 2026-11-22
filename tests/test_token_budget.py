import unittest
import json

from harness_agent import agent_runtime, tools_registry


class ToolSelectionBudgetTests(unittest.TestCase):
    def test_auto_profile_keeps_simple_filesystem_prompt_lean(self) -> None:
        selected = tools_registry.select_tools(
            "list dir and look for updates in the src folder",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertLess(len(selected), len(tools_registry.ALL_TOOLS))
        self.assertIn("list_dir", names)
        self.assertIn("find_files", names)
        self.assertIn("grep_files", names)
        self.assertIn("run_terminal", names)
        self.assertIn("git_command", names)
        self.assertNotIn("browser_open", names)
        self.assertNotIn("graphql_introspect", names)
        self.assertNotIn("ssh_run", names)

    def test_full_profile_keeps_every_tool_available(self) -> None:
        selected = tools_registry.select_tools("list dir", profile="full")

        self.assertEqual(
            [tool.name for tool in selected],
            [tool.name for tool in tools_registry.ALL_TOOLS],
        )

    def test_smart_selector_routes_visual_site_work_to_browser_tools(self) -> None:
        selected = tools_registry.select_tools(
            "open https://example.com and inspect what the visitor sees",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertIn("browser_open", names)
        self.assertIn("browser_screenshot", names)
        self.assertIn("start_process_instance", names)
        self.assertNotIn("graphql_introspect", names)

    def test_smart_selector_routes_introspection_schema_work_to_graphql_tools(self) -> None:
        selected = tools_registry.select_tools(
            "fetch the API type map from the endpoint and find User fields",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertIn("graphql_introspect", names)
        self.assertIn("graphql_schema_search", names)
        self.assertIn("rag_search", names)
        self.assertNotIn("browser_open", names)

    def test_smart_selector_routes_mcp_tool_cache_work_to_mcp_tools(self) -> None:
        selected = tools_registry.select_tools(
            "discover callable capabilities from the registered external server",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertIn("mcp_refresh_tools", names)
        self.assertIn("mcp_search_tools", names)
        self.assertIn("start_process_instance", names)
        self.assertNotIn("browser_open", names)

    def test_smart_selector_routes_ssh_upload_work_to_ssh_tools(self) -> None:
        selected = tools_registry.select_tools(
            "copy this archive to the bastion host",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertIn("ssh_upload", names)
        self.assertIn("ssh_run", names)
        self.assertNotIn("browser_open", names)

    def test_smart_selector_routes_scheduled_repeated_work_to_automation_tools(self) -> None:
        selected = tools_registry.select_tools(
            "make this workflow repeat each morning",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertIn("automation_schedule", names)
        self.assertIn("workflow_run", names)
        self.assertNotIn("graphql_introspect", names)

    def test_smart_selector_routes_runbook_work_to_workflow_tools(self) -> None:
        selected = tools_registry.select_tools(
            "turn these steps into a reusable command sequence",
            profile="auto",
        )
        names = {tool.name for tool in selected}

        self.assertIn("workflow_save", names)
        self.assertIn("workflow_run", names)
        self.assertNotIn("automation_schedule", names)


class ToolResultBudgetTests(unittest.TestCase):
    def test_short_tool_result_is_left_unchanged_for_model(self) -> None:
        content = '{"ok": true}'

        self.assertEqual(agent_runtime.compact_tool_result_for_model(content), content)

    def test_large_tool_result_is_truncated_for_model(self) -> None:
        content = "x" * 200

        compacted = agent_runtime.compact_tool_result_for_model(content, limit=50)
        payload = json.loads(compacted)

        self.assertTrue(payload["tool_result_compacted"])
        self.assertEqual(payload["original_bytes"], 200)
        self.assertEqual(payload["limit_bytes"], 50)
        self.assertIn("text_preview", payload)

    def test_large_json_tool_result_stays_valid_json_for_model(self) -> None:
        content = json.dumps({
            "ok": True,
            "path": "big.txt",
            "offset": 0,
            "bytes_read": 200,
            "file_size_bytes": 400,
            "content": "a" * 200,
        })

        compacted = agent_runtime.compact_tool_result_for_model(content, limit=80)
        payload = json.loads(compacted)

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["tool_result_compacted"])
        self.assertEqual(payload["path"], "big.txt")
        self.assertEqual(payload["offset"], 0)
        self.assertEqual(payload["file_size_bytes"], 400)
        self.assertIn("content_preview", payload)
        self.assertNotIn("content", payload)
        self.assertEqual(payload["continue_with"]["tool"], "read_file")
        self.assertEqual(payload["continue_with"]["args"]["path"], "big.txt")

    def test_large_json_tool_result_keeps_bounded_entries_preview(self) -> None:
        content = json.dumps({
            "ok": True,
            "path": "sample-project",
            "count": 500,
            "truncated": True,
            "entries": [
                {
                    "path": f"sample-project\\item-{i}",
                    "absolute_path": f"C:\\workspace\\sample-project\\item-{i}",
                    "exists": True,
                    "type": "directory",
                    "size_bytes": 4096,
                    "modified": "2026-07-09 10:00:00",
                }
                for i in range(80)
            ],
        })

        compacted = agent_runtime.compact_tool_result_for_model(content, limit=1200)
        payload = json.loads(compacted)

        self.assertTrue(payload["tool_result_compacted"])
        self.assertIn("entries", payload)
        self.assertGreater(len(payload["entries"]), 0)
        self.assertLess(len(payload["entries"]), 80)
        self.assertEqual(payload["entries_original_count"], 80)
        self.assertEqual(payload["entries_preview_count"], len(payload["entries"]))
        self.assertEqual(payload["entries"][0]["path"], "sample-project\\item-0")


if __name__ == "__main__":
    unittest.main()
