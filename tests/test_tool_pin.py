"""An explicit tool pin must beat the scorer, and must fail loudly when it is wrong.

Why this exists: the scorer binds ~86 of 113 tools for a prompt whose whole content is "grep
one file twice", because generic verbs score several specialist groups over threshold. A tool
set that size makes models corrupt tool ARGUMENTS - measured on nemotron at temperature 0,
bound 1 tool it emits query='tsg_id' 3/3, bound 86 it emits 'tsk_id' 5/5, which matches nothing
and is then reported as a clean zero. A caller that knows what the task needs must be able to
say so, and a typo in that pin must not quietly fall back to the 86.
"""
import unittest

from harness_agent import subagent, tools_registry


# The bake-off task, verbatim in shape: one file, one tool, two calls.
GREP_TASK = (
    "Count matching lines in exactly one file: micro_apps/DEV/airs-home/js/x.js\n\n"
    "Call grep_files TWICE, each time with glob_pattern set to that exact path and regex=true:\n"
    "  call 1: query = tsg_id\n  call 2: query = [Pp]ermission\n\n"
    "Read the matches_total field of each result. Reply with ONE line: TSG=<n> PERM=<n>"
)


def names(tools):
    return {t.name for t in tools}


class ToolPinTests(unittest.TestCase):
    def test_scorer_overbinds_the_motivating_task(self):
        # The regression witness. If this ever drops on its own the pin is still correct,
        # but the number in every comment above stops being true.
        scored = tools_registry.select_tools(GREP_TASK)
        self.assertGreater(len(scored), 60, "scorer no longer overbinds; revisit the pin rationale")
        self.assertIn("browser_open", names(scored), "browser group was the clearest overbind")

    def test_pin_replaces_selection_including_core(self):
        pinned = tools_registry.select_tools(GREP_TASK, groups=["filesystem"])
        self.assertEqual(names(pinned), names(tools_registry.FILESYSTEM_TOOLS))
        self.assertIn("grep_files", names(pinned))
        # The whole point: core's 38 tools are NOT added on top of the pin.
        self.assertNotIn("memory_save", names(pinned))
        self.assertLess(len(pinned), len(tools_registry.CORE_TOOLS))

    def test_pin_beats_a_prompt_that_scores_elsewhere(self):
        # Prompt is all browser words; the pin must still win.
        pinned = tools_registry.select_tools(
            "Open the browser, click login, screenshot the page", groups=["filesystem"]
        )
        self.assertEqual(names(pinned), names(tools_registry.FILESYSTEM_TOOLS))

    def test_multiple_groups_and_aliases(self):
        pinned = tools_registry.select_tools(GREP_TASK, groups=["filesystem", "browser"])
        self.assertIn("grep_files", names(pinned))
        self.assertIn("browser_open", names(pinned))
        # Aliases resolve to the same canonical group.
        self.assertEqual(
            names(tools_registry.select_tools("", groups=["subagent"])),
            names(tools_registry.select_tools("", groups=["delegation"])),
        )

    def test_unknown_group_raises_rather_than_falling_back(self):
        with self.assertRaises(ValueError) as ctx:
            tools_registry.select_tools(GREP_TASK, groups=["filesytem"])
        self.assertIn("filesytem", str(ctx.exception))
        self.assertIn("filesystem", str(ctx.exception), "error should list the valid names")

    def test_empty_pin_means_not_pinned_not_zero_tools(self):
        for empty in ([], None, ()):
            with self.subTest(empty=empty):
                self.assertIsNone(tools_registry.resolve_pinned_groups(empty))
                self.assertEqual(
                    names(tools_registry.select_tools(GREP_TASK, groups=empty)),
                    names(tools_registry.select_tools(GREP_TASK)),
                )

    def test_group_names_are_reported_for_the_ledger(self):
        # task_shape buckets runs by the groups they bound; a pinned run must report the pin,
        # not what the scorer would have picked.
        self.assertEqual(
            tools_registry.select_group_names(GREP_TASK, groups=["filesystem"]), ["filesystem"]
        )

    def test_pin_is_stable_across_retry_feedback(self):
        # select_tools_for_messages is what the runtime calls every turn. A pinned run must not
        # drift when a rejection string becomes the latest human message.
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [
            HumanMessage(content=GREP_TASK),
            AIMessage(content="TSG=0 PERM=0"),
            HumanMessage(content="Your previous answer was not accepted: answer too short."),
        ]
        pinned = tools_registry.select_tools_for_messages(messages, groups=["filesystem"])
        self.assertEqual(names(pinned), names(tools_registry.FILESYSTEM_TOOLS))


class ToolPinParsingTests(unittest.TestCase):
    def test_comma_separated_parsing(self):
        self.assertEqual(subagent._parse_tool_groups("filesystem, browser"), ["filesystem", "browser"])
        self.assertEqual(subagent._parse_tool_groups("  filesystem  "), ["filesystem"])

    def test_blank_is_not_a_pin(self):
        for blank in ("", "   ", ",", " , "):
            with self.subTest(blank=blank):
                self.assertIsNone(subagent._parse_tool_groups(blank))

    def test_bad_name_raises_at_parse_time(self):
        # Must raise HERE, not inside the runtime factory: the attempt loop treats a raise as
        # a failed attempt and escalates models, so a typo would burn the attempt budget and
        # surface as a model failure.
        with self.assertRaises(ValueError):
            subagent._parse_tool_groups("filesystem,nope")

    def test_entry_point_returns_a_clean_error_for_a_bad_pin(self):
        out = subagent.spawn_subagent.invoke({"task": "anything", "tool_groups": "nope"})
        self.assertIn('"ok": false', out.lower())
        self.assertIn("nope", out)


if __name__ == "__main__":
    unittest.main()
