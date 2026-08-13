"""Tool selection must not drift off the original task during a delegated run.

Selection is scored off the latest human message. In a delegated run the later human
messages are machinery - a retry's rejection feedback, a resume's follow-up - and scoring
those alone drops the task's own tool groups mid-run. These tests pin that behavior down
in both directions: interactive follows the newest message, delegated does not lose the
task's tools.
"""
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from harness_agent import tools_registry


BROWSER_TASK = (
    "Open the website https://example.com in the browser, click the login button, "
    "and screenshot the page so I can see what a visitor sees."
)
# What subagent._run_with_validation actually sends on a rejected attempt. Nothing in it
# scores as browser work.
RETRY_FEEDBACK = (
    "Your previous answer was not accepted: answer shorter than 500 characters.\n\n"
    "Revise or continue so the answer fully satisfies the original task, then return the "
    "corrected complete answer."
)


def names(tools):
    return {t.name for t in tools}


class ToolSelectionPinningTests(unittest.TestCase):
    def setUp(self):
        self.browser_names = names(tools_registry.SPECIALIST_GROUPS["browser"].tools)
        self.assertTrue(self.browser_names, "browser group should not be empty")

    def _select(self, messages, pin_to_first):
        # profile="auto" explicitly: HARNESS_TOOL_PROFILE=full in the environment would
        # bind everything and make this test vacuously pass.
        return names(
            tools_registry.select_tools_for_messages(
                messages, profile="auto", pin_to_first=pin_to_first
            )
        )

    def test_first_attempt_selects_browser_tools(self):
        messages = [SystemMessage(content="sys"), HumanMessage(content=BROWSER_TASK)]
        self.assertTrue(self.browser_names & self._select(messages, pin_to_first=False))
        self.assertTrue(self.browser_names & self._select(messages, pin_to_first=True))

    def test_retry_drops_browser_tools_when_unpinned(self):
        """The bug, kept as a regression witness: unpinned selection loses the task."""
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content=BROWSER_TASK),
            AIMessage(content="too short"),
            HumanMessage(content=RETRY_FEEDBACK),
        ]
        self.assertFalse(self.browser_names & self._select(messages, pin_to_first=False))

    def test_retry_keeps_browser_tools_when_pinned(self):
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content=BROWSER_TASK),
            AIMessage(content="too short"),
            HumanMessage(content=RETRY_FEEDBACK),
        ]
        self.assertTrue(self.browser_names & self._select(messages, pin_to_first=True))

    def test_resume_keeps_task_tools_and_adds_the_new_turn(self):
        """A resume asks something new; the run should gain tools, not trade them away."""
        messages = [
            SystemMessage(content="sys"),
            HumanMessage(content=BROWSER_TASK),
            AIMessage(content="done"),
            HumanMessage(content="Now run a graphql introspection query against the api schema."),
        ]
        selected = self._select(messages, pin_to_first=True)
        graphql_names = names(tools_registry.SPECIALIST_GROUPS["graphql"].tools)
        self.assertTrue(self.browser_names & selected, "lost the original task's tools")
        self.assertTrue(graphql_names & selected, "ignored the resume message")

    def test_no_human_messages_is_not_an_error(self):
        self._select([SystemMessage(content="sys")], pin_to_first=True)


if __name__ == "__main__":
    unittest.main()
