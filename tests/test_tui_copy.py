import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from textual.widgets import Markdown

from harness_agent import core, terminal_ui
from harness_agent.sessions import Session
from harness_agent.tui_app import ChatApp


class _FakeSessionManager:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runtime = SimpleNamespace(history=[])

    def open_sessions(self) -> list[Session]:
        return [self.session]

    def set_active(self, session_id: str) -> bool:
        return session_id == self.session.id

    def get_session(self, session_id: str):
        if session_id != self.session.id:
            return None
        return self.session, self.runtime


class TextualCopyTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="harness_tui_copy_test_"))
        core.set_workspace(self._tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_selecting_assistant_markdown_copies_to_clipboard(self) -> None:
        async def run_probe() -> None:
            session = Session(
                id="copytestsession",
                name="copy-test",
                model_name="test-model",
                workspace=str(core.WORKSPACE),
                created_at=core.now_str(),
                updated_at=core.now_str(),
            )
            app = ChatApp(_FakeSessionManager(session), session, {})
            async with app.run_test(size=(100, 30)) as pilot:
                await terminal_ui._write_assistant_message_history(
                    "Copyable assistant text\n\n- with markdown",
                    session_id=session.id,
                )
                await pilot.pause()
                markdown = app.query_one(Markdown)
                await pilot.mouse_down(markdown, offset=(2, 0))
                await pilot.mouse_up(markdown, offset=(19, 0))
                await pilot.pause()

                self.assertIn("Copyable assistant", app._clipboard)

        asyncio.run(run_probe())


if __name__ == "__main__":
    unittest.main()
