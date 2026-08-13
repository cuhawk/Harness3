import shutil
import tempfile
import unittest
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from harness_agent import core, sessions


class SerializeHistoryRoundTripTests(unittest.TestCase):
    def test_round_trips_every_message_type(self) -> None:
        original = [
            SystemMessage(content="you are a helpful agent"),
            HumanMessage(content="list the files"),
            AIMessage(
                content="",
                tool_calls=[{"name": "list_dir", "args": {"path": "."}, "id": "call_1"}],
                additional_kwargs={"reasoning_content": "I should list the directory first"},
            ),
            ToolMessage(content='{"ok": true}', name="list_dir", tool_call_id="call_1"),
            AIMessage(content="Here are the files."),
        ]

        data = sessions.serialize_history(original)
        restored = sessions.deserialize_history(data)

        self.assertEqual(len(restored), len(original))
        self.assertIsInstance(restored[0], SystemMessage)
        self.assertEqual(restored[0].content, "you are a helpful agent")
        self.assertIsInstance(restored[1], HumanMessage)
        self.assertEqual(restored[1].content, "list the files")
        self.assertIsInstance(restored[2], AIMessage)
        self.assertEqual(restored[2].tool_calls, original[2].tool_calls)
        self.assertEqual(
            restored[2].additional_kwargs.get("reasoning_content"),
            "I should list the directory first",
        )
        self.assertIsInstance(restored[3], ToolMessage)
        self.assertEqual(restored[3].content, '{"ok": true}')
        self.assertEqual(restored[3].name, "list_dir")
        self.assertEqual(restored[3].tool_call_id, "call_1")
        self.assertIsInstance(restored[4], AIMessage)
        self.assertEqual(restored[4].content, "Here are the files.")

    def test_serialized_form_is_json_safe(self) -> None:
        import json

        data = sessions.serialize_history([HumanMessage(content="hi")])
        # Must not raise - this is exactly what SessionStore writes to disk.
        json.dumps(data)

    def test_tool_message_artifact_round_trips_full_response(self) -> None:
        original = [
            ToolMessage(
                content='{"tool_result_compacted": true}',
                name="read_file",
                tool_call_id="call_1",
                artifact={
                    "full_content": '{"ok": true, "content": "full raw response"}',
                    "original_bytes": 45,
                },
            ),
        ]

        data = sessions.serialize_history(original)
        restored = sessions.deserialize_history(data)

        self.assertEqual(
            data[0]["artifact"]["full_content"],
            '{"ok": true, "content": "full raw response"}',
        )
        self.assertEqual(restored[0].content, '{"tool_result_compacted": true}')
        self.assertEqual(
            restored[0].artifact["full_content"],
            '{"ok": true, "content": "full raw response"}',
        )

    def test_unknown_message_type_is_skipped_on_deserialize(self) -> None:
        data = [{"type": "system", "content": "ok"}, {"type": "something_new", "content": "?"}]
        restored = sessions.deserialize_history(data)
        self.assertEqual(len(restored), 1)
        self.assertIsInstance(restored[0], SystemMessage)


class SessionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = Path(tempfile.mkdtemp(prefix="harness_session_test_"))
        core.set_workspace(self._tmp)
        self.store = sessions.SessionStore()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_create_then_list_round_trip(self) -> None:
        session = self.store.create("smoke-test", "some-model")
        found = [s for s in self.store.list() if s.id == session.id]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "smoke-test")

    def test_save_and_load_history(self) -> None:
        session = self.store.create("with-history", "some-model")
        history = [SystemMessage(content="sys"), HumanMessage(content="hello")]
        self.store.save(session.id, session, history)

        loaded_session, raw_history = self.store.load(session.id)
        self.assertEqual(loaded_session.id, session.id)
        restored = sessions.deserialize_history(raw_history)
        self.assertEqual([m.content for m in restored], ["sys", "hello"])

    def test_rename_and_delete(self) -> None:
        session = self.store.create("original-name", "some-model")
        renamed = self.store.rename(session.id, "new-name")
        self.assertEqual(renamed.name, "new-name")
        self.assertTrue(self.store.delete(session.id))
        self.assertFalse(self.store.delete(session.id))


if __name__ == "__main__":
    unittest.main()
